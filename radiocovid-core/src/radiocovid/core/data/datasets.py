# MIT License
#
# Copyright (c) 2025 @CedrickArmel, @samarita22, @TaxelleT & @Yeyecodes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import torch
from radiocovid.core.utils import RankedLogger
from torch.utils.data import Dataset, Sampler, Subset
from torchvision.datasets import DatasetFolder

log = RankedLogger(__name__, rank_zero_only=True)


class RadioCovidDataset(DatasetFolder):
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return dict(
            id=int(sha256(Path(path).stem.encode()).hexdigest()[:8], 16),
            input=sample,
            target=target,
        )


class RadioCovidSubset(Subset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super().__init__(dataset, indices)
        self._transform = transform
        self._target_transform = target_transform

    def __getitem__(self, idx):
        if isinstance(idx, list):
            output = []
            for i in idx:
                item = super().__getitem__(i)
                if self._transform is not None:
                    item["input"] = self._transform(item["input"])
                if self._target_transform is not None:
                    item["target"] = self._target_transform(item["target"])
                output.append(item)
            return output

        item = super().__getitem__(idx)

        if self._transform is not None:
            item["input"] = self._transform(item["input"])

        if self._target_transform is not None:
            item["target"] = self._target_transform(item["target"])
        return item

    def __getitems__(self, indices: list[int]):
        return self.__getitem__(indices)


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        batch_size: int,
        num_samples: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        replacement: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ):

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.dataset = dataset
        self.n = len(dataset)
        self.batch_size = batch_size

        if num_replicas is None:
            num_replicas = 1

        if rank is None:
            rank = 0

        if num_samples is None:
            num_samples = self.n

        self.global_num_samples = num_samples

        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.shuffle = shuffle
        self.base_seed = (
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )
        self.epoch = 0

        self.batch_size_eff = self.batch_size * self.num_replicas
        self.total_size = (
            math.ceil(self.global_num_samples / self.batch_size_eff)
            * self.batch_size_eff
        )
        # self.total_size = self.global_num_samples + (self.batch_size_eff - (self.batch_size_eff - (self.global_num_samples % self.batch_size_eff)))
        self.num_samples_per_rank = self.total_size // self.num_replicas  # per rank

    def set_epoch(self, epoch: int) -> None:
        log.info("Setting epoch in DistributedWeightedSampler")
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.global_num_samples,
            self.replacement,
            generator=g,
        ).tolist()

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank :: self.num_replicas]
        assert len(indices) == self.num_samples_per_rank
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank


class PaddingSampler(Sampler):
    def __init__(self, dataset, target_size, shuffle):
        super().__init__()
        self.dataset = dataset
        self.target_size = target_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.target_size > n:
            padding = indices[: self.target_size - n]
            indices += padding
        return iter(indices)

    def __len__(self):
        return self.target_size


class PaddedShardedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        if num_replicas is None:
            num_replicas = 1

        if rank is None:
            rank = 0

        self.dataset = dataset
        self.n = len(dataset)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.batch_size_eff = self.batch_size * self.num_replicas
        self.total_size = math.ceil(self.n / self.batch_size_eff) * self.batch_size_eff
        self.num_samples = self.total_size // self.num_replicas  # per rank

    def set_epoch(self, epoch: int) -> None:
        log.info("Setting epoch in PaddedShardedSampler")
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.n, generator=g).tolist()
        else:
            indices = list(range(self.n))

        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]

        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        indices = indices[self.rank :: self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def get_label_from_sample(path: str) -> int:
    d = {"normal": 0, "lung_opacity": 1, "covid": 2, "viral pneumonia": 3}
    return d[Path(path).stem.split("-")[0].lower()]
