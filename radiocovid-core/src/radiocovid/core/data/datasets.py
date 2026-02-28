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

from hashlib import sha256
from pathlib import Path
from typing import Any

from torch.utils.data import Subset
from torchvision.datasets import DatasetFolder


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
            id=int(sha256(path.encode()).hexdigest()[:8], 16),
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


def get_label_from_sample(path: str) -> int:
    d = {"normal": 0, "lung_opacity": 1, "covid": 2, "viral pneumonia": 3}
    return d[Path(path).stem.split("-")[0].lower()]
