from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import lightning.pytorch as L
import numpy as np
import torch
from radiocovid.core.utils import (
    RankedLogger,
    get_seeded_generator,
    seed_worker,
    worker_balanced_n_samples,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms.v2 import Transform

from .datasets import RadioCovidSubset

log = RankedLogger(__name__, rank_zero_only=True)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        train_transform: Transform,
        eval_transform: Transform,
        train_loader: partial[DataLoader],
        eval_loader: partial[DataLoader],
        class_retriever: Callable | None = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.class_retriever = class_retriever
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.train_set: None | RadioCovidSubset = None
        self.val_set: None | RadioCovidSubset = None
        self.test_set: None | RadioCovidSubset = None

        self.train_idx = None
        self.test_idx = None
        self.val_idx = None

    def setup(self, stage: Optional[str] = None):

        if (
            (self.train_set is None)
            or (self.test_set is None)
            or (self.val_set is None)
        ):

            if self.test_size and self.val_size:
                fit_idx, self.test_idx = train_test_split(
                    range(len(self.dataset)),
                    test_size=self.test_size,
                    stratify=self.dataset.targets,
                    random_state=self.seed,
                )

                self.train_idx, self.val_idx = train_test_split(
                    fit_idx,
                    test_size=self.val_size,
                    stratify=np.array(self.dataset.targets)[
                        np.setdiff1d(range(len(self.dataset)), self.test_idx)
                    ],
                    random_state=self.seed,
                )

            elif self.test_size and not self.val_size:
                self.train_idx, self.test_idx = train_test_split(
                    range(len(self.dataset)),
                    test_size=self.test_size,
                    stratify=self.dataset.targets,
                    random_state=self.seed,
                )

            elif not self.test_size and self.val_size:
                self.train_idx, self.val_idx = train_test_split(
                    range(len(self.dataset)),
                    test_size=self.val_size,
                    stratify=self.dataset.targets,
                    random_state=self.seed,
                )
            if stage in ["fit", "validate"]:
                if (
                    self.trainer.limit_val_batches
                    and (self.val_idx is not None)
                    and (self.val_set is None)
                ):
                    self.val_set = RadioCovidSubset(
                        self.dataset,
                        indices=self.val_idx,
                        transform=self.eval_transform,
                    )

                if (
                    self.trainer.limit_train_batches
                    and (stage == "fit")
                    and (self.train_set is None)
                ):
                    if self.class_retriever:
                        targets = []
                        meta_labels = []
                        meta_set = defaultdict(lambda: set())
                        for c, t in self.dataset.samples:
                            meta_labels.append(self.class_retriever(path=c))
                            meta_set[t].add(Path(c).stem.split("-")[0])
                            targets.append(t)
                    else:
                        targets = self.dataset.targets

                    if self.train_idx is None:
                        self.train_set = RadioCovidSubset(
                            self.dataset,
                            indices=range(len(self.dataset)),
                            transform=self.train_transform,
                        )
                        counter = Counter(
                            meta_labels if self.class_retriever else targets
                        )
                    else:
                        targets = np.array(targets)[self.train_idx].tolist()
                        meta_labels = np.array(meta_labels)[self.train_idx].tolist()
                        self.train_set = RadioCovidSubset(
                            self.dataset,
                            indices=self.train_idx,
                            transform=self.train_transform,
                        )
                        counter = Counter(
                            meta_labels if self.class_retriever else targets
                        )

                    class_weights = {c: 1.0 / n for c, n in counter.items()}

                    if self.class_retriever:
                        meta_sample_weights = [
                            class_weights[int(t)] for t in meta_labels
                        ]
                        self.sample_weights = torch.tensor(
                            [
                                (1 / len(meta_set[t])) * meta_sample_weights[i]
                                for i, t in enumerate(targets)
                            ],
                            dtype=torch.double,
                        )
                    else:
                        self.sample_weights = torch.tensor(
                            [class_weights[int(t)] for t in targets], dtype=torch.double
                        )

            elif (
                self.trainer.limit_test_batches
                and (stage == "test")
                and (self.test_idx is not None)
                and (self.test_set is None)
            ):
                self.test_set = RadioCovidSubset(
                    self.dataset, indices=self.test_idx, transform=self.eval_transform
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        log.info(f"Train set size : {len(self.train_set)}")  # type: ignore[arg-type]
        if self.sample_weights is None:
            self.sample_weights = torch.ones(len(self.train_set))  # type: ignore[arg-type]
        n = worker_balanced_n_samples(len(self.train_set), self.train_loader.keywords["batch_size"], self.trainer.world_size)  # type: ignore[arg-type]
        log.info(f"Train set size after padding : {n}")
        sampler = WeightedRandomSampler(
            weights=self.sample_weights, num_samples=n, replacement=True
        )
        return self.train_loader(
            dataset=self.train_set,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=get_seeded_generator(self.seed),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        log.info(f"Validation set size : {len(self.val_set)}")  # type: ignore[arg-type]
        n = worker_balanced_n_samples(len(self.val_set), self.eval_loader.keywords["batch_size"], self.trainer.world_size)  # type: ignore[arg-type]
        log.info(f"Validation set size after padding : {n}")
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.val_set)), num_samples=n, replacement=True, generator=get_seeded_generator(self.seed))  # type: ignore[arg-type]
        return self.eval_loader(dataset=self.val_set, sampler=sampler, worker_init_fn=seed_worker, generator=get_seeded_generator(self.seed))  # type: ignore[arg-type]

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        log.info(f"Test set size : {len(self.val_set)}")  # type: ignore[arg-type]
        n = worker_balanced_n_samples(len(self.test_set), self.eval_loader.keywords["batch_size"], self.trainer.world_size)  # type: ignore[arg-type]
        log.info(f"Test set size after padding : {n}")
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.test_set)), num_samples=n, replacement=True, generator=get_seeded_generator(self.seed))  # type: ignore[arg-type]
        return self.eval_loader(dataset=self.test_set, sampler=sampler, worker_init_fn=seed_worker, generator=get_seeded_generator(self.seed))  # type: ignore[arg-type]
