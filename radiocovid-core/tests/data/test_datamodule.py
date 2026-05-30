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

from functools import partial
from unittest.mock import MagicMock

from radiocovid.core.data.datamodule import DataModule
from radiocovid.core.data.datasets import RadioCovidDataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader


def _make_dm(root, test_size=0.34, val_size=0.5):
    dataset = RadioCovidDataset(
        root=str(root), loader=default_loader, extensions=(".png",)
    )
    dm = DataModule(
        dataset=dataset,
        train_transform=None,
        eval_transform=None,
        train_loader=partial(DataLoader, batch_size=2),
        eval_loader=partial(DataLoader, batch_size=2),
        test_size=test_size,
        val_size=val_size,
        seed=42,
    )
    trainer = MagicMock()
    trainer.limit_train_batches = 1
    trainer.limit_val_batches = 1
    trainer.limit_test_batches = 1
    trainer.world_size = 1
    trainer.global_rank = 0
    dm.trainer = trainer
    return dm


# --------------------------------------------------------------------------- #
# setup                                                                        #
# --------------------------------------------------------------------------- #


class TestDataModuleSetup:
    def test_fit_creates_train_and_val_sets(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder)
        dm.setup("fit")
        assert dm.train_set is not None
        assert dm.val_set is not None

    def test_test_stage_creates_test_set(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder)
        dm.setup("test")
        assert dm.test_set is not None

    def test_all_three_splits_disjoint(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder)
        dm.setup("fit")
        dm2 = _make_dm(tmp_image_folder)
        dm2.setup("test")
        train = set(dm.train_idx)
        val = set(dm.val_idx)
        test = set(dm2.test_idx)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_splits_cover_all_samples(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder)
        dm.setup("fit")
        dm.setup("test")
        all_idx = set(dm.train_idx) | set(dm.val_idx) | set(dm.test_idx)
        assert all_idx == set(range(len(dm.dataset)))

    def test_only_test_size_sets_train_and_test_idx(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder, test_size=0.34, val_size=0.0)
        dm.setup("fit")
        assert dm.train_idx is not None
        assert dm.test_idx is not None
        assert dm.val_idx is None

    def test_only_val_size_sets_train_and_val_idx(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder, test_size=0.0, val_size=0.5)
        dm.setup("fit")
        assert dm.train_idx is not None
        assert dm.val_idx is not None
        assert dm.test_idx is None

    def test_sample_weights_computed_after_fit(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder)
        dm.setup("fit")
        assert dm.sample_weights is not None
        assert len(dm.sample_weights) == len(dm.train_set)

    def test_balanced_weights_sum_to_equal_per_class(self, tmp_image_folder):
        dm = _make_dm(tmp_image_folder)
        dm.setup("fit")
        # Both classes should have equal total weight (inverse-frequency weighting)
        targets = [dm.dataset.targets[i] for i in dm.train_idx]
        unique = set(targets)
        class_weight_sums = {}
        for cls in unique:
            class_weight_sums[cls] = sum(
                dm.sample_weights[i].item() for i, t in enumerate(targets) if t == cls
            )
        sums = list(class_weight_sums.values())
        assert abs(sums[0] - sums[1]) < 1e-6
