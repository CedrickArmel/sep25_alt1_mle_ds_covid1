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

import os

import numpy as np
import pytest
import torch.nn as nn
from lightning_utilities.core.rank_zero import rank_zero_only
from PIL import Image

# Required by RankedLogger before any test runs
rank_zero_only.rank = 0

os.environ.setdefault("WANDB_MODE", "offline")

_CLASSES = ["covid", "normal"]


@pytest.fixture(scope="session")
def tmp_image_folder(tmp_path_factory):
    """Tiny synthetic ImageFolder: 2 classes, 3 RGB PNGs each."""
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    for cls in _CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(3):
            arr = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{cls}-{i:03d}.png")
    return root


@pytest.fixture
def fake_trainer():
    class FakeTrainer:
        limit_train_batches = 1
        limit_val_batches = 1
        limit_test_batches = 1
        world_size = 1
        logger = []
        loggers = []

    return FakeTrainer()


@pytest.fixture
def dummy_net():
    return nn.Linear(16, 2)
