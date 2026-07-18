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
import torch
import torch.nn as nn
from PIL import Image

os.environ.setdefault("WANDB_MODE", "offline")


@pytest.fixture
def tmp_png(tmp_path):
    """Single 32×32 RGB PNG."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def tiny_net():
    """Tiny Linear(3, 2) whose state_dict matches a saved ckpt with net.* keys."""
    return nn.Linear(3, 2)


@pytest.fixture
def tiny_ckpt(tmp_path, tiny_net):
    """torch.save'd ckpt dict with net.* prefixed keys + one unrelated key."""
    state = {f"net.{k}": v.clone() for k, v in tiny_net.state_dict().items()}
    state["other.weight"] = torch.zeros(1)
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": state}, path)
    return path, tiny_net.state_dict()
