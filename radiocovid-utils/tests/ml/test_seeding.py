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
import random

import numpy as np
import torch
from radiocovid.utils.ml.seeding import get_seeded_generator, set_seed


class TestGetSeededGenerator:
    def test_deterministic(self):
        g1 = get_seeded_generator(42)
        g2 = get_seeded_generator(42)
        t1 = torch.rand(5, generator=g1)
        t2 = torch.rand(5, generator=g2)
        assert torch.allclose(t1, t2)

    def test_large_seed_wraps(self):
        large = 2**33
        gen = get_seeded_generator(large)
        assert torch.rand(1, generator=gen).item() >= 0.0

    def test_different_seeds_differ(self):
        g1 = get_seeded_generator(0)
        g2 = get_seeded_generator(1)
        t1 = torch.rand(5, generator=g1)
        t2 = torch.rand(5, generator=g2)
        assert not torch.allclose(t1, t2)


class TestSetSeed:
    def test_sets_env_vars(self):
        set_seed(seed=7)
        assert os.environ.get("PYTHONHASHSEED") == "7"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_reproducible_torch(self):
        set_seed(42)
        a = torch.rand(3)
        set_seed(42)
        b = torch.rand(3)
        assert torch.allclose(a, b)

    def test_reproducible_numpy(self):
        set_seed(42)
        a = np.random.rand(3)
        set_seed(42)
        b = np.random.rand(3)
        assert np.allclose(a, b)

    def test_reproducible_random(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_seed_none_runs(self):
        set_seed(seed=None)
