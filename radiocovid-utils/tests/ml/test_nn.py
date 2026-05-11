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

import torch
import torch.nn as nn
from radiocovid.utils.ml.nn import initialize_weights


class TestInitializeWeights:
    def test_linear_xavier_normal(self):
        m = nn.Linear(8, 16)
        nn.init.zeros_(m.weight)
        initialize_weights(m, dist="normal")
        assert not torch.all(m.weight == 0)

    def test_linear_xavier_uniform(self):
        m = nn.Linear(8, 16)
        nn.init.zeros_(m.weight)
        initialize_weights(m, dist="uniform")
        assert not torch.all(m.weight == 0)

    def test_conv2d_kaiming_normal(self):
        m = nn.Conv2d(3, 8, 3)
        nn.init.zeros_(m.weight)
        initialize_weights(m, dist="normal")
        assert not torch.all(m.weight == 0)

    def test_conv2d_kaiming_uniform(self):
        m = nn.Conv2d(3, 8, 3)
        nn.init.zeros_(m.weight)
        initialize_weights(m, dist="uniform")
        assert not torch.all(m.weight == 0)

    def test_batchnorm_untouched(self):
        m = nn.BatchNorm2d(8)
        original_weight = m.weight.clone()
        initialize_weights(m, dist="normal")
        assert torch.allclose(m.weight, original_weight)
