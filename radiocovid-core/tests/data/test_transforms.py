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
from radiocovid.core.data.transforms import Squeeze


class TestSqueeze:
    def test_no_dim_squeezes_all_singletons(self):
        t = torch.randn(1, 3, 1, 4)
        result = Squeeze()(t, {})
        assert result.shape == (3, 4)

    def test_with_dim_squeezes_only_that_dim(self):
        t = torch.randn(1, 3, 1, 4)
        result = Squeeze(dim=0)(t, {})
        assert result.shape == (3, 1, 4)

    def test_no_singleton_dims_unchanged(self):
        t = torch.randn(2, 3, 4)
        result = Squeeze()(t, {})
        assert result.shape == (2, 3, 4)

    def test_already_squeezed(self):
        t = torch.randn(3)
        result = Squeeze()(t, {})
        assert result.shape == (3,)
