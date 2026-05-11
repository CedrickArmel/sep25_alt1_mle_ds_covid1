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

from radiocovid.utils.ml.distributed import (
    balance_data_world_size,
    get_process_group,
    worker_balanced_n_samples,
)


class TestWorkerBalancedNSamples:
    def test_already_divisible(self):
        assert worker_balanced_n_samples(100, 5, 4) == 100

    def test_pads_to_next_multiple(self):
        result = worker_balanced_n_samples(101, 5, 4)
        assert result > 101
        assert result % (5 * 4) == 0

    def test_worldsize_one(self):
        assert worker_balanced_n_samples(10, 3, 1) == 12

    def test_exact_batch(self):
        assert worker_balanced_n_samples(6, 6, 1) == 6


class TestBalanceDataWorldSize:
    def test_pads_to_multiple(self):
        data = [{"x": i} for i in range(7)]
        result = balance_data_world_size(data, batch=4, worldsize=2)
        assert len(result) % (4 * 2) == 0
        assert len(result) >= 7

    def test_no_op_when_aligned(self):
        data = [{"x": i} for i in range(8)]
        result = balance_data_world_size(data, batch=4, worldsize=2)
        assert len(result) == 8

    def test_returns_same_list_object(self):
        data = [{"x": i} for i in range(8)]
        result = balance_data_world_size(data, batch=4, worldsize=2)
        assert result is data


class TestGetProcessGroup:
    def test_returns_none_when_not_initialized(self):
        assert get_process_group() is None
