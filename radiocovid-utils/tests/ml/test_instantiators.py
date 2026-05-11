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

import functools

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from radiocovid.utils.ml.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    sequential_scheduler,
)
from torch.optim.lr_scheduler import SequentialLR


class TestInstantiateCallbacks:
    def test_none_cfg_returns_none(self):
        assert instantiate_callbacks(None) is None

    def test_empty_cfg_returns_none(self):
        assert instantiate_callbacks(OmegaConf.create({})) is None

    def test_non_dictconfig_raises(self):
        with pytest.raises(TypeError):
            instantiate_callbacks({"cb": {"_target_": "torch.nn.Dropout"}})

    def test_valid_target_instantiated(self):
        cfg = OmegaConf.create({"cb1": {"_target_": "torch.nn.Dropout", "p": 0.1}})
        result = instantiate_callbacks(cfg)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], nn.Dropout)

    def test_entry_without_target_skipped(self):
        cfg = OmegaConf.create(
            {
                "cb1": {"_target_": "torch.nn.Dropout", "p": 0.1},
                "cb2": {"p": 0.5},
            }
        )
        result = instantiate_callbacks(cfg)
        assert len(result) == 1

    def test_multiple_targets_all_instantiated(self):
        cfg = OmegaConf.create(
            {
                "cb1": {"_target_": "torch.nn.Dropout", "p": 0.1},
                "cb2": {"_target_": "torch.nn.ReLU"},
            }
        )
        result = instantiate_callbacks(cfg)
        assert len(result) == 2


class TestInstantiateLoggers:
    def test_none_cfg_returns_none(self):
        assert instantiate_loggers(None) is None

    def test_empty_cfg_returns_none(self):
        assert instantiate_loggers(OmegaConf.create({})) is None

    def test_non_dictconfig_raises(self):
        with pytest.raises(TypeError):
            instantiate_loggers({"lg": {"_target_": "torch.nn.Dropout"}})

    def test_valid_target_instantiated(self):
        cfg = OmegaConf.create({"lg1": {"_target_": "torch.nn.Dropout", "p": 0.2}})
        result = instantiate_loggers(cfg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_entry_without_target_skipped(self):
        cfg = OmegaConf.create(
            {
                "lg1": {"_target_": "torch.nn.ReLU"},
                "lg2": {"some_key": "val"},
            }
        )
        result = instantiate_loggers(cfg)
        assert len(result) == 1


class TestSequentialScheduler:
    def _optimizer(self):
        return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)

    def test_returns_sequential_lr(self):
        opt = self._optimizer()
        schedulers = [
            functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
            functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5),
        ]
        result = sequential_scheduler(opt, schedulers, milestones=[2])
        assert isinstance(result, SequentialLR)

    def test_stepping_advances_schedule(self):
        opt = self._optimizer()
        schedulers = [
            functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.1),
            functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
        ]
        sched = sequential_scheduler(opt, schedulers, milestones=[1])
        lr_before = opt.param_groups[0]["lr"]
        sched.step()
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after != lr_before
