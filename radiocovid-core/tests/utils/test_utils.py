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
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from radiocovid.core.utils.utils import (
    balance_data_world_size,
    extras,
    flatten_dict,
    get_metric_value,
    get_process_group,
    get_seeded_generator,
    initialize_weights,
    set_seed,
    task_wrapper,
    worker_balanced_n_samples,
)

# --------------------------------------------------------------------------- #
# flatten_dict                                                                 #
# --------------------------------------------------------------------------- #


class TestFlattenDict:
    def test_nested(self):
        d = {"a": {"b": {"c": 1}}}
        assert flatten_dict(d) == {"a.b.c": 1}

    def test_custom_sep(self):
        d = {"a": {"b": 1}}
        assert flatten_dict(d, sep="/") == {"a/b": 1}

    def test_empty(self):
        assert flatten_dict({}) == {}

    def test_already_flat(self):
        d = {"x": 1, "y": 2}
        assert flatten_dict(d) == d

    def test_mixed_depth(self):
        d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = flatten_dict(d)
        assert result == {"a": 1, "b.c": 2, "b.d.e": 3}


# --------------------------------------------------------------------------- #
# get_seeded_generator                                                         #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# worker_balanced_n_samples                                                    #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# balance_data_world_size                                                      #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# initialize_weights                                                           #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# set_seed                                                                     #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# get_metric_value                                                             #
# --------------------------------------------------------------------------- #


class TestGetMetricValue:
    def test_none_name_returns_none(self):
        assert get_metric_value({}, None) is None

    def test_missing_key_raises(self):
        with pytest.raises(Exception, match="Metric value not found"):
            get_metric_value({"a": torch.tensor(1.0)}, "b")

    def test_returns_float(self):
        val = get_metric_value({"loss": torch.tensor(0.5)}, "loss")
        assert isinstance(val, float)
        assert pytest.approx(val) == 0.5


# --------------------------------------------------------------------------- #
# get_process_group                                                            #
# --------------------------------------------------------------------------- #


class TestGetProcessGroup:
    def test_returns_none_when_not_initialized(self):
        assert get_process_group() is None


# --------------------------------------------------------------------------- #
# task_wrapper                                                                 #
# --------------------------------------------------------------------------- #


class TestTaskWrapper:
    def _cfg(self):
        return OmegaConf.create({"paths": {"output_dir": "/tmp"}})

    def test_success_returns_dicts(self):
        cfg = self._cfg()

        @task_wrapper
        def dummy(cfg):
            return {"loss": torch.tensor(1.0)}, {"model": None}

        metric_dict, object_dict = dummy(cfg)
        assert "loss" in metric_dict
        assert "model" in object_dict

    def test_re_raises_on_exception(self):
        cfg = self._cfg()

        @task_wrapper
        def bad(cfg):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            bad(cfg)


# --------------------------------------------------------------------------- #
# extras                                                                       #
# --------------------------------------------------------------------------- #


class TestExtras:
    def test_no_extras_key_returns_early(self):
        cfg = OmegaConf.create({})
        extras(cfg)  # should not raise

    def test_ignore_warnings(self):
        cfg = OmegaConf.create({"extras": {"ignore_warnings": True}})
        extras(cfg)

    def test_print_config_dispatches(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "radiocovid.core.utils.utils.print_config_tree",
            lambda *a, **kw: called.update({"called": True}),
        )
        cfg = OmegaConf.create({"extras": {"print_config": True}})
        extras(cfg)
        assert called.get("called")

    def test_enforce_tags_dispatches(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "radiocovid.core.utils.utils.enforce_tags",
            lambda *a, **kw: called.update({"called": True}),
        )
        cfg = OmegaConf.create({"extras": {"enforce_tags": True}})
        extras(cfg)
        assert called.get("called")
