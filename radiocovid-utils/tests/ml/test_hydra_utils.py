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

import pytest
import torch
from omegaconf import OmegaConf
from radiocovid.utils.ml.hydra_utils import extras, get_metric_value, task_wrapper


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


class TestExtras:
    def test_no_extras_key_returns_early(self):
        cfg = OmegaConf.create({})
        extras(cfg)

    def test_ignore_warnings(self):
        cfg = OmegaConf.create({"extras": {"ignore_warnings": True}})
        extras(cfg)

    def test_print_config_dispatches(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "radiocovid.utils.ml.hydra_utils.print_config_tree",
            lambda *a, **kw: called.update({"called": True}),
        )
        cfg = OmegaConf.create({"extras": {"print_config": True}})
        extras(cfg)
        assert called.get("called")

    def test_enforce_tags_dispatches(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "radiocovid.utils.ml.hydra_utils.enforce_tags",
            lambda *a, **kw: called.update({"called": True}),
        )
        cfg = OmegaConf.create({"extras": {"enforce_tags": True}})
        extras(cfg)
        assert called.get("called")
