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

import torch.nn as nn
from omegaconf import OmegaConf
from radiocovid.utils.ml.logging_utils import log_hyperparameters


def _cfg():
    return OmegaConf.create(
        {
            "module": {"lr": 0.001},
            "datamodule": {"batch_size": 32},
            "trainer": {"max_epochs": 10},
        }
    )


class FakeLogger:
    def __init__(self):
        self.logged = {}

    def log_hyperparams(self, params):
        self.logged.update(params)


class FakeTrainer:
    def __init__(self, loggers=None):
        self.loggers = loggers or []

    @property
    def logger(self):
        return self.loggers


class TestLogHyperparameters:
    def test_no_logger_returns_early(self):
        trainer = FakeTrainer(loggers=[])
        model = nn.Linear(4, 2)
        log_hyperparameters({"cfg": _cfg(), "model": model, "trainer": trainer})

    def test_dispatches_to_each_logger(self):
        lg1 = FakeLogger()
        lg2 = FakeLogger()
        trainer = FakeTrainer(loggers=[lg1, lg2])
        model = nn.Linear(4, 2)
        log_hyperparameters({"cfg": _cfg(), "model": model, "trainer": trainer})
        assert lg1.logged
        assert lg2.logged

    def test_total_param_count(self):
        lg = FakeLogger()
        trainer = FakeTrainer(loggers=[lg])
        model = nn.Linear(4, 8)
        log_hyperparameters({"cfg": _cfg(), "model": model, "trainer": trainer})
        assert lg.logged["model/params/total"] == 40

    def test_trainable_vs_non_trainable_count(self):
        lg = FakeLogger()
        trainer = FakeTrainer(loggers=[lg])
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        for p in model[0].parameters():
            p.requires_grad = False
        log_hyperparameters({"cfg": _cfg(), "model": model, "trainer": trainer})
        assert lg.logged["model/params/trainable"] == 18
        assert lg.logged["model/params/non_trainable"] == 40
        assert lg.logged["model/params/total"] == 58

    def test_cfg_fields_logged(self):
        lg = FakeLogger()
        trainer = FakeTrainer(loggers=[lg])
        model = nn.Linear(2, 2)
        log_hyperparameters({"cfg": _cfg(), "model": model, "trainer": trainer})
        assert "module" in lg.logged
        assert "datamodule" in lg.logged
        assert "trainer" in lg.logged
