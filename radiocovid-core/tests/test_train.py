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

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from radiocovid.core.train import train

_MINIMAL_CFG = {
    "trainer": {"_target_": "lightning.pytorch.Trainer"},
    "datamodule": {"_target_": "radiocovid.core.data.datamodule.DataModule"},
    "module": {"_target_": "radiocovid.core.models.LModule"},
}


def _mock_instantiate(trainer, datamodule, module):
    """Returns a fake hydra.utils.instantiate that hands back pre-built objects in call order."""
    queue = [trainer, datamodule, module]

    def fake(cfg, **kwargs):
        return queue.pop(0)

    return fake


def _make_trainer():
    t = MagicMock()
    t.callback_metrics = {}
    t.checkpoint_callback = None
    return t


# --------------------------------------------------------------------------- #
# train()                                                                      #
# --------------------------------------------------------------------------- #


class TestTrainFunction:
    def test_returns_two_dicts(self, monkeypatch):
        trainer = _make_trainer()
        monkeypatch.setattr(
            "radiocovid.core.train.hydra.utils.instantiate",
            _mock_instantiate(trainer, MagicMock(), MagicMock()),
        )
        cfg = OmegaConf.create({**_MINIMAL_CFG, "train": True, "test": False})
        result = train(cfg)
        assert isinstance(result, tuple) and len(result) == 2
        metric_dict, object_dict = result
        assert isinstance(metric_dict, dict)
        assert isinstance(object_dict, dict)

    def test_object_dict_has_expected_keys(self, monkeypatch):
        trainer = _make_trainer()
        monkeypatch.setattr(
            "radiocovid.core.train.hydra.utils.instantiate",
            _mock_instantiate(trainer, MagicMock(), MagicMock()),
        )
        cfg = OmegaConf.create({**_MINIMAL_CFG, "train": True, "test": False})
        _, object_dict = train(cfg)
        for key in ("cfg", "datamodule", "model", "trainer"):
            assert key in object_dict

    def test_train_false_does_not_call_fit(self, monkeypatch):
        trainer = _make_trainer()
        monkeypatch.setattr(
            "radiocovid.core.train.hydra.utils.instantiate",
            _mock_instantiate(trainer, MagicMock(), MagicMock()),
        )
        cfg = OmegaConf.create({**_MINIMAL_CFG, "train": False, "test": False})
        train(cfg)
        trainer.fit.assert_not_called()

    def test_train_true_calls_fit(self, monkeypatch):
        trainer = _make_trainer()
        monkeypatch.setattr(
            "radiocovid.core.train.hydra.utils.instantiate",
            _mock_instantiate(trainer, MagicMock(), MagicMock()),
        )
        cfg = OmegaConf.create({**_MINIMAL_CFG, "train": True, "test": False})
        train(cfg)
        trainer.fit.assert_called_once()

    def test_test_true_calls_test(self, monkeypatch):
        trainer = _make_trainer()
        monkeypatch.setattr(
            "radiocovid.core.train.hydra.utils.instantiate",
            _mock_instantiate(trainer, MagicMock(), MagicMock()),
        )
        cfg = OmegaConf.create({**_MINIMAL_CFG, "train": True, "test": True})
        train(cfg)
        trainer.test.assert_called_once()

    def test_ckpt_path_from_checkpoint_callback(self, monkeypatch):
        trainer = _make_trainer()
        trainer.checkpoint_callback = MagicMock(best_model_path="/tmp/best.ckpt")
        monkeypatch.setattr(
            "radiocovid.core.train.hydra.utils.instantiate",
            _mock_instantiate(trainer, MagicMock(), MagicMock()),
        )
        cfg = OmegaConf.create({**_MINIMAL_CFG, "train": True, "test": True})
        train(cfg)
        _, kwargs = trainer.test.call_args
        assert kwargs.get("ckpt_path") == "/tmp/best.ckpt"
