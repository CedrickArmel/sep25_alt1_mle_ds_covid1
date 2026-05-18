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

import pytest
from omegaconf import OmegaConf
from radiocovid.core.utils.rich_utils import enforce_tags, print_config_tree

# --------------------------------------------------------------------------- #
# print_config_tree                                                            #
# --------------------------------------------------------------------------- #


class TestPrintConfigTree:
    def _cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "datamodule": {"batch_size": 32},
                "module": {"lr": 1e-3},
                "paths": {"output_dir": str(tmp_path)},
            }
        )

    def test_runs_without_error(self, tmp_path):
        print_config_tree(self._cfg(tmp_path))

    def test_missing_print_order_field_skipped(self, tmp_path):
        # cfg has no "callbacks" key — should not raise, just log a warning
        print_config_tree(self._cfg(tmp_path), print_order=("datamodule", "callbacks"))

    def test_save_to_file_writes_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        print_config_tree(self._cfg(tmp_path), save_to_file=True)
        assert (tmp_path / "config_tree.log").exists()

    def test_extra_fields_included_after_print_order(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "datamodule": {"x": 1},
                "extra_field": "hello",
                "paths": {"output_dir": str(tmp_path)},
            }
        )
        # Should not raise even when extra_field is not in print_order
        print_config_tree(cfg, print_order=("datamodule",))


# --------------------------------------------------------------------------- #
# enforce_tags                                                                 #
# --------------------------------------------------------------------------- #


class TestEnforceTags:
    def test_tags_present_no_prompt(self, monkeypatch):
        prompt_called = {}
        monkeypatch.setattr(
            "radiocovid.core.utils.rich_utils.Prompt.ask",
            lambda *a, **kw: prompt_called.update({"called": True}) or "dev",
        )
        cfg = OmegaConf.create({"tags": ["v1", "exp"]})
        enforce_tags(cfg)
        assert not prompt_called

    def test_no_tags_prompts_and_sets(self, monkeypatch):
        monkeypatch.setattr(
            "radiocovid.core.utils.rich_utils.HydraConfig",
            lambda: MagicMock(cfg=MagicMock(hydra=MagicMock(job={}))),
        )
        monkeypatch.setattr(
            "radiocovid.core.utils.rich_utils.Prompt.ask",
            lambda *a, **kw: "train, v2",
        )
        cfg = OmegaConf.create({})
        enforce_tags(cfg)
        assert cfg.tags == ["train", "v2"]

    def test_multirun_without_tags_raises(self, monkeypatch):
        monkeypatch.setattr(
            "radiocovid.core.utils.rich_utils.HydraConfig",
            lambda: MagicMock(cfg=MagicMock(hydra=MagicMock(job={"id": 0}))),
        )
        cfg = OmegaConf.create({})
        with pytest.raises(ValueError, match="multirun"):
            enforce_tags(cfg)

    def test_save_to_file_writes_tags(self, tmp_path, monkeypatch):
        cfg = OmegaConf.create(
            {
                "tags": ["smoke"],
                "paths": {"output_dir": str(tmp_path)},
            }
        )
        enforce_tags(cfg, save_to_file=True)
        assert (tmp_path / "tags.log").exists()
