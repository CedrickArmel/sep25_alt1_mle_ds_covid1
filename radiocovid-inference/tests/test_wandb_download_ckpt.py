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

from wandb_download_ckpt import choose_metric, download_artifact, find_model_artifact


def _make_artifact(name):
    art = MagicMock()
    art.name = name
    return art


def _make_run(summary=None):
    run = MagicMock()
    run.summary = summary or {}
    run.logged_artifacts.return_value = []
    return run


# --------------------------------------------------------------------------- #
# find_model_artifact                                                          #
# --------------------------------------------------------------------------- #


class TestFindModelArtifact:
    def test_returns_first_model_artifact(self):
        art1 = _make_artifact("model-abc123:v0")
        art2 = _make_artifact("dataset-xyz:v1")
        run = _make_run()
        run.logged_artifacts.return_value = [art1, art2]
        result = find_model_artifact(run)
        assert result is art1

    def test_matches_artifact_containing_model(self):
        art = _make_artifact("best_model_weights")
        run = _make_run()
        run.logged_artifacts.return_value = [art]
        result = find_model_artifact(run)
        assert result is art

    def test_returns_none_when_no_match(self):
        art = _make_artifact("dataset-train:v0")
        run = _make_run()
        run.logged_artifacts.return_value = [art]
        result = find_model_artifact(run)
        assert result is None

    def test_returns_none_for_empty_artifacts(self):
        run = _make_run()
        run.logged_artifacts.return_value = []
        result = find_model_artifact(run)
        assert result is None


# --------------------------------------------------------------------------- #
# choose_metric                                                                #
# --------------------------------------------------------------------------- #


class TestChooseMetric:
    def test_returns_best_val_score_when_present(self):
        runs = [_make_run({"best_val_score": 0.9})]
        assert choose_metric(runs) == "best_val_score"

    def test_falls_back_to_val_score(self):
        runs = [_make_run({"val_score": 0.8})]
        assert choose_metric(runs) == "val_score"

    def test_falls_back_to_val_accuracy(self):
        runs = [_make_run({"val_accuracy": 0.75})]
        assert choose_metric(runs) == "val_accuracy"

    def test_prefers_best_val_score_over_others(self):
        runs = [
            _make_run({"best_val_score": 0.9, "val_score": 0.8, "val_accuracy": 0.7})
        ]
        assert choose_metric(runs) == "best_val_score"

    def test_returns_none_when_no_candidate_present(self):
        runs = [_make_run({"train_loss": 0.1})]
        assert choose_metric(runs) is None

    def test_returns_none_for_empty_runs(self):
        assert choose_metric([]) is None


# --------------------------------------------------------------------------- #
# download_artifact                                                            #
# --------------------------------------------------------------------------- #


class TestDownloadArtifact:
    def test_returns_artifact_on_first_tag(self):
        artifact = MagicMock()
        api = MagicMock()
        api.artifact.return_value = artifact
        result = download_artifact(api, "org", "proj", "run123")
        assert result is artifact
        # Should try "best" tag first
        first_call_path = api.artifact.call_args_list[0][0][0]
        assert "best" in first_call_path

    def test_falls_back_through_tags(self):
        artifact = MagicMock()
        api = MagicMock()

        def side_effect(path):
            if "best" in path:
                raise Exception("not found")
            if "latest" in path:
                raise Exception("not found")
            return artifact

        api.artifact.side_effect = side_effect
        result = download_artifact(api, "org", "proj", "run123")
        assert result is artifact

    def test_returns_none_when_all_tags_fail(self):
        api = MagicMock()
        api.artifact.side_effect = Exception("not found")
        result = download_artifact(api, "org", "proj", "run123")
        assert result is None

    def test_artifact_path_includes_run_id(self):
        api = MagicMock()
        api.artifact.return_value = MagicMock()
        download_artifact(api, "myorg", "myproj", "run_xyz")
        called_path = api.artifact.call_args_list[0][0][0]
        assert "run_xyz" in called_path
        assert "myorg" in called_path
        assert "myproj" in called_path
