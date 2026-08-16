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

import json

import numpy as np
import pytest
from PIL import Image

from radiocovid.inference.inference_logger import extract_image_features, log_prediction


@pytest.fixture
def rgb_image():
    """32×32 random RGB image."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def uniform_image():
    """All-grey 32×32 image — predictable statistics."""
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# extract_image_features
# ---------------------------------------------------------------------------


class TestExtractImageFeatures:
    def test_returns_three_keys(self, rgb_image):
        feats = extract_image_features(rgb_image)
        assert set(feats.keys()) == {"img_mean", "img_std", "img_entropy"}

    def test_all_values_are_floats(self, rgb_image):
        feats = extract_image_features(rgb_image)
        for key, val in feats.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_img_mean_in_unit_interval(self, rgb_image):
        feats = extract_image_features(rgb_image)
        assert 0.0 <= feats["img_mean"] <= 1.0

    def test_img_std_non_negative(self, rgb_image):
        feats = extract_image_features(rgb_image)
        assert feats["img_std"] >= 0.0

    def test_img_entropy_positive(self, rgb_image):
        feats = extract_image_features(rgb_image)
        assert feats["img_entropy"] > 0.0

    def test_uniform_image_has_zero_std(self, uniform_image):
        feats = extract_image_features(uniform_image)
        assert feats["img_std"] < 1e-6

    def test_uniform_image_mean_approx_half(self, uniform_image):
        feats = extract_image_features(uniform_image)
        assert abs(feats["img_mean"] - 128 / 255) < 0.01


# ---------------------------------------------------------------------------
# log_prediction — disabled (default)
# ---------------------------------------------------------------------------


class TestLogPredictionDisabled:
    def test_no_file_created_when_disabled(self, tmp_path, monkeypatch, rgb_image):
        monkeypatch.setenv("ENABLE_INFERENCE_LOGGING", "0")
        monkeypatch.setenv("INFERENCE_LOG_DIR", str(tmp_path / "logs"))
        log_prediction(rgb_image, "NORMAL", 0.95, "run_abc")
        assert not (tmp_path / "logs" / "predictions.jsonl").exists()

    def test_default_is_disabled(self, tmp_path, monkeypatch, rgb_image):
        monkeypatch.delenv("ENABLE_INFERENCE_LOGGING", raising=False)
        monkeypatch.setenv("INFERENCE_LOG_DIR", str(tmp_path / "logs"))
        log_prediction(rgb_image, "NORMAL", 0.95, "run_abc")
        assert not (tmp_path / "logs" / "predictions.jsonl").exists()


# ---------------------------------------------------------------------------
# log_prediction — enabled
# ---------------------------------------------------------------------------


class TestLogPredictionEnabled:
    @pytest.fixture(autouse=True)
    def enable_logging(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENABLE_INFERENCE_LOGGING", "1")
        monkeypatch.setenv("INFERENCE_LOG_DIR", str(tmp_path / "logs"))
        self.log_file = tmp_path / "logs" / "predictions.jsonl"

    def test_creates_file_on_first_call(self, rgb_image):
        log_prediction(rgb_image, "NORMAL", 0.9, "run_1")
        assert self.log_file.exists()

    def test_record_has_expected_keys(self, rgb_image):
        log_prediction(rgb_image, "NORMAL", 0.9, "run_1")
        record = json.loads(self.log_file.read_text())
        expected = {
            "timestamp",
            "label",
            "confidence",
            "img_mean",
            "img_std",
            "img_entropy",
            "model_run_id",
        }
        assert set(record.keys()) == expected

    def test_label_stored_correctly(self, rgb_image):
        log_prediction(rgb_image, "ABNORMAL", 0.75, "run_1")
        record = json.loads(self.log_file.read_text())
        assert record["label"] == "ABNORMAL"

    def test_confidence_stored_correctly(self, rgb_image):
        log_prediction(rgb_image, "NORMAL", 0.876543, "run_1")
        record = json.loads(self.log_file.read_text())
        assert abs(record["confidence"] - 0.876543) < 1e-5

    def test_model_run_id_stored(self, rgb_image):
        log_prediction(rgb_image, "NORMAL", 0.9, "my_run_42")
        record = json.loads(self.log_file.read_text())
        assert record["model_run_id"] == "my_run_42"

    def test_multiple_calls_append_lines(self, rgb_image):
        log_prediction(rgb_image, "NORMAL", 0.9, "run_1")
        log_prediction(rgb_image, "ABNORMAL", 0.6, "run_1")
        log_prediction(rgb_image, "NORMAL", 0.85, "run_1")
        lines = self.log_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, rgb_image):
        for _ in range(3):
            log_prediction(rgb_image, "NORMAL", 0.9, "run_1")
        for line in self.log_file.read_text().strip().split("\n"):
            json.loads(line)

    def test_timestamp_is_iso_format(self, rgb_image):
        from datetime import datetime

        log_prediction(rgb_image, "NORMAL", 0.9, "run_1")
        record = json.loads(self.log_file.read_text())
        datetime.fromisoformat(record["timestamp"])


# ---------------------------------------------------------------------------
# log_prediction — error resilience
# ---------------------------------------------------------------------------


class TestLogPredictionErrorResilience:
    def test_does_not_raise_on_bad_log_dir(self, monkeypatch, rgb_image):
        monkeypatch.setenv("ENABLE_INFERENCE_LOGGING", "1")
        # Use a path that cannot be created (root-level on most systems)
        monkeypatch.setenv("INFERENCE_LOG_DIR", "/proc/bad_path_that_cannot_exist")
        # Should not raise — just log a warning
        log_prediction(rgb_image, "NORMAL", 0.9, "run_1")

    def test_does_not_raise_on_invalid_image(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENABLE_INFERENCE_LOGGING", "1")
        monkeypatch.setenv("INFERENCE_LOG_DIR", str(tmp_path / "logs"))
        # Pass None instead of Image — triggers exception in extract_image_features
        log_prediction(None, "NORMAL", 0.9, "run_1")  # type: ignore[arg-type]
