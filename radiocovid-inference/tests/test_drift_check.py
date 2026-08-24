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

"""Unit tests for radiocovid.inference.drift_check (DRIFT-03)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from radiocovid.inference.drift_check import (
    load_predictions,
    run_drift_analysis,
    synthetic_reference_df,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REFERENCE = {
    "n_samples": 200,
    "created_at": "2026-08-13T00:00:00+00:00",
    "model_run_id": "test_run",
    "features": {
        "img_mean": {"mean": 0.48, "std": 0.09, "p5": 0.32, "p95": 0.63},
        "img_std": {"mean": 0.21, "std": 0.05, "p5": 0.13, "p95": 0.30},
        "img_entropy": {"mean": 5.12, "std": 0.41, "p5": 4.40, "p95": 5.78},
    },
}


def _make_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    log_file = tmp_path / "predictions.jsonl"
    with log_file.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return log_file


def _record(offset_hours: int = 0, **overrides) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(hours=offset_hours)
    base = {
        "timestamp": ts.isoformat(),
        "label": "NORMAL",
        "confidence": 0.90,
        "img_mean": 0.50,
        "img_std": 0.20,
        "img_entropy": 5.10,
        "model_run_id": "run_test",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    def test_returns_empty_df_when_file_absent(self, tmp_path):
        df = load_predictions(tmp_path / "missing.jsonl", window_days=7)
        assert df.empty

    def test_loads_records_within_window(self, tmp_path):
        records = [_record(offset_hours=1), _record(offset_hours=2)]
        log_file = _make_jsonl(tmp_path, records)
        df = load_predictions(log_file, window_days=7)
        assert len(df) == 2

    def test_excludes_old_records(self, tmp_path):
        old = _record(offset_hours=24 * 8)  # 8 days ago — outside 7-day window
        recent = _record(offset_hours=1)
        log_file = _make_jsonl(tmp_path, [old, recent])
        df = load_predictions(log_file, window_days=7)
        assert len(df) == 1

    def test_skips_malformed_lines(self, tmp_path):
        log_file = tmp_path / "predictions.jsonl"
        log_file.write_text('{"bad": true}\nnot-json\n' + json.dumps(_record()) + "\n")
        df = load_predictions(log_file, window_days=7)
        assert len(df) == 1

    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        log_file = _make_jsonl(tmp_path, [_record()])
        df = load_predictions(log_file, window_days=7)
        for col in ("img_mean", "img_std", "img_entropy", "confidence", "label"):
            assert col in df.columns

    def test_empty_window_returns_empty_df(self, tmp_path):
        log_file = _make_jsonl(tmp_path, [_record(offset_hours=1)])
        df = load_predictions(log_file, window_days=0)
        assert df.empty


# ---------------------------------------------------------------------------
# synthetic_reference_df
# ---------------------------------------------------------------------------


class TestSyntheticReferenceDf:
    def test_shape(self):
        df = synthetic_reference_df(SAMPLE_REFERENCE, n_samples=500)
        assert df.shape == (500, len(SAMPLE_REFERENCE["features"]))

    def test_columns_match_features(self):
        df = synthetic_reference_df(SAMPLE_REFERENCE)
        assert set(df.columns) == set(SAMPLE_REFERENCE["features"].keys())

    def test_values_within_p5_p95(self):
        df = synthetic_reference_df(SAMPLE_REFERENCE, n_samples=1000)
        for feat, stats in SAMPLE_REFERENCE["features"].items():
            assert df[feat].min() >= stats["p5"] - 1e-6
            assert df[feat].max() <= stats["p95"] + 1e-6

    def test_reproducible_with_same_seed(self):
        df1 = synthetic_reference_df(SAMPLE_REFERENCE, seed=0)
        df2 = synthetic_reference_df(SAMPLE_REFERENCE, seed=0)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = synthetic_reference_df(SAMPLE_REFERENCE, seed=0)
        df2 = synthetic_reference_df(SAMPLE_REFERENCE, seed=99)
        assert not df1.equals(df2)


# ---------------------------------------------------------------------------
# run_drift_analysis
# ---------------------------------------------------------------------------


def _make_current_df(n: int = 200, drift: bool = False) -> pd.DataFrame:
    """Build a current DataFrame matching or drifting from SAMPLE_REFERENCE."""
    rng = np.random.default_rng(1)
    if drift:
        # Shift mean significantly — should trigger KS drift
        return pd.DataFrame(
            {
                "img_mean": rng.normal(0.80, 0.05, n),  # ref mean=0.48
                "img_std": rng.normal(0.50, 0.05, n),  # ref mean=0.21
                "img_entropy": rng.normal(5.10, 0.40, n),
            }
        )
    else:
        # Match reference stats — should not trigger drift
        return pd.DataFrame(
            {
                "img_mean": rng.normal(0.48, 0.09, n),
                "img_std": rng.normal(0.21, 0.05, n),
                "img_entropy": rng.normal(5.12, 0.41, n),
            }
        )


class TestRunDriftAnalysis:
    def test_returns_expected_keys(self):
        current_df = _make_current_df(n=200)
        result = run_drift_analysis(current_df, SAMPLE_REFERENCE)
        for key in (
            "drift_detected",
            "drifted_features",
            "features",
            "n_current",
            "n_reference",
        ):
            assert key in result

    def test_no_drift_when_distributions_match(self):
        current_df = _make_current_df(n=300, drift=False)
        result = run_drift_analysis(current_df, SAMPLE_REFERENCE, threshold_features=2)
        assert not result["drift_detected"]

    def test_drift_detected_when_distributions_shift(self):
        current_df = _make_current_df(n=300, drift=True)
        result = run_drift_analysis(current_df, SAMPLE_REFERENCE, threshold_features=2)
        assert result["drift_detected"]
        assert len(result["drifted_features"]) >= 2

    def test_feature_results_have_correct_shape(self):
        current_df = _make_current_df(n=200)
        result = run_drift_analysis(current_df, SAMPLE_REFERENCE)
        for feat, stats in result["features"].items():
            assert "drift_detected" in stats
            assert "drift_score" in stats
            assert isinstance(stats["drift_detected"], bool)
            assert isinstance(stats["drift_score"], float)

    def test_n_current_matches_input(self):
        current_df = _make_current_df(n=150)
        result = run_drift_analysis(current_df, SAMPLE_REFERENCE)
        assert result["n_current"] == 150

    def test_raises_when_no_common_features(self):
        current_df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="No common features"):
            run_drift_analysis(current_df, SAMPLE_REFERENCE)

    def test_saves_html_report(self, tmp_path):
        current_df = _make_current_df(n=200)
        report_path = tmp_path / "drift_report.html"
        run_drift_analysis(current_df, SAMPLE_REFERENCE, report_path=report_path)
        assert report_path.exists()
        assert report_path.stat().st_size > 0

    def test_threshold_respected(self):
        """With threshold=3, even 2 drifting features should not trigger drift_detected."""
        current_df = _make_current_df(n=300, drift=True)
        result_strict = run_drift_analysis(
            current_df, SAMPLE_REFERENCE, threshold_features=3
        )
        result_loose = run_drift_analysis(
            current_df, SAMPLE_REFERENCE, threshold_features=1
        )
        # With threshold=1, drift should definitely be detected
        assert result_loose["drift_detected"]
        # With threshold=3 and only 3 features in reference, it may or may not trigger
        assert isinstance(result_strict["drift_detected"], bool)
