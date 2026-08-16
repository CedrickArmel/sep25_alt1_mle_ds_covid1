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

"""Covariate-shift detection using Evidently (Kolmogorov-Smirnov test).

This module is deliberately separated from ``run_drift_check.py`` so that
the core logic can be unit-tested without the CLI scaffolding.

Typical usage::

    from radiocovid.inference.drift_check import (
        load_predictions,
        run_drift_analysis,
    )

    current_df = load_predictions(Path("data/inference_logs/predictions.jsonl"), window_days=7)
    result = run_drift_analysis(current_df, reference, report_path=Path("reports/drift.html"))
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical feature order — confidence may be absent when --skip-model was used
FEATURES = ["img_mean", "img_std", "img_entropy", "confidence"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions(jsonl_path: Path, window_days: int) -> pd.DataFrame:
    """Load JSONL prediction log, keeping only records within *window_days*.

    Malformed lines are skipped with a warning so a single bad record never
    aborts the whole drift check.

    Args:
        jsonl_path: Path to ``predictions.jsonl`` written by ``inference_logger``.
        window_days: How many days back to look (inclusive, UTC).

    Returns:
        DataFrame with columns matching the JSONL schema, or an empty DataFrame
        if the file is absent or no records fall in the window.
    """
    if not jsonl_path.exists():
        log.warning("Prediction log not found: %s", jsonl_path)
        return pd.DataFrame()

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows: list[dict] = []

    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                ts = datetime.fromisoformat(record["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    rows.append(record)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("Line %d — skipping malformed record: %s", lineno, exc)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference reconstruction
# ---------------------------------------------------------------------------


def synthetic_reference_df(
    reference: dict, n_samples: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Reconstruct a reference DataFrame from stored summary statistics.

    ``build_reference.py`` stores only aggregated stats (mean, std, p5, p95).
    Evidently requires actual rows to compare against, so we generate synthetic
    data from a Normal distribution clipped to [p5, p95].

    This is a reasonable approximation for monitoring purposes — the KS test
    compares the *shape* of distributions, not exact values.

    Args:
        reference: Parsed ``reference_distribution.json``.
        n_samples: Number of synthetic rows to generate (≥ 200 recommended).
        seed: NumPy random seed for reproducibility.

    Returns:
        DataFrame with one column per feature present in the reference.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    for feat, stats in reference["features"].items():
        std = max(float(stats["std"]), 1e-8)
        samples = rng.normal(float(stats["mean"]), std, n_samples)
        p5 = stats.get("p5", samples.min())
        p95 = stats.get("p95", samples.max())
        data[feat] = np.clip(samples, p5, p95).astype(np.float64)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Drift analysis
# ---------------------------------------------------------------------------


def run_drift_analysis(
    current_df: pd.DataFrame,
    reference: dict,
    threshold_features: int = 2,
    report_path: Optional[Path] = None,
) -> dict:
    """Run Evidently KS drift test on all features available in both datasets.

    Args:
        current_df: DataFrame of recent predictions from ``load_predictions``.
        reference: Parsed ``reference_distribution.json``.
        threshold_features: Minimum number of drifting features to set
            ``drift_detected=True`` (default: 2 out of 4, configurable via
            ``DRIFT_THRESHOLD_FEATURES`` env var).
        report_path: If provided, save the Evidently HTML report to this path.

    Returns:
        Dictionary with shape::

            {
                "drift_detected": bool,
                "drifted_features": ["img_mean", ...],
                "threshold_features": int,
                "features": {
                    "img_mean": {"drift_detected": bool, "drift_score": float},
                    ...
                },
                "n_current": int,
                "n_reference": int,
            }

    Raises:
        ValueError: If no features are common to reference and current data.
        ImportError: If evidently is not installed.
    """
    # Lazy imports — evidently is an optional dependency
    # Evidently 0.7+ API (ValueDrift replaces ColumnDriftMetric from 0.4.x)
    from evidently import Report
    from evidently.metrics import ValueDrift

    ref_df = synthetic_reference_df(reference)

    # Only test features present in both reference stats and current data
    available = [
        f
        for f in FEATURES
        if f in reference.get("features", {}) and f in current_df.columns
    ]
    if not available:
        raise ValueError(
            "No common features between reference and current data. "
            f"Reference features: {list(reference.get('features', {}).keys())}. "
            f"Current columns: {list(current_df.columns)}."
        )

    log.info("Running KS drift test on features: %s", available)

    # ValueDrift returns p-value as result.value; drift when p < threshold (default 0.05)
    _KS_PVALUE_THRESHOLD = 0.05
    metrics = [ValueDrift(column=f, method="ks") for f in available]
    report = Report(metrics=metrics)
    snapshot = report.run(
        reference_data=ref_df[available].reset_index(drop=True),
        current_data=current_df[available].reset_index(drop=True),
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_path))
        log.info("HTML report saved → %s", report_path)

    # Parse results — display_name = "Value drift for {feature}", value = KS p-value
    feature_results: dict[str, dict] = {}
    _PREFIX = "Value drift for "
    for result in snapshot.metric_results.values():
        display = getattr(result, "display_name", "")
        if not display.startswith(_PREFIX):
            continue
        feat = display[len(_PREFIX) :]
        p_value = float(getattr(result, "value", float("nan")))
        feature_results[feat] = {
            "drift_detected": p_value < _KS_PVALUE_THRESHOLD,
            "drift_score": p_value,
        }

    drifted_features = [f for f, r in feature_results.items() if r["drift_detected"]]
    drift_detected = len(drifted_features) >= threshold_features

    result = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "threshold_features": threshold_features,
        "features": feature_results,
        "n_current": len(current_df),
        "n_reference": len(ref_df),
    }

    log.info(
        "Drift analysis complete — drift_detected=%s, drifted=%s (%d/%d threshold)",
        drift_detected,
        drifted_features,
        len(drifted_features),
        threshold_features,
    )
    return result
