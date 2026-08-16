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

"""Drift check CLI — entry point for the radiocovid_monitoring DAG.

Usage (from repo root)::

    uv run python scripts/run_drift_check.py --window 7 --min-samples 50

Environment variables (all optional, override CLI defaults):

    DRIFT_WINDOW_DAYS         int   Days of predictions to analyse (default 7)
    DRIFT_MIN_SAMPLES         int   Skip if fewer predictions available (default 50)
    DRIFT_THRESHOLD_FEATURES  int   Min drifting features to alert (default 2)
    DRIFT_REPORT_DIR          str   Directory for HTML reports (default reports/)
    DRIFT_FAIL_ON_DETECT      0|1   Exit 1 if drift detected (default 0)
    INFERENCE_LOG_DIR         str   Directory of predictions.jsonl (default data/inference_logs)
    ENABLE_WANDB_LOGGING      0|1   Push results to W&B (default 0)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repo root (no python-dotenv dependency)
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

try:
    from radiocovid.inference.drift_check import load_predictions, run_drift_analysis
except ImportError as exc:
    print(
        f"ERROR: radiocovid-inference package not found.\n"
        f"Run: pip install -e radiocovid-inference/\n"
        f"Details: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------


def _push_to_wandb(result: dict, report_path: Path | None) -> None:
    """Log drift result to W&B run summary and push HTML report as artifact."""
    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "radiologist"),
            job_type="drift-check",
            name=f"drift-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        )

        # Log scalar summary
        wandb.log(
            {
                "drift_detected": int(result["drift_detected"]),
                "n_drifted_features": len(result["drifted_features"]),
                "n_current_samples": result["n_current"],
            }
        )
        for feat, stats in result["features"].items():
            wandb.log(
                {
                    f"drift/{feat}/detected": int(stats["drift_detected"]),
                    f"drift/{feat}/score": stats["drift_score"],
                }
            )

        # Push HTML report as artifact
        if report_path and report_path.exists():
            artifact = wandb.Artifact(
                name="drift_report",
                type="monitoring",
                description="Evidently drift report (KS test)",
                metadata={
                    "drift_detected": result["drift_detected"],
                    "drifted_features": result["drifted_features"],
                },
            )
            artifact.add_file(str(report_path))
            run.log_artifact(artifact, aliases=["latest"])
            log.info("Drift report artifact pushed to W&B")

        run.finish()
    except Exception as exc:
        log.warning("W&B push failed (non-blocking): %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run covariate-shift drift check on recent inference predictions."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=int(os.environ.get("DRIFT_WINDOW_DAYS", "7")),
        help="Number of days of predictions to analyse (default: 7).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=int(os.environ.get("DRIFT_MIN_SAMPLES", "50")),
        help="Minimum number of predictions required to run the check (default: 50).",
    )
    parser.add_argument(
        "--threshold-features",
        type=int,
        default=int(os.environ.get("DRIFT_THRESHOLD_FEATURES", "2")),
        help="Min drifting features to declare drift (default: 2).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(os.environ.get("INFERENCE_LOG_DIR", "data/inference_logs"))
        / "predictions.jsonl",
        help="Path to predictions JSONL log (default: data/inference_logs/predictions.jsonl).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / "data" / "reference_distribution.json",
        help="Path to reference_distribution.json (default: data/reference_distribution.json).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(os.environ.get("DRIFT_REPORT_DIR", "reports")),
        help="Directory for HTML drift reports (default: reports/).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load reference
    # ------------------------------------------------------------------
    if not args.reference.exists():
        log.error(
            "Reference distribution not found: %s\n"
            "Run: make build-reference  (or scripts/build_reference.py)",
            args.reference,
        )
        sys.exit(1)

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    log.info(
        "Reference loaded — n_samples=%d, created_at=%s",
        reference.get("n_samples", "?"),
        reference.get("created_at", "?"),
    )

    # ------------------------------------------------------------------
    # Load current predictions
    # ------------------------------------------------------------------
    log.info("Loading predictions from %s (window=%dd) …", args.log_file, args.window)
    current_df = load_predictions(args.log_file, window_days=args.window)

    if len(current_df) < args.min_samples:
        log.info(
            "Not enough data, skipping — got %d sample(s), need at least %d.",
            len(current_df),
            args.min_samples,
        )
        sys.exit(0)

    log.info("Loaded %d predictions for drift analysis.", len(current_df))

    # ------------------------------------------------------------------
    # Run drift analysis + generate HTML report
    # ------------------------------------------------------------------
    report_filename = f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d')}.html"
    report_path = args.report_dir / report_filename

    result = run_drift_analysis(
        current_df=current_df,
        reference=reference,
        threshold_features=args.threshold_features,
        report_path=report_path,
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    drift_detected = result["drift_detected"]
    drifted = result["drifted_features"]

    if drift_detected:
        log.warning(
            "DRIFT DETECTED on %d/%d features: %s",
            len(drifted),
            args.threshold_features,
            drifted,
        )
    else:
        log.info(
            "No drift detected (%d/%d features drifted, threshold=%d).",
            len(drifted),
            len(result["features"]),
            args.threshold_features,
        )

    for feat, stats in result["features"].items():
        log.info(
            "  %-15s drift=%s  score=%.4f",
            feat,
            "YES" if stats["drift_detected"] else "no ",
            stats["drift_score"],
        )

    # ------------------------------------------------------------------
    # W&B push (optional)
    # ------------------------------------------------------------------
    if os.environ.get("ENABLE_WANDB_LOGGING", "0") == "1":
        _push_to_wandb(result, report_path)

    # ------------------------------------------------------------------
    # Exit code
    # ------------------------------------------------------------------
    if drift_detected and os.environ.get("DRIFT_FAIL_ON_DETECT", "0") == "1":
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
