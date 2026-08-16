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

"""Build the drift-monitoring reference distribution from the val/test set.

Usage (from repo root):
    uv run python scripts/build_reference.py --data-dir data/train_folder/val
    uv run python scripts/build_reference.py --data-dir data/train_folder/val \\
        --output data/reference_distribution.json --overwrite

The script:
  1. Walks *data-dir* recursively and collects all image files.
  2. Loads the @production model from the W&B registry.
  3. For each image, extracts {img_mean, img_std, img_entropy} and runs
     inference to get the model confidence.
  4. Computes per-feature statistics (mean, std, p5, p95) over all images.
  5. Saves the result as JSON (and optionally as a W&B artifact).

The output JSON is consumed by scripts/run_drift_check.py (DRIFT-03).
Run this script after every model promotion to keep the reference up-to-date.
    make build-reference
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repo root (no python-dotenv dependency)
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

# radiocovid.inference must be installed (uv sync or pip install -e radiocovid-inference/)
try:
    from radiocovid.inference.inference_logger import extract_image_features
    from radiocovid.inference.predict import get_transform
    from radiocovid.inference.predict import load_model as _load_wandb_model
    from radiocovid.inference.predict import predict as _run_predict
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

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(data_dir: Path, max_samples: int) -> list[Path]:
    """Return up to *max_samples* image paths found recursively under *data_dir*."""
    paths = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not paths:
        log.error("No image files found under %s", data_dir)
        sys.exit(1)
    if len(paths) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(paths), size=max_samples, replace=False)
        paths = [paths[i] for i in sorted(idx)]
        log.info("Sampled %d / %d images (--max-samples)", max_samples, len(paths))
    return paths


def build_reference(
    data_dir: Path,
    output_path: Path,
    max_samples: int,
    skip_model: bool,
    push_wandb: bool,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        log.error("%s already exists. Use --overwrite to replace it.", output_path)
        sys.exit(1)

    log.info("Collecting images from %s …", data_dir)
    image_paths = collect_images(data_dir, max_samples)
    log.info("Found %d images", len(image_paths))

    # Load production model (optional: can be skipped to build stats-only reference)
    model = device = transform = model_run_id = None
    if not skip_model:
        log.info("Loading @production model from W&B registry …")
        try:
            model, device, meta = _load_wandb_model()
            transform = get_transform()
            model_run_id = meta.get("run_id", "unknown")
            log.info("Model loaded — run_id=%s", model_run_id)
        except Exception as exc:
            log.warning(
                "Could not load W&B model (%s). "
                "Building image-stats-only reference (confidence will be absent).",
                exc,
            )

    # Extract features for each image
    rows: list[dict] = []
    failed = 0
    for i, path in enumerate(image_paths, 1):
        if i % 100 == 0:
            log.info("  %d / %d …", i, len(image_paths))
        try:
            image = Image.open(path).convert("RGB")
            feats = extract_image_features(image)
            if model is not None:
                _, conf = _run_predict(model, image, transform, device)
                feats["confidence"] = round(float(conf), 6)
            rows.append(feats)
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            failed += 1

    if not rows:
        log.error("No features extracted — aborting.")
        sys.exit(1)

    log.info("Extracted features for %d images (%d skipped)", len(rows), failed)

    # Compute per-feature statistics
    feature_names = list(rows[0].keys())
    stats: dict[str, dict] = {}
    for feat in feature_names:
        values = np.array([r[feat] for r in rows], dtype=np.float64)
        stats[feat] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    from datetime import datetime, timezone

    reference = {
        "n_samples": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_run_id": model_run_id or "none",
        "data_dir": str(data_dir),
        "features": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    log.info("Reference saved → %s", output_path)

    # Optional: push as W&B artifact
    if push_wandb:
        _push_to_wandb(reference, output_path)


def _push_to_wandb(reference: dict, json_path: Path) -> None:
    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "radiologist"),
            job_type="reference-build",
            name=f"reference-{reference['created_at'][:10]}",
        )
        artifact = wandb.Artifact(
            name="reference_distribution",
            type="monitoring",
            description="Drift monitoring reference distribution (val/test set features)",
            metadata={
                "n_samples": reference["n_samples"],
                "model_run_id": reference["model_run_id"],
            },
        )
        artifact.add_file(str(json_path))
        run.log_artifact(artifact, aliases=["latest"])
        run.finish()
        log.info("Reference artifact pushed to W&B as 'reference_distribution:latest'")
    except Exception as exc:
        log.warning("W&B push failed (non-blocking): %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the drift-monitoring reference distribution from val/test images."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory with val/test images (ImageFolder layout or flat).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "reference_distribution.json",
        help="Output JSON path (default: data/reference_distribution.json).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max images to sample (default: 2000, random seed 42).",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model loading — build image-stats reference only (no confidence).",
    )
    parser.add_argument(
        "--push-wandb",
        action="store_true",
        help="Upload the JSON as a W&B artifact 'reference_distribution:latest'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        log.error("--data-dir does not exist: %s", args.data_dir)
        sys.exit(1)

    build_reference(
        data_dir=args.data_dir,
        output_path=args.output,
        max_samples=args.max_samples,
        skip_model=args.skip_model,
        push_wandb=args.push_wandb,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
