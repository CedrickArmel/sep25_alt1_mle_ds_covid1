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

"""Inference logger — log image features + prediction to a JSONL file.

Each call to ``log_prediction`` appends one JSON line to
``<INFERENCE_LOG_DIR>/predictions.jsonl``.

Activation:
    Set ``ENABLE_INFERENCE_LOGGING=1`` in the environment (default: ``0``).
    When disabled every function is a no-op so tests remain unaffected.

Storage:
    Local JSONL file in ``INFERENCE_LOG_DIR`` (default: ``data/inference_logs``).
    The file is a shared Docker volume accessible to the drift-check container.
    W&B is used only by the drift-check script (DRIFT-03) for reporting.

Schema (one JSON object per line):
    timestamp     : ISO-8601 UTC string
    label         : "NORMAL" | "ABNORMAL"
    confidence    : float in [0, 1]
    img_mean      : mean pixel intensity (grayscale, normalised 0-1)
    img_std       : pixel standard deviation (grayscale, normalised 0-1)
    img_entropy   : histogram entropy (higher = more complex texture)
    model_run_id  : W&B run_id of the model that produced the prediction
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_image_features(image: Image.Image) -> dict[str, float]:
    """Return ``img_mean``, ``img_std`` and ``img_entropy`` for *image*.

    The image is converted to grayscale and normalised to [0, 1] before
    computing statistics.  All arithmetic uses numpy — no scipy dependency.
    """
    arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist += 1e-10
    hist /= hist.sum()
    entropy = float(-(hist * np.log(hist)).sum())
    return {
        "img_mean": float(arr.mean()),
        "img_std": float(arr.std()),
        "img_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------


def log_prediction(
    image: Image.Image,
    label: str,
    confidence: float,
    model_run_id: str,
) -> None:
    """Append one prediction record to the JSONL log file.

    The call is a **no-op** when ``ENABLE_INFERENCE_LOGGING != "1"``.
    Any I/O or serialisation error is caught and logged as a warning so
    that a logging failure never causes a 500 on ``/predict``.
    """
    if os.environ.get("ENABLE_INFERENCE_LOGGING", "0") != "1":
        return

    try:
        features = extract_image_features(image)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "confidence": round(float(confidence), 6),
            "model_run_id": model_run_id,
            **features,
        }

        log_dir = Path(os.environ.get("INFERENCE_LOG_DIR", "data/inference_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "predictions.jsonl"

        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    except Exception as exc:
        logger.warning("Inference logging failed (non-blocking): %s", exc)
