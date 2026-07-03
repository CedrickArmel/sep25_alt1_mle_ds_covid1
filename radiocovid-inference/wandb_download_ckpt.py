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


from __future__ import annotations

from typing import Any

_METRIC_CANDIDATES = ["best_val_score", "val_score", "val_accuracy"]
_ARTIFACT_TAGS = ["best", "latest", "v0"]


def find_model_artifact(run: Any):
    """Return the first logged artifact whose name contains 'model', or None."""
    for artifact in run.logged_artifacts():
        if "model" in artifact.name:
            return artifact
    return None


def choose_metric(runs: list) -> str | None:
    """Return the first metric candidate present in any run summary, or None."""
    for candidate in _METRIC_CANDIDATES:
        for run in runs:
            if candidate in run.summary:
                return candidate
    return None


def fetch_production_model(api: Any, entity: str, registered_model_name: str):
    """Fetch the artifact tagged 'production' from the W&B Model Registry.

    Returns the artifact or None if not found.
    """
    try:
        return api.artifact(f"{entity}/model-registry/{registered_model_name}:production")
    except Exception:
        return None


def download_artifact(api: Any, org: str, proj: str, run_id: str):
    """Try to fetch the model artifact for *run_id* using known aliases.

    Tries 'best' -> 'latest' -> 'v0'. Returns the artifact on the first
    successful attempt, or None if all aliases fail.
    """
    for tag in _ARTIFACT_TAGS:
        try:
            return api.artifact(f"{org}/{proj}/model-{run_id}:{tag}")
        except Exception:
            continue
    return None
