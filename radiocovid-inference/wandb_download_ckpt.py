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
