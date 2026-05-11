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

"""Checkpoint resolution: local path or W&B artifact download."""

from pathlib import Path

import wandb


def find_model_artifact(run):
    artifacts = list(run.logged_artifacts())
    for art in artifacts:
        if art.name.startswith("model-") or "model" in art.name:
            return art
    return None


def choose_metric(runs):
    candidates = ["best_val_score", "val_score", "val_accuracy"]
    for metric in candidates:
        if any(run.summary.get(metric) is not None for run in runs):
            return metric
    return None


def download_artifact(api, entity, project, run_id):
    fallback_tags = ["best", "latest", "v0"]
    for tag in fallback_tags:
        artifact_name = f"model-{run_id}:{tag}"
        artifact_path = f"{entity}/{project}/{artifact_name}"
        try:
            artifact = api.artifact(artifact_path)
            print(f"Downloading artifact: {artifact_path}")
            return artifact
        except Exception:
            pass
    return None


def download_best_artifact(
    entity: str = "yebouetc", project: str = "radiocovid"
) -> Path:
    """Download the best model artifact from W&B based on val metric.

    Args:
        entity: W&B entity name.
        project: W&B project name.

    Returns:
        Path to the downloaded artifact directory.
    """
    api = wandb.Api()
    print(f"Fetching all runs from {entity}/{project}...")

    runs = list(api.runs(f"{entity}/{project}"))
    if not runs:
        raise RuntimeError("No runs found!")

    metric_name = choose_metric(runs)
    if metric_name is None:
        raise RuntimeError(
            f"No valid metric found. Available metrics: {sorted(runs[0].summary.keys())}"
        )

    print(f"Using metric: {metric_name}")

    runs_with_metric = [run for run in runs if run.summary.get(metric_name) is not None]
    if not runs_with_metric:
        raise RuntimeError(f"No runs contain metric '{metric_name}'")

    runs_with_artifact = []
    for run in runs_with_metric:
        artifact = find_model_artifact(run)
        if artifact is not None:
            runs_with_artifact.append((run, run.summary[metric_name], artifact))

    if runs_with_artifact:
        best_run, best_value, best_artifact = max(
            runs_with_artifact, key=lambda item: item[1]
        )
        print(f"Best run with model artifact: {best_run.name} (ID: {best_run.id})")
        print(f"  {metric_name}: {best_value}")
        artifact = best_artifact
    else:
        best_run = max(runs_with_metric, key=lambda run: run.summary[metric_name])
        print(f"Best run by metric: {best_run.name} (ID: {best_run.id})")
        artifact = download_artifact(api, entity, project, best_run.id)
        if artifact is None:
            available = [art.name for art in best_run.logged_artifacts()]
            raise RuntimeError(f"No artifact found. Available on best run: {available}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    artifact_dir = artifact.download(str(models_dir))
    print(f"Model downloaded to: {artifact_dir}")
    return Path(artifact_dir)


def resolve_checkpoint(uri_or_path: str) -> Path:
    """Resolve a checkpoint path or wandb:// URI to a local Path.

    Args:
        uri_or_path: Local file path or ``wandb://entity/project`` URI.

    Returns:
        Resolved local path.
    """
    if uri_or_path.startswith("wandb://"):
        parts = uri_or_path[len("wandb://") :].split("/")
        entity = parts[0] if len(parts) > 0 else "yebouetc"
        project = parts[1] if len(parts) > 1 else "radiocovid"
        artifact_dir = download_best_artifact(entity=entity, project=project)
        ckpts = list(artifact_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found in {artifact_dir}")
        return ckpts[0]
    return Path(uri_or_path)


def _cli():
    """CLI entry point: download the best W&B artifact to models/."""
    import sys

    entity = sys.argv[1] if len(sys.argv) > 1 else "yebouetc"
    project = sys.argv[2] if len(sys.argv) > 2 else "radiocovid"
    try:
        path = download_best_artifact(entity=entity, project=project)
        print(f"Done. Checkpoint at: {path}")
        raise SystemExit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
