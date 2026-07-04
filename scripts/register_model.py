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

"""
Promote the best model from a W&B project to the Model Registry.

Location: scripts/register_model.py (run from repository root)

Usage:
    uv run --with wandb python scripts/register_model.py              # dry-run
    uv run --with wandb python scripts/register_model.py --promote    # apply

Reads WANDB_* from .env at the repository root. Scans WANDB_PROJECT (default:
radiologist) for the best run that has a model artifact, then links it to the
existing registry collection.

The registry must already exist (e.g. Radiocovid-classifier). Promoting adds a
new version and moves the alias — older versions remain in the registry history.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repository root (no python-dotenv dependency)
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import wandb  # noqa: E402

ENTITY = os.environ["WANDB_ENTITY"]
PROJECT = os.environ.get("WANDB_PROJECT", "radiologist")
REGISTRY = os.environ.get("WANDB_REGISTRY", "Radiocovid-classifier")
COLLECTION = os.environ.get("WANDB_REGISTRY_MODEL", "radiocovid-classifier")
ALIAS = os.environ.get("WANDB_REGISTRY_ALIAS", "production")

METRIC_CANDIDATES = ["best_val_score", "val_score", "val_accuracy"]


def _scalar(val) -> float:
    if hasattr(val, "get"):
        return float(val.get("max", val.get("last", 0.0)))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _registry_fetch_path() -> str:
    return f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"


def _registry_link_target() -> str:
    """Path passed to ``artifact.link()`` (collection, without alias)."""
    return f"{ENTITY}/wandb-registry-{REGISTRY}/{COLLECTION}"


def _run_id_from_source(source_name: str | None) -> str | None:
    if not source_name:
        return None
    base = source_name.split(":")[0]
    if base.startswith("model-"):
        return base[len("model-") :]
    return None


def _describe_artifact(artifact) -> dict:
    source_name = getattr(artifact, "source_name", None)
    return {
        "name": f"{artifact.name}:{artifact.version}",
        "source_name": source_name,
        "run_id": _run_id_from_source(source_name),
        "source_project": getattr(artifact, "source_project", None),
    }


def _pick_metric(runs) -> str:
    for candidate in METRIC_CANDIDATES:
        if any(candidate in r.summary for r in runs):
            return candidate
    raise RuntimeError("No recognisable validation metric found in runs")


def _run_metric(api: wandb.Api, run_id: str | None, project: str | None, metric: str):
    """Return (score, project_used) for a run id, or (None, None) if unavailable."""
    if not run_id:
        return None, None

    projects = [p for p in (project, PROJECT, "radiocovid", "radiologist") if p]
    seen: set[str] = set()
    for proj in projects:
        if proj in seen:
            continue
        seen.add(proj)
        try:
            run = api.run(f"{ENTITY}/{proj}/{run_id}")
            if metric in run.summary:
                return _scalar(run.summary[metric]), proj
        except Exception:
            continue
    return None, None


def _print_run_block(
    title: str,
    run_id: str | None,
    metric: str,
    score,
    project: str | None,
    artifact_info: dict,
):
    print(title)
    print(f"  run_id       : {run_id or 'unknown'}")
    if score is not None:
        print(f"  {metric:<14}: {score:.4f}")
    else:
        print(f"  {metric:<14}: (unavailable — run not found or metric missing)")
    if project:
        print(f"  project      : {project}")
    print(f"  artifact     : {artifact_info.get('name', '—')}")
    print(f"  source       : {artifact_info.get('source_name', '—')}")


def find_best_artifact(api: wandb.Api):
    """Return (artifact, run_id, metric, score) for the best run with a model artifact."""
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    if not runs:
        raise RuntimeError(f"No runs found in {ENTITY}/{PROJECT}")

    metric = _pick_metric(runs)

    sorted_runs = sorted(
        [r for r in runs if metric in r.summary],
        key=lambda r: _scalar(r.summary[metric]),
        reverse=True,
    )

    for run in sorted_runs:
        score = _scalar(run.summary[metric])
        for artifact in run.logged_artifacts():
            if "model" in artifact.name:
                return artifact, run.id, metric, score

    raise RuntimeError(
        f"No model artifact found in any run of {ENTITY}/{PROJECT}. "
        "Ensure training ran with log_model=True."
    )


def get_current_production(api: wandb.Api):
    """Return the artifact currently tagged with the production alias, or None."""
    try:
        return api.artifact(_registry_fetch_path())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote the best W&B training artifact to the Model Registry."
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Link the artifact and move the registry alias (default: dry-run only).",
    )
    args = parser.parse_args()

    print(f"Entity    : {ENTITY}")
    print(f"Project   : {ENTITY}/{PROJECT}")
    print(f"Registry  : wandb-registry-{REGISTRY}/{COLLECTION}")
    print(f"Alias     : {ALIAS}")
    print()

    api = wandb.Api(overrides={"entity": ENTITY})

    print("Scanning runs for best model artifact…")
    artifact, run_id, metric, score = find_best_artifact(api)
    candidate = _describe_artifact(artifact)

    print()
    current = get_current_production(api)
    if current is None:
        print("Current production model in registry:")
        print("  (none — registry alias not set or not accessible)")
    else:
        current_info = _describe_artifact(current)
        current_score, current_project = _run_metric(
            api,
            current_info["run_id"],
            current_info["source_project"],
            metric,
        )
        _print_run_block(
            "Current production model in registry:",
            current_info["run_id"],
            metric,
            current_score,
            current_project,
            current_info,
        )
    print()

    _print_run_block(
        "Best candidate:",
        run_id,
        metric,
        score,
        PROJECT,
        candidate,
    )

    if current is not None:
        current_info = _describe_artifact(current)
        current_score, _ = _run_metric(
            api,
            current_info["run_id"],
            current_info["source_project"],
            metric,
        )
        if current_score is not None:
            delta = score - current_score
            sign = "+" if delta >= 0 else ""
            print()
            print(f"Comparison ({metric}):")
            print(f"  candidate vs production: {sign}{delta:.4f}")
            if delta > 0:
                print("  → candidate is better on this metric")
            elif delta < 0:
                print("  → production is better — promotion would be a downgrade")
            else:
                print("  → same score — check other criteria before promoting")
    print()

    link_target = _registry_link_target()
    if not args.promote:
        print("DRY-RUN — no changes made.")
        print(f"Would link {candidate['name']} → {link_target}  alias={ALIAS!r}")
        print("Run with --promote to apply.")
        return 0

    if current is not None:
        current_info = _describe_artifact(current)
        if current_info["source_name"] == candidate["source_name"]:
            print("Already in production — nothing to do.")
            return 0

    print(f"Linking to registry: {link_target}  alias={ALIAS!r}")
    artifact.link(target_path=link_target, aliases=[ALIAS])

    print()
    print(f"Done! '{COLLECTION}:{ALIAS}' now points to run {run_id}.")
    print(
        f"Registry UI: https://wandb.ai/{ENTITY}/registry/model"
        f"?selectionPath={ENTITY}%2F{COLLECTION}"
    )
    print("Reload inference:  curl -X POST http://localhost:8000/reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
