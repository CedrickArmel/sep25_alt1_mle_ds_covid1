"""Download a model checkpoint from Weights & Biases.

Simple script to download a checkpoint from a W&B run id or run name.

Example:
  python scripts/wandb_download_ckpt.py skilled-haze-6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def find_run(api: wandb.Api, entity: str, project: str, run_ref: str):
    run_path = f"{entity}/{project}/{run_ref}"
    try:
        return api.run(run_path)
    except Exception:
        pass

    print(f"Run not found by path {run_path}. Trying to search by run name...")
    try:
        runs = list(api.runs(f"{entity}/{project}", {"name": run_ref}))
    except Exception:
        runs = []

    if runs:
        print(f"Found {len(runs)} run(s) with name '{run_ref}'. Using run id {runs[0].id}.")
        return runs[0]

    # Last fallback: scan a few runs for exact name match
    print("Searching for exact run name among recent runs...")
    try:
        for run in api.runs(f"{entity}/{project}"):
            if run.name == run_ref:
                print(f"Found run by exact name: {run.id}")
                return run
    except Exception:
        pass

    raise RuntimeError(f"Run not found by id or name: {run_ref}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a checkpoint file from a W&B run by id or run name."
    )
    parser.add_argument("run_ref", help="W&B run id or run name")
    parser.add_argument("--entity", default="yebouetc", help="W&B entity/team")
    parser.add_argument("--project", default="radiocovid", help="W&B project")
    args = parser.parse_args()

    api = wandb.Api()

    print(f"Looking for run in {args.entity}/{args.project} with reference '{args.run_ref}'")
    try:
        run = find_run(api, args.entity, args.project, args.run_ref)
    except Exception as exc:
        print(f"Error: {exc}")
        print("Make sure the run exists, the project is correct, and you have access.")
        return 1

    ckpt_files = [f for f in run.files() if f.name.endswith(".ckpt")]
    if not ckpt_files:
        print(f"No .ckpt files found in run {run.id} ({run.name})")
        print("Available files:")
        for f in run.files():
            print(f"  - {f.name}")
        return 1

    selected_file = None
    for priority in ["best.ckpt", "last.ckpt"]:
        for f in ckpt_files:
            if priority in f.name:
                selected_file = f
                break
        if selected_file:
            break
    else:
        selected_file = ckpt_files[0]

    print(f"Downloading: {selected_file.name} from run {run.id} ({run.name})")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / f"{args.run_ref}.ckpt"

    downloaded = selected_file.download(root=str(models_dir), replace=True)
    downloaded_path = Path(downloaded)
    if downloaded_path.is_dir():
        downloaded_path = downloaded_path / selected_file.name

    if downloaded_path != output_path:
        downloaded_path.replace(output_path)

    print(f"✅ Checkpoint saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())