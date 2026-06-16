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

import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Classes
# -------------------------------
CLASSES = ["NORMAL", "ABNORMAL"]


# -------------------------------
# Model
# -------------------------------
def build_model(num_classes: int, device: torch.device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device):
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = {
        k.replace("net.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }

    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded ✅")


# -------------------------------
# W&B helpers
# -------------------------------
def _best_run_artifact(entity: str, project: str):
    """Connect to W&B, find the run with the best validation metric,
    and return (artifact, metadata). Raises RuntimeError if nothing is found."""
    import wandb
    from wandb_download_ckpt import choose_metric, download_artifact, find_model_artifact

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    if not runs:
        raise RuntimeError(f"No runs found in W&B project {entity}/{project}")

    metric = choose_metric(runs)
    if metric is None:
        raise RuntimeError("No recognisable validation metric found in W&B runs")

    valid_runs = [r for r in runs if metric in r.summary]
    best_run = max(valid_runs, key=lambda r: r.summary[metric])
    logger.info(f"Best run: {best_run.id}  ({metric}={best_run.summary[metric]:.4f})")

    artifact = download_artifact(api, entity, project, best_run.id)
    if artifact is None:
        artifact = find_model_artifact(best_run)
    if artifact is None:
        raise RuntimeError(f"No model artifact found for run {best_run.id}")

    meta = {
        "run_id": best_run.id,
        "metric": metric,
        "metric_value": round(best_run.summary[metric], 4),
        "entity": entity,
        "project": project,
        "classes": CLASSES,
    }
    return artifact, meta


# -------------------------------
# API HELPERS ✅
# -------------------------------
def load_model():
    entity = os.environ.get("WANDB_ENTITY", "yebouetc")
    project = os.environ.get("WANDB_PROJECT", "radiocovid")

    logger.info(f"Fetching best model from W&B: {entity}/{project}")
    artifact, meta = _best_run_artifact(entity, project)

    artifact_dir = Path(artifact.download())
    ckpt_files = list(artifact_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise RuntimeError(f"No .ckpt file found in artifact for run {meta['run_id']}")

    checkpoint_path = ckpt_files[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASSES), device)
    load_checkpoint(model, checkpoint_path, device)

    return model, device, meta


def get_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[128, 128, 128], std=[65, 65, 65]),
        ]
    )


def predict(model, image: "Image.Image | Path", transform, device):
    if isinstance(image, Path):
        image = Image.open(image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    return CLASSES[pred_idx], confidence
