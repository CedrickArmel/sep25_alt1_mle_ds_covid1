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
CLASSES = [
    "NORMAL",
    "ABNORMAL",
]  # default fallback; overridden at load time from checkpoint


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
def _run_id_from_artifact_name(artifact_name: str) -> str | None:
    """Extract run id from artifact names like ``model-5mr8ud16`` or ``model-5mr8ud16:v0``."""
    base = artifact_name.split(":")[0]
    prefix = "model-"
    if base.startswith(prefix):
        return base[len(prefix) :]
    return None


def _source_artifact_paths(entity: str, registry_artifact) -> list[str]:
    """Build candidate paths for the original training artifact linked in the registry."""
    source_name = getattr(registry_artifact, "source_name", None)
    if not source_name:
        return []

    projects: list[str] = []
    source_project = getattr(registry_artifact, "source_project", None)
    if source_project:
        projects.append(source_project)
    for candidate in (
        os.environ.get("WANDB_REGISTRY_SOURCE_PROJECT"),
        "radiocovid",
        os.environ.get("WANDB_PROJECT", "radiologist"),
    ):
        if candidate and candidate not in projects:
            projects.append(candidate)

    paths: list[str] = []
    for project in projects:
        if "/" in source_name:
            paths.append(source_name)
        else:
            paths.append(f"{entity}/{project}/{source_name}")
    return list(dict.fromkeys(paths))


def _download_artifact(api, entity: str, registry_artifact):
    """Download registry artifact, falling back to the source training artifact if needed."""
    try:
        return Path(registry_artifact.download(skip_cache=True)), "registry"
    except Exception as exc:
        if "does not have access" not in str(exc).lower():
            raise

        logger.warning(
            "Registry download denied (permissions). Trying source training artifact…"
        )
        for path in _source_artifact_paths(entity, registry_artifact):
            try:
                logger.info(f"Trying source artifact: {path}")
                source = api.artifact(path)
                return Path(source.download()), path
            except Exception as src_exc:
                logger.warning(f"Source artifact unavailable at {path}: {src_exc}")

        raise RuntimeError(
            "Cannot download the registry artifact (permission denied) and no source "
            "artifact was accessible. Ask a teammate to grant registry download access, "
            "or set WANDB_REGISTRY_SOURCE_PROJECT to the project that owns the model."
        ) from exc


def _registry_artifact(entity: str, registry: str, registry_model: str, alias: str):
    """Fetch a model artifact from the W&B Model Registry.

    Returns (artifact, metadata). Raises on missing registry entry or API error.

    W&B registry path format (see docs):
        wandb-registry-{registry}/{collection}:{alias}

    The team entity is passed via ``Api(overrides={"entity": ...})``, not in the path.
    Set ``WANDB_REGISTRY_ARTIFACT`` to the full path copied from the W&B UI to skip guessing.
    """
    import wandb

    api = wandb.Api(overrides={"entity": entity})
    override_path = os.environ.get("WANDB_REGISTRY_ARTIFACT")
    if override_path:
        path = override_path
    else:
        path = f"wandb-registry-{registry}/{registry_model}:{alias}"
    logger.info(f"Fetching model from W&B registry: {path} (entity={entity})")
    artifact = api.artifact(path)

    source_name = getattr(artifact, "source_name", None)
    run_id = _run_id_from_artifact_name(source_name or artifact.name)
    meta = {
        "run_id": run_id or "unknown",
        "entity": entity,
        "registry": registry,
        "registry_model": registry_model,
        "registry_alias": alias,
        "registry_artifact": f"{artifact.name}:{artifact.version}",
        "source_name": source_name,
        "source_project": getattr(artifact, "source_project", None),
        "classes": CLASSES,
    }
    logger.info(
        "Registry artifact resolved ✅  "
        f"registry={meta['registry_artifact']}  source={source_name}  run_id={meta['run_id']}"
    )
    return artifact, meta


# -------------------------------
# API HELPERS ✅
# -------------------------------
def load_model():
    global CLASSES
    # Local-path override: set MODEL_CKPT_PATH to skip W&B registry lookup entirely.
    local_ckpt = os.environ.get("MODEL_CKPT_PATH")
    if local_ckpt:
        checkpoint_path = Path(local_ckpt)
        if not checkpoint_path.exists():
            raise RuntimeError(f"MODEL_CKPT_PATH not found: {checkpoint_path}")
        meta = {
            "run_id": "local",
            "entity": "local",
            "registry": "local",
            "registry_model": "local",
            "registry_alias": "local",
            "registry_artifact": checkpoint_path.name,
            "source_name": checkpoint_path.name,
            "source_project": "local",
            "classes": CLASSES,
            "downloaded_from": "local",
        }
        logger.info(f"Loading model from local path: {checkpoint_path}")
    else:
        entity = os.environ.get("WANDB_ENTITY", "yebouetc")
        registry = os.environ.get("WANDB_REGISTRY", "radiocovid-classifier")
        registry_model = os.environ.get("WANDB_REGISTRY_MODEL", "radiocovid-classifier")
        registry_alias = os.environ.get("WANDB_REGISTRY_ALIAS", "production")

        artifact, meta = _registry_artifact(
            entity, registry, registry_model, registry_alias
        )

        import wandb

        api = wandb.Api(overrides={"entity": entity})
        artifact_dir, download_from = _download_artifact(api, entity, artifact)
        meta["downloaded_from"] = download_from

        ckpt_files = list(artifact_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise RuntimeError(
                f"No .ckpt file found after download from {download_from}"
            )

        checkpoint_path = ckpt_files[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read num_classes directly from the checkpoint to avoid size mismatches
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net_state = {
        k.replace("net.", ""): v
        for k, v in raw_ckpt["state_dict"].items()
        if k.startswith("net.")
    }
    num_classes = net_state["fc.weight"].shape[0]
    logger.info(f"Detected {num_classes} classes from checkpoint")

    # Build generic class labels if the checkpoint doesn't match our default list
    if num_classes != len(CLASSES):
        CLASSES = [f"class_{i}" for i in range(num_classes)]
        logger.warning(
            f"CLASSES overridden to {CLASSES} — update predict.py with real label names"
        )
    meta["classes"] = CLASSES

    model = build_model(num_classes, device)
    load_checkpoint(model, checkpoint_path, device)

    return model, device, meta


def get_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.502, 0.502, 0.502], std=[0.255, 0.255, 0.255]),
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
