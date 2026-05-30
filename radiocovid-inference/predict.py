import logging
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
CLASSES = ["NORMAL", "COVID"]


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
    weights_only=False  # ✅ CLAVE
)

    state_dict = {
        k.replace("net.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }

    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded ✅")


# -------------------------------
# API HELPERS ✅
# -------------------------------
def load_model():
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "models" / "model.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(len(CLASSES), device)
    load_checkpoint(model, checkpoint_path, device)

    return model, device


def get_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )


def predict_image(model, image: Image.Image, transform, device):
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    return CLASSES[pred_idx], confidence