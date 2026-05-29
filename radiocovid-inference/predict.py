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
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Constants
# -------------------------------
CLASSES = ["CLASSE_0", "CLASSE_1"]


# -------------------------------
# Model utilities
# -------------------------------
def build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device):
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = {
        k.replace("net.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }

    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded successfully")


# -------------------------------
# Prediction
# -------------------------------
def predict(model, image_path: Path, transform, device):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    return CLASSES[pred_idx], confidence


# -------------------------------
# Main
# -------------------------------
def main():
    project_root = Path(__file__).resolve().parents[1]

    checkpoint_path = project_root / "models" / "model.ckpt"
    data_dir = project_root / "data" / "03_inference_images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    images = list(data_dir.rglob("*.png"))
    if not images:
        raise RuntimeError("No images found for inference")

    image_path = random.choice(images)
    logger.info("Selected image: %s", image_path)

    model = build_model(len(CLASSES), device)
    load_checkpoint(model, checkpoint_path, device)

    label, confidence = predict(model, image_path, transform, device)

    logger.info("Prediction: %s (confidence=%.3f)", label, confidence)


if __name__ == "__main__":
    main()
