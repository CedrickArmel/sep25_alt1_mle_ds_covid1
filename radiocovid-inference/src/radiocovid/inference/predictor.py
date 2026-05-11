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
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ["covid", "normal"]

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)


class Predictor:
    """Load a trained checkpoint and run inference on images."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        classes: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu",
        transform=None,
    ):
        self.classes = classes or DEFAULT_CLASSES
        self.device = torch.device(device) if isinstance(device, str) else device
        self.transform = transform or DEFAULT_TRANSFORM

        self.model = self._build_model(len(self.classes))
        self._load_checkpoint(Path(checkpoint_path))
        self.model.train(mode=False)  # set to inference mode

    def _build_model(self, num_classes: int) -> torch.nn.Module:
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.to(self.device)
        return model

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = {
            k.replace("net.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("net.")
        }
        self.model.load_state_dict(state_dict)
        logger.info("Checkpoint loaded successfully")

    def predict(
        self,
        image: Union[Path, str, bytes, Image.Image],
    ) -> Tuple[str, float]:
        """Run inference on a single image.

        Args:
            image: File path, raw bytes, or PIL Image.

        Returns:
            Tuple of (predicted class label, confidence score).
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image)).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        x = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs.max().item()

        return self.classes[pred_idx], confidence

    def predict_batch(
        self,
        images: List[Union[Path, str, bytes, Image.Image]],
    ) -> List[Tuple[str, float]]:
        """Run inference on a list of images."""
        return [self.predict(img) for img in images]
