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

import torch
import torch.nn as nn
from radiocovid.inference.predictor import DEFAULT_CLASSES, Predictor
from torchvision import models


def _make_resnet50_ckpt(tmp_path, num_classes=2):
    """Build a minimal resnet50 checkpoint compatible with Predictor."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = {f"net.{k}": v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": state}, path)
    return path


class TestPredictor:
    def test_loads_checkpoint(self, tmp_path):
        ckpt = _make_resnet50_ckpt(tmp_path)
        p = Predictor(checkpoint_path=ckpt, classes=DEFAULT_CLASSES, device="cpu")
        assert p.model is not None

    def test_predict_returns_label_and_confidence(self, tmp_path, tmp_png):
        ckpt = _make_resnet50_ckpt(tmp_path)
        p = Predictor(checkpoint_path=ckpt, classes=DEFAULT_CLASSES, device="cpu")
        label, confidence = p.predict(tmp_png)
        assert label in DEFAULT_CLASSES
        assert 0.0 <= confidence <= 1.0

    def test_predict_from_bytes(self, tmp_path, tmp_png):
        ckpt = _make_resnet50_ckpt(tmp_path)
        p = Predictor(checkpoint_path=ckpt, classes=DEFAULT_CLASSES, device="cpu")
        image_bytes = tmp_png.read_bytes()
        label, confidence = p.predict(image_bytes)
        assert label in DEFAULT_CLASSES

    def test_predict_batch(self, tmp_path, tmp_png):
        ckpt = _make_resnet50_ckpt(tmp_path)
        p = Predictor(checkpoint_path=ckpt, classes=DEFAULT_CLASSES, device="cpu")
        results = p.predict_batch([tmp_png, tmp_png])
        assert len(results) == 2
        for label, conf in results:
            assert label in DEFAULT_CLASSES
            assert 0.0 <= conf <= 1.0
