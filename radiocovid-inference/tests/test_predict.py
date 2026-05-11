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
from predict import CLASSES, build_model, load_checkpoint
from predict import predict as run_predict
from torchvision import transforms

# --------------------------------------------------------------------------- #
# build_model                                                                  #
# --------------------------------------------------------------------------- #


class TestBuildModel:
    def test_fc_replaced(self):
        device = torch.device("cpu")
        model = build_model(num_classes=4, device=device)
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.out_features == 4

    def test_inference_mode(self):
        model = build_model(num_classes=2, device=torch.device("cpu"))
        assert not model.training

    def test_on_cpu(self):
        model = build_model(num_classes=2, device=torch.device("cpu"))
        for p in model.parameters():
            assert p.device.type == "cpu"


# --------------------------------------------------------------------------- #
# load_checkpoint                                                              #
# --------------------------------------------------------------------------- #


class TestLoadCheckpoint:
    def test_strips_net_prefix_and_loads(self, tiny_ckpt):
        ckpt_path, original_state = tiny_ckpt
        model = nn.Linear(3, 2)
        load_checkpoint(model, ckpt_path, device=torch.device("cpu"))
        for k, v in model.state_dict().items():
            assert torch.allclose(v, original_state[k])

    def test_ignores_non_net_keys(self, tiny_ckpt):
        ckpt_path, _ = tiny_ckpt
        model = nn.Linear(3, 2)
        load_checkpoint(model, ckpt_path, device=torch.device("cpu"))


# --------------------------------------------------------------------------- #
# predict                                                                      #
# --------------------------------------------------------------------------- #


class TestPredict:
    def _transform(self):
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    def test_returns_class_and_confidence(self, tmp_png, monkeypatch):
        torch.manual_seed(0)
        logits = torch.tensor([[10.0, -10.0]])

        class FixedModel(nn.Module):
            def forward(self, x):
                return logits.expand(x.shape[0], -1)

        model = FixedModel()
        label, confidence = run_predict(
            model, tmp_png, self._transform(), torch.device("cpu")
        )
        assert label == CLASSES[0]
        assert 0.0 <= confidence <= 1.0

    def test_confidence_in_unit_interval(self, tmp_png):
        model = build_model(num_classes=len(CLASSES), device=torch.device("cpu"))
        label, confidence = run_predict(
            model, tmp_png, self._transform(), torch.device("cpu")
        )
        assert 0.0 <= confidence <= 1.0

    def test_label_is_valid_class(self, tmp_png):
        model = build_model(num_classes=len(CLASSES), device=torch.device("cpu"))
        label, _ = run_predict(model, tmp_png, self._transform(), torch.device("cpu"))
        assert label in CLASSES
