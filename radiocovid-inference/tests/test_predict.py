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
from radiocovid.inference.predict import (
    CLASSES,
    build_model,
    get_transform,
    load_checkpoint,
)
from radiocovid.inference.predict import predict as run_predict
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

    def test_accepts_pil_image_directly(self, tmp_png):
        from PIL import Image as PilImage

        img = PilImage.open(tmp_png).convert("RGB")
        model = build_model(num_classes=len(CLASSES), device=torch.device("cpu"))
        label, confidence = run_predict(
            model, img, self._transform(), torch.device("cpu")
        )
        assert label in CLASSES
        assert 0.0 <= confidence <= 1.0


# --------------------------------------------------------------------------- #
# load_model (registry)                                                        #
# --------------------------------------------------------------------------- #


class TestRegistryArtifact:
    def test_fetches_artifact_from_registry_path(self, monkeypatch):
        import wandb
        from radiocovid.inference import predict as predict_module

        class FakeArtifact:
            name = "radiocovid-classifier"
            version = "v0"
            source_name = "model-5mr8ud16:v0"
            source_project = "radiocovid"

        captured = {}

        class FakeApi:
            def __init__(self, overrides=None):
                pass

            def artifact(self, path):
                captured["path"] = path
                return FakeArtifact()

        monkeypatch.setattr(wandb, "Api", FakeApi)

        artifact, meta = predict_module._registry_artifact(
            "my-entity", "model", "radiocovid-classifier", "production"
        )

        assert (
            captured["path"] == "wandb-registry-model/radiocovid-classifier:production"
        )
        assert artifact.name == "radiocovid-classifier"
        assert meta["run_id"] == "5mr8ud16"
        assert meta["registry_model"] == "radiocovid-classifier"
        assert meta["registry_alias"] == "production"
        assert meta["registry_artifact"] == "radiocovid-classifier:v0"


# --------------------------------------------------------------------------- #
# get_transform                                                                #
# --------------------------------------------------------------------------- #


class TestGetTransform:
    def test_returns_compose(self):
        assert isinstance(get_transform(), transforms.Compose)

    def test_output_is_float_tensor(self, tmp_png):
        from PIL import Image as PilImage

        img = PilImage.open(tmp_png).convert("RGB")
        out = get_transform()(img)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32

    def test_output_has_three_channels(self, tmp_png):
        from PIL import Image as PilImage

        img = PilImage.open(tmp_png).convert("RGB")
        out = get_transform()(img)
        assert out.shape[0] == 3
