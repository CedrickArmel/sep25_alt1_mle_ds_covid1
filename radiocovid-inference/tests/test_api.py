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

import io
from unittest.mock import patch

import numpy as np
import pytest
import radiocovid.inference.api as _api_module
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image
from radiocovid.inference.api import app
from torchvision import transforms

_META = {
    "run_id": "5mr8ud16",
    "entity": "test_entity",
    "registry": "Radiocovid-classifier",
    "registry_model": "radiocovid-classifier",
    "registry_alias": "production",
    "registry_artifact": "radiocovid-classifier:v0",
    "source_name": "model-5mr8ud16:v0",
    "source_project": "radiocovid",
    "downloaded_from": "registry",
    "classes": ["NORMAL", "ABNORMAL"],
}


class _FixedModel(nn.Module):
    """Always predicts class 0 (NORMAL) with high confidence."""

    def forward(self, x):
        return torch.tensor([[10.0, -10.0]]).expand(x.shape[0], -1)


def _basic_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


@pytest.fixture(autouse=True)
def isolate_state():
    """Guarantee a clean _state before and after every test."""
    _api_module._state.clear()
    yield
    _api_module._state.clear()


_TEST_API_KEY = "test-secret-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", _TEST_API_KEY)
    model = _FixedModel()
    model.eval()
    with (
        patch(
            "radiocovid.inference.api.load_model",
            return_value=(model, torch.device("cpu"), _META),
        ),
        patch(
            "radiocovid.inference.api.get_transform", return_value=_basic_transform()
        ),
    ):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def png_bytes():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# /health                                                                      #
# --------------------------------------------------------------------------- #


class TestHealthEndpoint:
    def test_returns_ok_when_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_returns_503_when_model_not_loaded(self, client):
        _api_module._state.clear()
        resp = client.get("/health")
        assert resp.status_code == 503


# --------------------------------------------------------------------------- #
# /info                                                                        #
# --------------------------------------------------------------------------- #


class TestInfoEndpoint:
    def test_returns_meta_when_model_loaded(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == _META["run_id"]
        assert data["registry_model"] == _META["registry_model"]
        assert data["registry_alias"] == _META["registry_alias"]

    def test_returns_503_when_model_not_loaded(self, client):
        _api_module._state.clear()
        resp = client.get("/info")
        assert resp.status_code == 503


# --------------------------------------------------------------------------- #
# /reload                                                                      #
# --------------------------------------------------------------------------- #


class TestReloadEndpoint:
    def test_returns_reloaded_status(self, client):
        with (
            patch(
                "radiocovid.inference.api.load_model",
                return_value=(_FixedModel(), torch.device("cpu"), _META),
            ),
            patch(
                "radiocovid.inference.api.get_transform",
                return_value=_basic_transform(),
            ),
        ):
            resp = client.post("/reload", headers={"X-API-Key": _TEST_API_KEY})
        assert resp.status_code == 200
        assert resp.json()["status"] == "reloaded"

    def test_reload_repopulates_meta(self, client):
        updated_meta = {
            **_META,
            "run_id": "run_updated",
            "downloaded_from": "theradiologist/radiologist/model-run_updated:v0",
        }
        with (
            patch(
                "radiocovid.inference.api.load_model",
                return_value=(_FixedModel(), torch.device("cpu"), updated_meta),
            ),
            patch(
                "radiocovid.inference.api.get_transform",
                return_value=_basic_transform(),
            ),
        ):
            resp = client.post("/reload", headers={"X-API-Key": _TEST_API_KEY})
        assert resp.json()["run_id"] == "run_updated"

    def test_returns_403_without_api_key(self, client):
        resp = client.post("/reload")
        assert resp.status_code == 403

    def test_returns_403_with_wrong_api_key(self, client):
        resp = client.post("/reload", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# /predict                                                                     #
# --------------------------------------------------------------------------- #


class TestPredictEndpoint:
    def test_returns_label_and_probability(self, client, png_bytes):
        resp = client.post(
            "/predict",
            files={"file": ("img.png", png_bytes, "image/png")},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "label" in body
        assert "probability" in body

    def test_label_is_valid_class(self, client, png_bytes):
        resp = client.post(
            "/predict",
            files={"file": ("img.png", png_bytes, "image/png")},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert resp.json()["label"] in ["NORMAL", "ABNORMAL"]

    def test_probability_in_unit_interval(self, client, png_bytes):
        resp = client.post(
            "/predict",
            files={"file": ("img.png", png_bytes, "image/png")},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        prob = resp.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_returns_400_for_invalid_image(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("bad.bin", b"not-an-image", "application/octet-stream")},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert resp.status_code == 400

    def test_returns_403_without_api_key(self, client, png_bytes):
        resp = client.post(
            "/predict", files={"file": ("img.png", png_bytes, "image/png")}
        )
        assert resp.status_code == 403

    def test_returns_403_with_wrong_api_key(self, client, png_bytes):
        resp = client.post(
            "/predict",
            files={"file": ("img.png", png_bytes, "image/png")},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403
