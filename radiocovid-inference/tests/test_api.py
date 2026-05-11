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

from unittest.mock import MagicMock, patch

import pytest
import radiocovid.inference.api as api_module
from fastapi.testclient import TestClient
from radiocovid.inference.api import app


@pytest.fixture(autouse=True)
def reset_predictor():
    api_module._predictor = None
    yield
    api_module._predictor = None


@pytest.fixture
def loaded_client():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = ("covid", 0.95)
    api_module._predictor = mock_predictor
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def unloaded_client():
    return TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_health_returns_ok(self, loaded_client):
        with patch("radiocovid.inference.api.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(ckpt="models/model.ckpt")
            resp = loaded_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_includes_ckpt(self, loaded_client):
        with patch("radiocovid.inference.api.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(ckpt="models/best.ckpt")
            resp = loaded_client.get("/health")
        assert resp.json()["ckpt"] == "models/best.ckpt"


class TestPredict:
    def test_predict_with_png_returns_json(self, tmp_png, loaded_client):
        with open(tmp_png, "rb") as f:
            resp = loaded_client.post(
                "/predict",
                files={"file": ("sample.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "label" in body
        assert "confidence" in body

    def test_predict_label_and_confidence_types(self, tmp_png, loaded_client):
        with open(tmp_png, "rb") as f:
            resp = loaded_client.post(
                "/predict",
                files={"file": ("sample.png", f, "image/png")},
            )
        body = resp.json()
        assert isinstance(body["label"], str)
        assert isinstance(body["confidence"], float)

    def test_predict_model_not_loaded_returns_503(self, tmp_png, unloaded_client):
        with open(tmp_png, "rb") as f:
            resp = unloaded_client.post(
                "/predict",
                files={"file": ("sample.png", f, "image/png")},
            )
        assert resp.status_code == 503

    def test_predict_non_image_returns_400(self, loaded_client):
        resp = loaded_client.post(
            "/predict",
            files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
        )
        assert resp.status_code == 400

    def test_predict_empty_file_returns_400(self, loaded_client):
        resp = loaded_client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_predict_propagates_prediction_error(self, tmp_png, loaded_client):
        api_module._predictor.predict.side_effect = ValueError("bad image")
        with open(tmp_png, "rb") as f:
            resp = loaded_client.post(
                "/predict",
                files={"file": ("sample.png", f, "image/png")},
            )
        assert resp.status_code == 422
