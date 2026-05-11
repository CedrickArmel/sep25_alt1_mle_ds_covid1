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

import httpx
import pytest

from radiocovid.app.client import InferenceClient


def _mock_transport(status: int, body: dict):
    def handler(request):
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


class TestInferenceClientPredict:
    def test_returns_label_and_confidence(self, tmp_png):
        transport = _mock_transport(200, {"label": "covid", "confidence": 0.95})
        client = InferenceClient("http://fake", timeout=5.0)
        client._base_url = "http://fake"

        with httpx.Client(transport=transport) as http:
            resp = http.post(
                "http://fake/predict",
                files={"file": ("img.png", tmp_png.read_bytes(), "image/png")},
            )
        result = resp.json()
        assert result["label"] == "covid"
        assert result["confidence"] == pytest.approx(0.95)

    def test_raises_on_server_error(self, tmp_png):
        transport = _mock_transport(500, {"detail": "internal error"})
        client = InferenceClient.__new__(InferenceClient)
        client._base_url = "http://fake"
        client._timeout = 5.0

        with httpx.Client(transport=transport) as http:
            resp = http.post(
                "http://fake/predict",
                files={"file": ("img.png", tmp_png.read_bytes(), "image/png")},
            )
        assert resp.status_code == 500

    def test_connect_error_raises_connection_error(self):
        client = InferenceClient("http://localhost:19999", timeout=1.0)
        with pytest.raises((ConnectionError, Exception)):
            client.health()


class TestInferenceClientHealth:
    def test_health_returns_dict(self):
        transport = _mock_transport(200, {"status": "ok", "ckpt": "models/model.ckpt"})

        with httpx.Client(transport=transport) as http:
            resp = http.get("http://fake/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert "ckpt" in body
