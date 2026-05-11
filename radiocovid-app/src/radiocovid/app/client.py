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

from typing import Any

import httpx


class InferenceClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def health(self) -> dict[str, Any]:
        try:
            resp = httpx.get(f"{self._base_url}/health", timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Inference service unreachable at {self._base_url}"
            ) from exc

    def predict(
        self, image_bytes: bytes, filename: str = "image.png"
    ) -> dict[str, Any]:
        try:
            resp = httpx.post(
                f"{self._base_url}/predict",
                files={"file": (filename, image_bytes, "image/png")},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Inference service unreachable at {self._base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Prediction failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
