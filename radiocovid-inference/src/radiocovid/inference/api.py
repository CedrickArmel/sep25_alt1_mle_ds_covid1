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

"""FastAPI inference service exposing /health and /predict."""

from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .checkpoints import resolve_checkpoint
from .config import get_config
from .predictor import Predictor

_predictor: Optional[Predictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    cfg = get_config()
    ckpt = resolve_checkpoint(cfg.ckpt)
    _predictor = Predictor(
        checkpoint_path=ckpt,
        classes=cfg.classes,
        device=cfg.device,
    )
    yield
    _predictor = None


app = FastAPI(title="RadioCOVID Inference API", lifespan=lifespan)


@app.get("/health")
def health():
    cfg = get_config()
    return {"status": "ok", "ckpt": cfg.ckpt}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        label, confidence = _predictor.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    return JSONResponse({"label": label, "confidence": round(confidence, 4)})


def main():
    cfg = get_config()
    uvicorn.run(
        "radiocovid.inference.api:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
