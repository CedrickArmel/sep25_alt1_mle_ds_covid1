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
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security.api_key import APIKeyHeader
from PIL import Image
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from radiocovid.inference.predict import get_transform, load_model
from radiocovid.inference.predict import predict as run_predict

load_dotenv()

_state: dict = {}

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_api_key(key: str | None = Depends(_API_KEY_HEADER)) -> None:
    expected = os.environ.get("API_KEY", "")
    if expected and key != expected:
        raise HTTPException(status_code=403, detail="Accès refusé")


def _load_into_state():
    model, device, meta = load_model()
    _state["model"] = model
    _state["device"] = device
    _state["transform"] = get_transform()
    _state["info"] = meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model from W&B Model Registry...")
    _load_into_state()
    print("Model loaded ✅")
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)

# Contador de predicciones por clase (NORMAL / ABNORMAL)
_predictions_counter = Counter(
    "radiocovid_predictions_total",
    "Nombre total de prédictions par label",
    ["label"],
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    if not _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/info")
def info():
    if not _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _state["info"]


@app.post("/reload", dependencies=[Depends(_check_api_key)])
def reload():
    print("Reloading model from W&B Model Registry...")
    _load_into_state()
    print("Model reloaded ✅")
    return {"status": "reloaded", **_state["info"]}


@app.post("/predict", dependencies=[Depends(_check_api_key)])
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide")

    if not _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    label, confidence = run_predict(
        _state["model"], image, _state["transform"], _state["device"]
    )

    _predictions_counter.labels(label=label).inc()

    return {"label": label, "probability": round(confidence, 4)}
