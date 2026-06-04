from contextlib import asynccontextmanager
import io

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

load_dotenv()

from predict import get_transform, load_model
from predict import predict as run_predict

_state: dict = {}


def _load_into_state():
    model, device, meta = load_model()
    _state["model"] = model
    _state["device"] = device
    _state["transform"] = get_transform()
    _state["info"] = meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model from W&B...")
    _load_into_state()
    print("Model loaded ✅")
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


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


@app.post("/reload")
def reload():
    print("Reloading model from W&B...")
    _load_into_state()
    print("Model reloaded ✅")
    return {"status": "reloaded", **_state["info"]}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    label, confidence = run_predict(
        _state["model"], image, _state["transform"], _state["device"]
    )

    return {"label": label, "probability": round(confidence, 4)}
