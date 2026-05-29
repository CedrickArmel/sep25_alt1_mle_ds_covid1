from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms

# Ajusta estos nombres si tu modelo usa clases específicas.
CLASSES = ["CLASSE_0", "CLASSE_1"]

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)


def build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        k.replace("net.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }
    model.load_state_dict(state_dict)


def predict_image(model, image: Image.Image, transform, device):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    return CLASSES[pred_idx], float(confidence)


def get_model_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "models" / "model.ckpt"


MODEL_PATH = get_model_checkpoint_path()
model = build_model(len(CLASSES), device)
load_checkpoint(model, MODEL_PATH, device)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    label, confidence = predict_image(model, image, transform, device)
    return {"label": label, "confidence": confidence}
