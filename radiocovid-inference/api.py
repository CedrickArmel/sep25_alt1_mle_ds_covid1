from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from predict import load_model, get_transform, predict_image

app = FastAPI()

# variables globales (vacías al inicio)
model = None
device = None
transform = None


@app.on_event("startup")
def load_everything():
    global model, device, transform
    print("Loading model...")

    model, device = load_model()
    transform = get_transform()

    print("Model loaded ✅")


@app.get("/")
def home():
    return {"message": "Radiocovid API running ✅"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(model, image, transform, device)

    return {
        "label": label,
        "probability": round(confidence, 4)
    }