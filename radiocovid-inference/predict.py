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

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# ===============================
# CONFIGURATION
# ===============================
CKPT_PATH = "/home/ubuntu/sep25_alt1_mle_ds_covid1/models/model.ckpt"
DATA_DIR = "/home/ubuntu/sep25_alt1_mle_ds_covid1/data/03_inference_images"
CLASSES = ["CLASSE_0", "CLASSE_1"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# TRANSFORMATIONS (identiques au training)
# ===============================
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

# ===============================
# CONSTRUIRE LE MODÈLE (ResNet50 )
# ===============================
print(" Construction ResNet50...")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.to(device)
model.eval()

# ===============================
# CHARGER LES POIDS DU CHECKPOINT
# ===============================
print("Chargement des poids...")
ckpt = torch.load(CKPT_PATH, map_location=device)

state_dict = {
    k.replace("net.", ""): v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("net.")
}

model.load_state_dict(state_dict)
print("Poids chargés avec succès")

# ===============================
# IMAGE ALÉATOIRE
# ===============================
images = list(Path(DATA_DIR).rglob("*.png"))
if not images:
    raise RuntimeError("❌ Aucune image trouvée")

img_path = random.choice(images)
print(f"Image : {img_path}")

img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# ===============================
# PRÉDICTION
# ===============================
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs.max().item()

print("\nRÉSULTAT")
print("--------------------")
print(f"Prédiction : {CLASSES[pred]}")
print(f"Confiance  : {confidence:.3f}")
