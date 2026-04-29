import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms, models

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
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

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
