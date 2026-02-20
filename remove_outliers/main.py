from remove_outliers import remove_outliers
from glob import glob
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import shutil

print(" Début du nettoyage des données")

# Chemins
DATA_ROOT = Path(__file__).parent / "data"
RAW_DATA = DATA_ROOT / "01_raw/COVID-19_Radiography_Dataset"
CLEAN_ROOT = DATA_ROOT / "02_Cleaned_data/COVID-19_Radiography_Dataset"
REPS = ["COVID", "Lung_Opacity", "Viral Pneumonia", "Normal"]

# Charger les données
images = defaultdict(list)
masks = defaultdict(list)
image_paths = defaultdict(list)
mask_paths = defaultdict(list)

print("\n Chargement des données...")
for rep in tqdm(REPS, desc="Classes"):
    class_key = rep.lower().replace(" ", "_")

    img_files = sorted(glob(str(RAW_DATA / rep / "images/*.png")))
    mask_files = sorted(glob(str(RAW_DATA / rep / "masks/*.png")))

    for img_path, mask_path in zip(img_files, mask_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        images[class_key].append(img)
        masks[class_key].append(mask)
        image_paths[class_key].append(img_path)
        mask_paths[class_key].append(mask_path)

print("\n Chargement terminé :")
for classe in images.keys():
    print(f"  {classe} → {len(images[classe])} images")

# 🔍 Détection des outliers
print("\n Détection des outliers...")
valid_indices = remove_outliers(images, masks)

print("\n Nettoyage terminé :")
for classe, indices in valid_indices.items():
    print(f"  {classe} → {len(indices)} images conservées")

#  Copie des images valides
print("\n Copie des images propres...")
for classe, indices in valid_indices.items():

    img_dest = CLEAN_ROOT / classe / "images"
    mask_dest = CLEAN_ROOT / classe / "masks"

    img_dest.mkdir(parents=True, exist_ok=True)
    mask_dest.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(indices, desc=f"Copie {classe}"):

        img_path = image_paths[classe][idx]
        mask_path = mask_paths[classe][idx]

        shutil.copy2(img_path, img_dest / Path(img_path).name)
        shutil.copy2(mask_path, mask_dest / Path(mask_path).name)

print("\n Terminé ! Images propres dans 02_Cleaned_data/")
