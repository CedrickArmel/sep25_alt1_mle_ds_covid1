from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

# ========== 1. RUTAS BASE ==========

# Ruta al proyecto: /home/ubuntu/sep25_alt1_mle_ds_covid1
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ruta al dataset crudo
RAW_DATA = PROJECT_ROOT / "data" / "01_raw" / "COVID-19_Radiography_Dataset"

REPS = ["COVID", "Lung_Opacity", "Viral Pneumonia", "Normal"]

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DATA:", RAW_DATA)
print("RAW_DATA existe:", RAW_DATA.exists())

# ========== 2. CARGAR SOLO RUTAS (DEBUG) ==========

image_paths_dict = {}

for rep in REPS:
    key = rep.lower().replace(" ", "_")
    img_dir = RAW_DATA / rep / "images"

    paths = sorted(glob(str(img_dir / "*.png")))

    image_paths_dict[key] = paths

    print(f"\nClase {key}")
    print("  Carpeta imágenes:", img_dir)
    print("  ¿Existe carpeta?:", img_dir.exists())
    print("  Número de imágenes encontradas:", len(paths))
    print("  Ejemplos:", paths[:3])

# Comprobación: ¿tenemos rutas?
if not any(len(v) > 0 for v in image_paths_dict.values()):
    raise RuntimeError("❌ No se encontraron imágenes .png. Revisa la ruta RAW_DATA.")

print("\n✅ Rutas bien cargadas, fin del script (todavía sin balancear).")