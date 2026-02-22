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

from glob import glob
from pathlib import Path

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
