# -*- coding: utf-8 -*-
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

import glob
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yaml
from PIL import Image

# =========================
# CONFIG APP
# =========================
st.set_page_config(
    page_title="RadioCovid – Projet Poumons", page_icon="🫁", layout="wide"
)

st.sidebar.title("🫁 Menu")
PAGES = ["EDA", "Rééquilibrage", "Modèle", "Résultats", "Prédiction", "Conclusion"]
page = st.sidebar.radio("Aller à :", PAGES)

# =========================
# CONSTANTES / CHEMINS
# =========================
ROOT = Path(".")
DATA_ROOT = ROOT / "data"
TRAIN_DIR = DATA_ROOT / "train"
DUMPS_DIR = DATA_ROOT / "00_dumps"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
NB_EDA_1 = ROOT / "notebooks/1_0_eda_radiography.ipynb"
NB_EDA_2 = ROOT / "notebooks/1.0_ta_eda_.ipynb"

MODULE_YAML = ROOT / "radiocovid-core/src/radiocovid/core/configs/module/default.yaml"
DATAMODULE_YAML = ROOT / "radiocovid-core/src/radiocovid/core/configs/datamodule.yaml"
REPORTS_DIR = ROOT / "reports"
REPORTS_FIG_DIR = REPORTS_DIR / "figures"


# =========================
# HELPERS
# =========================
@st.cache_data
def load_yaml_safe(path: Path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"_warning": f"Impossible de charger {path}: {e}"}


@st.cache_data
def list_classes(train_dir: Path):
    if not train_dir.exists():
        return []
    return [c for c in os.listdir(train_dir) if (train_dir / c).is_dir()]


@st.cache_data
def count_by_class(train_dir: Path):
    classes = list_classes(train_dir)
    return {c: len(list((train_dir / c).glob("*.png"))) for c in classes}


def get_example_images(folder: Path, n=3):
    imgs = sorted(glob.glob(str(folder / "*.png")))
    return imgs[:n]


@st.cache_data
def find_haralick_dumps(dumps_dir: Path):
    """
    Cherche fichiers: haralick_features_{feature}_{rep}.npy
    Retourne: dict[feature][rep] -> np.ndarray
    """
    results = {}
    if not dumps_dir.exists():
        return results
    files = sorted(dumps_dir.glob("haralick_features_*_*.npy"))
    for f in files:
        # parse nom
        # ex: haralick_features_entropy_covid.npy
        name = f.stem  # sans .npy
        parts = name.split("_")
        # ["haralick","features","{feature}","{rep}"]
        if len(parts) < 4:
            continue
        feature = parts[2]
        rep = "_".join(parts[3:])
        try:
            arr = np.load(f)
        except Exception:
            continue
        results.setdefault(feature, {})
        results[feature][rep] = arr
    return results


def plot_haralick_boxplots(hara: dict):
    if not hara:
        st.info("ℹ️ Aucun dump Haralick trouvé dans `data/00_dumps`.")
        return
    features = list(hara.keys())
    if not features:
        st.info("ℹ️ Pas de features Haralick disponibles.")
        return
    reps = list(next(iter(hara.values())).keys())
    data_list = []
    for ft in features:
        for rep in reps:
            # On agrège (moyenne) pour une boîte par classe
            vals = (
                np.mean(hara[ft][rep], axis=0)
                if hara[ft][rep].ndim > 1
                else hara[ft][rep]
            )
            vals = np.array(vals).ravel()
            for v in vals:
                data_list.append(
                    {
                        "Feature": ft.capitalize(),
                        "Classe": rep.upper(),
                        "Valeur": float(v),
                    }
                )
    if not data_list:
        st.info("ℹ️ Pas de données valides pour boxplots.")
        return
    df = pd.DataFrame(data_list)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Feature", y="Valeur", hue="Classe", ax=ax)
    ax.set_title("Distribution des features Haralick par classe")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Valeur (agrégée)")
    ax.legend(loc="best")
    st.pyplot(fig)


def plot_haralick_mean_curves(hara: dict):
    if not hara:
        return
    features = list(hara.keys())
    reps = list(next(iter(hara.values())).keys())
    cols = st.columns(2)
    for i, ft in enumerate(features):
        fig, ax = plt.subplots(figsize=(6, 4))
        for rep in reps:
            arr = hara[ft][rep]
            if arr.ndim == 2:
                # Si GLCM stockée comme (distance, directions*?)
                mean_curve = np.mean(arr, axis=1)
            else:
                mean_curve = np.array(arr).ravel()
            ax.plot(mean_curve, label=rep.upper())
        ax.set_title(f"{ft.capitalize()} – courbes moyennes")
        ax.set_xlabel("Distance (pixels)")
        ax.set_ylabel("Valeur")
        ax.legend()
        cols[i % 2].pyplot(fig)


@st.cache_data
def find_checkpoints(models_dir: Path):
    return sorted(models_dir.glob("*.ckpt"))


@st.cache_data
def scan_training_logs(logs_dir: Path):
    """
    Cherche des CSV ou métriques simples dans logs/
    """
    if not logs_dir.exists():
        return []
    metrics = list(logs_dir.rglob("*.csv"))
    return metrics


def render_resume_card():
    # Inyecta CSS solo una vez por sesión
    if not st.session_state.get("_resume_css_added", False):
        st.markdown(
            """
            <style>
                .rc-card{
                    background: #0f172a; /* slate-900 */
                    color: #e2e8f0;      /* slate-200 */
                    padding: 1.25rem 1.5rem;
                    border-radius: 12px;
                    border: 1px solid #334155; /* slate-700 */
                    box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset;
                }
                .rc-card h3, .rc-card h4{
                    margin-top: 0.2rem;
                    color: #f1f5f9; /* slate-100 */
                }
                .rc-card p{
                    margin: 0.2rem 0 0.8rem 0;
                }
                .rc-card ul{
                    margin: 0.2rem 0 0.8rem 1.1rem;
                }
                .rc-tag{
                    display:inline-block;
                    background:#1f2937; /* gray-800 */
                    color:#cbd5e1;      /* slate-300 */
                    padding: 2px 10px;
                    border-radius: 999px;
                    font-size: 0.85rem;
                    margin-right: 6px;
                    margin-bottom: 4px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_resume_css_added"] = True

    st.markdown(
        """
<div class="rc-card">

<h3>📌 Résumé des volumes du dataset</h3>

<p>L’ensemble de données <em>COVID‑19 Radiography</em> présente des volumes très variables selon les catégories, ce qui introduit un <strong>déséquilibre de classes important</strong> :</p>

<ul>
  <li><strong>COVID</strong>
    <ul>
      <li>Nombre d’images : <strong>3 616</strong></li>
      <li>Nombre de masques : <strong>3 616</strong></li>
      <li>Taille des images : <strong>299×299</strong></li>
      <li>Taille des masques : <strong>256×256</strong></li>
    </ul>
  </li>
  <li><strong>Lung Opacity</strong>
    <ul>
      <li>Nombre d’images : <strong>6 012</strong></li>
      <li>Nombre de masques : <strong>6 012</strong></li>
      <li>Taille des images : <strong>299×299</strong></li>
      <li>Taille des masques : <strong>256×256</strong></li>
    </ul>
  </li>
  <li><strong>Viral Pneumonia</strong>
    <ul>
      <li>Nombre d’images : <strong>1 345</strong></li>
      <li>Nombre de masques : <strong>1 345</strong></li>
      <li>Taille des images : <strong>299×299</strong></li>
      <li>Taille des masques : <strong>256×256</strong></li>
    </ul>
  </li>
  <li><strong>Normal</strong>
    <ul>
      <li>Nombre d’images : <strong>10 192</strong></li>
      <li>Nombre de masques : <strong>10 192</strong></li>
      <li>Taille des images : <strong>299×299</strong></li>
      <li>Taille des masques : <strong>256×256</strong></li>
    </ul>
  </li>
</ul>

<h4>🔎 Points importants à retenir</h4>
<ul>
  <li>Le dataset est <strong>largement dominé par la classe Normal</strong>, suivie de <em>Lung Opacity</em>.</li>
  <li>Les classes <strong>COVID</strong> et <strong>Viral Pneumonia</strong> sont <strong>sous‑représentées</strong>, ce qui peut biaiser l’entraînement.</li>
  <li>Toutes les images ont une taille homogène (<strong>299×299</strong>) tandis que les masques sont fournis en <strong>256×256</strong>.</li>
  <li>Cette différence de résolution implique un <strong>redimensionnement systématique</strong> lors du prétraitement.</li>
  <li>Le déséquilibre entre classes justifie une étape de <strong>rééquilibrage</strong> avant l’entraînement du modèle.</li>
</ul>

<div>
  <span class="rc-tag">Déséquilibre</span>
  <span class="rc-tag">Normalisation</span>
  <span class="rc-tag">Prétraitement</span>
  <span class="rc-tag">Rééquilibrage</span>
</div>

</div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 1) EDA — basé sur les figures enregistrées dans reports/
# =========================
if page == "EDA":
    st.title("🔍 EDA — Exploration des Données")

    st.markdown("""
    **Objectif** : présenter l’exploration du dataset (source Kaggle) et l’analyse texturale (GLCM/Haralick) à partir
    des figures **déjà générées** par le notebook d’EDA.

    **Dataset** : COVID‑19 Radiography Database (Kaggle)
    👉 https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    """)

    # -------- utilitaires locaux --------
    def first_existing(*paths: Path) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    def list_harlick_figs() -> list[Path]:
        # On cherche d'abord dans reports/, puis dans reports/figures/
        patterns = [
            str(REPORTS_DIR / "harlick*.png"),
            str(REPORTS_FIG_DIR / "harlick*.png"),
        ]
        files = []
        for pat in patterns:
            files.extend(sorted(Path().glob(pat)))
        # Déduplication en conservant l'ordre
        seen = set()
        ordered = []
        for f in files:
            if f not in seen:
                ordered.append(f)
                seen.add(f)
        return ordered

    # -------- 1) Répartition / volumétrie --------
    st.header("📦 Volumétrie & répartition des classes")
    dist_fig = first_existing(
        REPORTS_DIR / "images.png",
        REPORTS_FIG_DIR / "images.png",
    )

    if dist_fig is None:
        st.warning(
            "⚠️ Aucune figure de répartition trouvée (recherché `reports/images.png`)."
        )
    else:
        st.image(
            str(dist_fig),
            caption="Distribution du dataset par pathologie",
            use_container_width=True,
        )
        st.markdown("""
        **Lecture** : la figure récapitule la répartition des images par catégorie.
        Elle permet de vérifier d’un coup d’œil les **déséquilibres de classes** (ex. sur‑représentation de *Normal*).
        """)

    render_resume_card()

    # -------- 2) Analyse texturale (GLCM / Haralick) --------
    st.header("🧪 Analyse texturale — GLCM & caractéristiques de Haralick")

    st.markdown("""
    La **GLCM** (*Gray‑Level Co‑occurrence Matrix*) capture la structure spatiale des niveaux de gris.
    À partir de cette matrice, on calcule des **caractéristiques de Haralick** (ex. *contrast, energy, entropy, homogeneity, correlation*),
    qui permettent d’analyser la **texture pulmonaire** (motifs, granularité, régularité).

    **Idées clés (issues du notebook)** :
    - Le **contrast** ressort comme **fortement discriminant** entre catégories.
    - **Entropy** et **homogeneity** sont souvent **corrélées négativement**, signe d’un compromis *désordre ↔ homogénéité*.
    - Des **différences d’intensité moyenne** entre classes imposent une **normalisation** avant l’entraînement.
    """)

    # Helpers para ordenar "harlick, harlick1, harlick2, ..."
    def natural_sort_key(p: Path):
        tokens = re.findall(r"\d+|\D+", p.stem.lower())
        return [int(t) if t.isdigit() else t for t in tokens]

    def list_harlick_figs_carousel() -> list[Path]:
        # Busca en reports/ y reports/figures/ todos los harlick*.png
        candidates = []
        for pat in [
            str(REPORTS_DIR / "harlick*.png"),
            str(REPORTS_FIG_DIR / "harlick*.png"),
        ]:
            candidates.extend([Path(p) for p in glob.glob(pat)])
        # Elimina duplicados manteniendo el primero encontrado
        unique = {str(p): p for p in candidates}
        files = list(unique.values())
        files.sort(key=natural_sort_key)
        return files

    harlick_figs = list_harlick_figs_carousel()

    if not harlick_figs:
        st.info("ℹ️ Aucune figure Haralick trouvée (recherché `reports/harlick*.png`).")
    else:
        # Estado del carrusel
        key = "harlick_carousel"
        if f"{key}_pos" not in st.session_state:
            st.session_state[f"{key}_pos"] = 1  # 1..N

        n = len(harlick_figs)

        # Controles: ←  imagen  →  (en 3 columnas)
        col_prev, col_img, col_next = st.columns([1, 6, 1])

        with col_prev:
            if st.button("◀", use_container_width=True, key=f"{key}_prev"):
                st.session_state[f"{key}_pos"] = (
                    n
                    if st.session_state[f"{key}_pos"] == 1
                    else st.session_state[f"{key}_pos"] - 1
                )

        with col_next:
            if st.button("▶", use_container_width=True, key=f"{key}_next"):
                st.session_state[f"{key}_pos"] = (
                    1
                    if st.session_state[f"{key}_pos"] == n
                    else st.session_state[f"{key}_pos"] + 1
                )

        # Slider de posición (sincronizado)
        pos = st.slider(
            "Position", 1, n, st.session_state[f"{key}_pos"], key=f"{key}_slider"
        )
        st.session_state[f"{key}_pos"] = pos
        idx = pos - 1

        # Muestra una sola figura con caption y ratio
        current = harlick_figs[idx]
        caption = f"{current.stem.replace('_',' ').capitalize()}  ({pos}/{n})"
        col_img.image(str(current), caption=caption, use_container_width=True)

        with st.expander("📝 Interprétation (résumé)"):
            st.markdown("""
            - **Contrast** : très informatif pour différencier les catégories – signatures texturales plus marquées.
            - **Entropy vs Homogeneity** : corrélation **fortement négative** (textures désorganisées vs régulières).
            - **Petites distances (0–3 px)** : forte corrélation locale malgré l’irrégularité – *désordre structuré*.
            - **Normalisation** recommandée pour corriger les biais d’intensité entre classes.
            - **Outliers** (poumons hors cadre, asymétries extrêmes) à **retirer** avant l’entraînement.
            """)

    # -------- 3) Biais, limites & prochaines étapes --------
    st.header("⚠️ Biais & limites observés")
    st.markdown("""
    - **Déséquilibre de classes** (ex. *Normal* > autres) → à corriger par **rééquilibrage** (under/over sampling).
    - **Variations d’acquisition** (distance/zoom, qualité des masques) → **normalisation** indispensable.
    - **Masques hors cadre / asymétries extrêmes** → **filtrage** via règles (IQR, contrôle de bords).
    """)

    st.subheader("🔮 Prochaines étapes côté EDA")
    st.markdown("""
    - Intégrer des visualisations **avant/après** normalisation.
    - Ajouter des métriques **par classe** (moyennes Haralick, histogrammes d’intensité).
    - Documenter un **protocole de nettoyage** reproductible (manifest + règles).
    """)
# =========================
# 2) Rééquilibrage
# =========================
elif page == "Rééquilibrage":
    st.title("⚖️ Rééquilibrage des classes")

    st.markdown("""
**But** : Expliquer et illustrer la création d’un dataset équilibré (binaire ou multiclasses)
via les symlinks de `train_folder.py`.

> Cette page **n’exécute pas** le rééquilibrage (sécurité VM).
> Elle **explique** le pipeline et montre la distribution si `data/train` est présent.
""")

    if not TRAIN_DIR.exists():
        st.warning(
            "⚠️ `data/train` introuvable. Impossible d’afficher la distribution réelle."
        )
        st.stop()

    st.subheader("📉 Distribution actuelle")
    counts = count_by_class(TRAIN_DIR)
    st.write(counts)

    st.subheader("🧭 Pipeline (résumé)")
    st.code("""
- Entrée : manifest parquet (issu de clean)
- Mapping de classes (binaire vs multiclasses)
- Création de symlinks par classe équilibrée
- Dossier final prêt pour l'entraînement
""")

# =========================
# 3) Modèle
# =========================
elif page == "Modèle":
    st.title("🧠 Modèle (VGG11) & Config Hydra")

    st.markdown("""
**But** : Présenter l’architecture et la configuration utilisées pour l’entraînement (Hydra + Lightning).
""")

    st.subheader("📄 Module (default.yaml)")
    module_cfg = load_yaml_safe(MODULE_YAML)
    st.json(module_cfg)

    st.subheader("📄 DataModule (datamodule.yaml)")
    dm_cfg = load_yaml_safe(DATAMODULE_YAML)
    st.json(dm_cfg)

    st.subheader("⚙️ Pipeline d'entraînement (résumé)")
    st.code("""
- callbacks, loggers
- Lightning Trainer
- DataModule + Module
- fit(), puis test()
""")

# =========================
# 4) Résultats
# =========================
elif page == "Résultats":
    st.title("📈 Résultats d'entraînement")

    st.markdown("""
**But** : Afficher les courbes et métriques si des logs existent (`logs/`).
""")

    metrics_files = scan_training_logs(LOGS_DIR)
    if not metrics_files:
        st.info(
            "ℹ️ Aucun log détecté dans `logs/`. Lorsque des métriques seront disponibles, elles seront affichées ici."
        )
    else:
        st.success(f"✅ Fichiers métriques trouvés : {len(metrics_files)}")
        st.write("Exemple de prévisualisation (premier CSV détecté) :")
        try:
            df = pd.read_csv(metrics_files[0])
            st.dataframe(df.head(20))
        except Exception as e:
            st.warning(f"Impossible de lire {metrics_files[0]} : {e}")

# =========================
# 5) Prédiction
# =========================
elif page == "Prédiction":
    st.title("🔮 Prédiction (checkpoint requis)")

    ckpts = find_checkpoints(MODELS_DIR)
    if not ckpts:
        st.error("""
❌ Aucun checkpoint `.ckpt` trouvé dans `models/`.

Pour activer la prédiction :
1) Entraîner le modèle (idéalement dans Colab)
2) Déposer `best.ckpt` dans `models/`
3) Recharger l'application
""")
        st.stop()

    st.success(f"✅ Checkpoint détecté : {ckpts[-1].name}")
    st.info(
        "Chargement & inference : à intégrer selon le format Lightning (RadioCovidModule)."
    )

# =========================
# 6) Conclusion
# =========================
elif page == "Conclusion":
    st.title("🔚 Conclusion & Perspectives")

    st.markdown("""
### 🧾 Conclusion
- Pipeline mis en place : **EDA → nettoyage → rééquilibrage → configuration du modèle**
- Architecture utilisée : **VGG11 (torchvision)** sous **Lightning + Hydra**
- L’app Streamlit est prête à intégrer **résultats entraînés** et **prédiction** dès que les artefacts sont fournis.

### 🔮 Perspectives
- Intégration des **courbes réelles** (logs/), **matrices de confusion** et **Grad‑CAM**
- Gestion **binaire vs multiclasses** via configuration
- Déploiement final (Streamlit Cloud / conteneur)
""")
