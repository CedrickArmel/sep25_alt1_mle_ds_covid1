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

"""Streamlit UI for RadioCOVID — calls the inference HTTP API for predictions."""

import glob
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml  # type: ignore[import-untyped]

from .client import InferenceClient
from .config import get_config

st.set_page_config(
    page_title="RadioCovid – Projet Poumons", page_icon="🫁", layout="wide"
)

st.sidebar.title("🫁 Menu")
PAGES = ["EDA", "Rééquilibrage", "Modèle", "Résultats", "Prédiction", "Conclusion"]
page = st.sidebar.radio("Aller à :", PAGES)

_cfg = get_config()

ROOT = Path(".")
DATA_ROOT = ROOT / "data"
TRAIN_DIR = DATA_ROOT / "train"
LOGS_DIR = ROOT / "logs"
MODULE_YAML = ROOT / "radiocovid-core/src/radiocovid/core/configs/module/default.yaml"
DATAMODULE_YAML = ROOT / "radiocovid-core/src/radiocovid/core/configs/datamodule.yaml"
REPORTS_DIR = _cfg.reports_dir
REPORTS_FIG_DIR = REPORTS_DIR / "figures"


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


@st.cache_data
def scan_training_logs(logs_dir: Path):
    if not logs_dir.exists():
        return []
    return list(logs_dir.rglob("*.csv"))


def render_resume_card():
    if not st.session_state.get("_resume_css_added", False):
        st.markdown(
            """
            <style>
                .rc-card{background:#0f172a;color:#e2e8f0;padding:1.25rem 1.5rem;
                    border-radius:12px;border:1px solid #334155;
                    box-shadow:0 0 0 1px rgba(255,255,255,0.04) inset}
                .rc-card h3,.rc-card h4{margin-top:0.2rem;color:#f1f5f9}
                .rc-card p{margin:0.2rem 0 0.8rem 0}
                .rc-card ul{margin:0.2rem 0 0.8rem 1.1rem}
                .rc-tag{display:inline-block;background:#1f2937;color:#cbd5e1;
                    padding:2px 10px;border-radius:999px;font-size:0.85rem;
                    margin-right:6px;margin-bottom:4px}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_resume_css_added"] = True

    st.markdown(
        """
<div class="rc-card">
<h3>📌 Résumé des volumes du dataset</h3>
<p>L'ensemble de données <em>COVID‑19 Radiography</em> présente des volumes très variables selon les
catégories, ce qui introduit un <strong>déséquilibre de classes important</strong> :</p>
<ul>
  <li><strong>COVID</strong> — 3 616 images (299×299)</li>
  <li><strong>Lung Opacity</strong> — 6 012 images (299×299)</li>
  <li><strong>Viral Pneumonia</strong> — 1 345 images (299×299)</li>
  <li><strong>Normal</strong> — 10 192 images (299×299)</li>
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
# 1) EDA
# =========================
if page == "EDA":
    st.title("🔍 EDA — Exploration des Données")

    st.markdown("""
    **Objectif** : présenter l'exploration du dataset (source Kaggle) et l'analyse texturale
    (GLCM/Haralick) à partir des figures **déjà générées** par le notebook d'EDA.

    **Dataset** : COVID‑19 Radiography Database (Kaggle)
    """)

    def first_existing(*paths: Path) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    st.header("📦 Volumétrie & répartition des classes")
    dist_fig = first_existing(
        REPORTS_DIR / "images.png", REPORTS_FIG_DIR / "images.png"
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

    render_resume_card()

    st.header("🧪 Analyse texturale — GLCM & caractéristiques de Haralick")
    st.markdown("""
    La **GLCM** capture la structure spatiale des niveaux de gris.
    Caractéristiques de Haralick (*contrast, energy, entropy, homogeneity, correlation*) analysent la
    **texture pulmonaire**.
    """)

    def natural_sort_key(p: Path):
        tokens = re.findall(r"\d+|\D+", p.stem.lower())
        return [int(t) if t.isdigit() else t for t in tokens]

    harlick_figs = sorted(
        {
            str(p): p
            for pat in [
                str(REPORTS_DIR / "harlick*.png"),
                str(REPORTS_FIG_DIR / "harlick*.png"),
            ]
            for p in [Path(x) for x in glob.glob(pat)]
        }.values(),
        key=natural_sort_key,
    )

    if not harlick_figs:
        st.info("ℹ️ Aucune figure Haralick trouvée.")
    else:
        key = "harlick_carousel"
        if f"{key}_pos" not in st.session_state:
            st.session_state[f"{key}_pos"] = 1
        n = len(harlick_figs)
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
        pos = st.slider(
            "Position", 1, n, st.session_state[f"{key}_pos"], key=f"{key}_slider"
        )
        st.session_state[f"{key}_pos"] = pos
        current = harlick_figs[pos - 1]
        col_img.image(
            str(current),
            caption=f"{current.stem.replace('_',' ').capitalize()} ({pos}/{n})",
            use_container_width=True,
        )

    st.header("⚠️ Biais & limites observés")
    st.markdown("""
    - **Déséquilibre de classes** → à corriger par rééquilibrage.
    - **Variations d'acquisition** → normalisation indispensable.
    - **Masques hors cadre / asymétries extrêmes** → filtrage IQR.
    """)

# =========================
# 2) Rééquilibrage
# =========================
elif page == "Rééquilibrage":
    st.title("⚖️ Rééquilibrage des classes")
    st.markdown("""
**But** : Expliquer et illustrer la création d'un dataset équilibré via les symlinks de `train_folder.py`.
""")
    if not TRAIN_DIR.exists():
        st.warning(
            "⚠️ `data/train` introuvable. Impossible d'afficher la distribution réelle."
        )
        st.stop()
    st.subheader("📉 Distribution actuelle")
    st.write(count_by_class(TRAIN_DIR))
    st.subheader("🧭 Pipeline (résumé)")
    st.code(
        "Entrée manifest parquet → mapping classes → symlinks → dossier équilibré prêt"
    )

# =========================
# 3) Modèle
# =========================
elif page == "Modèle":
    st.title("🧠 Modèle & Config Hydra")
    st.subheader("📄 Module (default.yaml)")
    st.json(load_yaml_safe(MODULE_YAML))
    st.subheader("📄 DataModule (datamodule.yaml)")
    st.json(load_yaml_safe(DATAMODULE_YAML))
    st.subheader("⚙️ Pipeline d'entraînement")
    st.code("callbacks → loggers → Trainer → DataModule + Module → fit() → test()")

# =========================
# 4) Résultats
# =========================
elif page == "Résultats":
    st.title("📈 Résultats d'entraînement")
    metrics_files = scan_training_logs(LOGS_DIR)
    if not metrics_files:
        st.info("ℹ️ Aucun log détecté dans `logs/`.")
    else:
        st.success(f"✅ {len(metrics_files)} fichier(s) métriques trouvé(s)")
        try:
            df = pd.read_csv(metrics_files[0])
            st.dataframe(df.head(20))
        except Exception as e:
            st.warning(f"Impossible de lire {metrics_files[0]} : {e}")

# =========================
# 5) Prédiction
# =========================
elif page == "Prédiction":
    st.title("🔮 Prédiction via API d'inférence")

    client = InferenceClient(_cfg.inference_url)

    try:
        health = client.health()
        st.success(f"✅ Service en ligne — checkpoint : `{health.get('ckpt', 'N/A')}`")
    except ConnectionError:
        st.error(
            f"❌ Service d'inférence injoignable à `{_cfg.inference_url}`.\n\n"
            "Démarrez le service avec `radiocovid-serve` et vérifiez `RADIOCOVID_INFERENCE_URL`."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Chargez une radiographie (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        col_img, col_result = st.columns([1, 1])
        col_img.image(uploaded, caption="Image chargée", use_container_width=True)
        with st.spinner("Analyse en cours…"):
            try:
                result = client.predict(uploaded.read(), filename=uploaded.name)
                label = result["label"]
                confidence = result["confidence"]
                col_result.metric("Diagnostic", label.upper())
                col_result.metric("Confiance", f"{confidence:.1%}")
            except (ConnectionError, RuntimeError) as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")

# =========================
# 6) Conclusion
# =========================
elif page == "Conclusion":
    st.title("🔚 Conclusion & Perspectives")
    st.markdown("""
### 🧾 Conclusion
- Pipeline mis en place : **EDA → nettoyage → rééquilibrage → entraînement**
- Architecture : **ResNet50** sous **Lightning + Hydra**
- API d'inférence déployable via `radiocovid-serve`

### 🔮 Perspectives
- Courbes réelles (logs/), matrices de confusion, Grad‑CAM
- Gestion binaire vs multiclasses via configuration
- Déploiement Streamlit Cloud / conteneur Docker
""")
