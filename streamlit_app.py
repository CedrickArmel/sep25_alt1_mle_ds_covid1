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

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RadioCovid",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
ROOT = Path(".")
INFERENCE_LOG_DIR = Path(os.environ.get("INFERENCE_LOG_DIR", "data/inference_logs"))
PREDICTIONS_FILE = INFERENCE_LOG_DIR / "predictions.jsonl"
REFERENCE_FILE = ROOT / "data" / "reference_distribution.json"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* ---- metric cards ---- */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #64748b; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0f172a; }

/* ---- divider ---- */
hr { border-color: #e2e8f0; margin: 1.5rem 0; }

/* ---- badge ---- */
.badge-ok {
    display:inline-block;
    background:#dcfce7; color:#166534;
    padding:4px 16px; border-radius:999px;
    font-weight:600; font-size:0.9rem;
}
.badge-warn {
    display:inline-block;
    background:#fef9c3; color:#854d0e;
    padding:4px 16px; border-radius:999px;
    font-weight:600; font-size:0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫁 RadioCovid")
    st.markdown("*Système d'aide à la décision en radiologie*")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🩻 Prédiction", "📊 Monitoring"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # API status chip
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API en ligne ✓")
        else:
            st.warning("API — erreur")
    except Exception:
        st.error("API hors ligne")

    st.markdown(
        "<div style='position:absolute;bottom:1.5rem;left:1.5rem;"
        "font-size:0.75rem;color:#64748b;'>v1.0 · RadioCovid 2026</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()
    rows = []
    with PREDICTIONS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_reference() -> dict:
    if not REFERENCE_FILE.exists():
        return {}
    try:
        return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _style_chart(ax, title=""):
    ax.set_title(title, fontsize=11, fontweight="600", pad=8, color="#0f172a")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.set_facecolor("#f8fafc")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#e2e8f0")


# ===========================================================================
# PAGE 1 — ACCUEIL
# ===========================================================================
if page == "🏠 Accueil":
    # Hero
    st.markdown(
        """
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
            padding:3rem 2.5rem 2.5rem;border-radius:16px;margin-bottom:2rem;">
  <div style="font-size:0.85rem;color:#7dd3fc;font-weight:600;
              letter-spacing:.1em;text-transform:uppercase;margin-bottom:.6rem;">
    Projet MLOps — Promotion 2025/2026
  </div>
  <h1 style="color:white;font-size:2.4rem;font-weight:800;margin:0 0 .6rem;">
    🫁 RadioCovid
  </h1>
  <p style="color:#cbd5e1;font-size:1.1rem;margin:0 0 1.5rem;max-width:600px;">
    Système de classification automatique de radiographies thoraciques
    avec pipeline MLOps complet — de la donnée brute à la détection de dérive en production.
  </p>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;">
    <span style="background:#1e40af;color:#bfdbfe;padding:4px 14px;
                 border-radius:999px;font-size:.82rem;font-weight:600;">VGG-11 · PyTorch</span>
    <span style="background:#065f46;color:#a7f3d0;padding:4px 14px;
                 border-radius:999px;font-size:.82rem;font-weight:600;">FastAPI · Docker</span>
    <span style="background:#7c2d12;color:#fed7aa;padding:4px 14px;
                 border-radius:999px;font-size:.82rem;font-weight:600;">Airflow · W&B · DVC</span>
    <span style="background:#4a1d96;color:#ddd6fe;padding:4px 14px;
                 border-radius:999px;font-size:.82rem;font-weight:600;">Evidently · Prometheus</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images d'entraînement", "21 165", "Dataset Kaggle COVID-19")
    c2.metric("Temps d'inférence", "< 2 s", "Par radiographie")
    c3.metric("Classes", "2", "NORMAL · ABNORMAL")
    c4.metric("Pipeline", "100% automatisé", "Ingest → Train → Deploy")

    st.markdown("---")

    # How it works
    st.markdown("### Comment ça fonctionne")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
<div style="background:#f0f9ff;border-left:4px solid #0284c7;
            padding:1.2rem;border-radius:0 8px 8px 0;height:160px;">
  <div style="font-size:1.5rem;margin-bottom:.5rem;">📤</div>
  <div style="font-weight:700;color:#0f172a;margin-bottom:.4rem;">1. Upload</div>
  <div style="color:#475569;font-size:.9rem;">
    Le professionnel de santé dépose une radiographie thoracique via l'interface.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div style="background:#f0fdf4;border-left:4px solid #16a34a;
            padding:1.2rem;border-radius:0 8px 8px 0;height:160px;">
  <div style="font-size:1.5rem;margin-bottom:.5rem;">🤖</div>
  <div style="font-weight:700;color:#0f172a;margin-bottom:.4rem;">2. Analyse IA</div>
  <div style="color:#475569;font-size:.9rem;">
    Le modèle VGG-11 classifie l'image et retourne un score de confiance en moins de 2 secondes.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
<div style="background:#fdf4ff;border-left:4px solid #9333ea;
            padding:1.2rem;border-radius:0 8px 8px 0;height:160px;">
  <div style="font-size:1.5rem;margin-bottom:.5rem;">📊</div>
  <div style="font-weight:700;color:#0f172a;margin-bottom:.4rem;">3. Surveillance</div>
  <div style="color:#475569;font-size:.9rem;">
    Chaque prédiction est loggée. Le système détecte automatiquement toute dérive du modèle.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Architecture
    st.markdown("### Architecture du système")
    st.markdown(
        """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;">
<pre style="color:#0f172a;font-size:.82rem;line-height:1.7;margin:0;font-family:'Courier New',monospace;">
  ┌─────────────────────────────────────────────────────────────────┐
  │                        PIPELINE MLOPS                          │
  │                                                                 │
  │  [Données brutes]                                               │
  │       │ DVC + Google Drive                                      │
  │       ▼                                                         │
  │  [ETL Container] ──► [Train Container] ──► [W&B Model Registry]│
  │                            │ Airflow @weekly                    │
  │                            ▼                                    │
  │  [Inference API] ◄── [NGINX Gateway] ◄── [Streamlit / Client]  │
  │       │ FastAPI + Auth           rate limit · TLS-ready         │
  │       │                                                         │
  │       ├──► [Prometheus] ──► [Grafana]   (métriques temps réel) │
  │       └──► [JSONL logs] ──► [Evidently] (détection de dérive)  │
  │                                │ Airflow @daily                 │
  │                                └──► [Retraining si drift]       │
  └─────────────────────────────────────────────────────────────────┘
</pre>
</div>""",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE 2 — PRÉDICTION
# ===========================================================================
elif page == "🩻 Prédiction":
    st.markdown(
        """
<div style="margin-bottom:1.5rem;">
  <h1 style="margin:0 0 .3rem;color:#0f172a;">🩻 Analyse de radiographie</h1>
  <p style="color:#64748b;margin:0;">
    Déposez une radiographie thoracique pour obtenir une classification automatique.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
.diag-card { padding:2rem;border-radius:16px;text-align:center;margin-top:1rem; }
.diag-normal  { background:linear-gradient(135deg,#064e3b,#065f46);border:2px solid #10b981; }
.diag-abnormal{ background:linear-gradient(135deg,#7f1d1d,#991b1b);border:2px solid #ef4444; }
.diag-label   { font-size:2.2rem;font-weight:800;color:white;letter-spacing:.05em; }
.diag-prob    { font-size:1rem;color:rgba(255,255,255,.85);margin-top:.4rem; }
.disclaimer   { background:#f8fafc;border-left:4px solid #f59e0b;padding:.75rem 1rem;
                border-radius:0 8px 8px 0;color:#78716c;font-size:.82rem;margin-top:1.5rem; }
</style>""",
        unsafe_allow_html=True,
    )

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("#### 📤 Image à analyser")
        uploaded = st.file_uploader(
            "Formats acceptés : PNG, JPG, JPEG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded:
            uploaded.seek(0)
            img_bytes = uploaded.read()
            if len(img_bytes) == 0:
                st.error("Fichier vide — essayez de re-uploader l'image.")
            else:
                st.image(img_bytes, caption=uploaded.name, use_container_width=True)

    with col_res:
        st.markdown("#### 🔬 Résultat")
        if not uploaded:
            st.markdown(
                """
<div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;
            padding:3rem;text-align:center;color:#94a3b8;">
  <div style="font-size:2rem;margin-bottom:.5rem;">📂</div>
  Déposez une image pour lancer l'analyse
</div>""",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyse en cours…"):
                try:
                    headers = {"X-API-Key": API_KEY} if API_KEY else {}
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, img_bytes, uploaded.type)},
                        headers=headers,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        label = data["label"]
                        prob = data["probability"]
                        pct = round(prob * 100, 1)
                        is_normal = label == "NORMAL"
                        icon = "✅" if is_normal else "⚠️"
                        card = "diag-normal" if is_normal else "diag-abnormal"
                        st.markdown(
                            f"""
<div class="diag-card {card}">
  <div class="diag-label">{icon} {label}</div>
  <div class="diag-prob">Confiance : <strong>{pct}%</strong></div>
</div>""",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(prob, text=f"Score de confiance : {pct}%")

                        with st.expander("📋 Détails techniques"):
                            try:
                                info = requests.get(f"{API_URL}/info", timeout=5).json()
                            except Exception:
                                info = {}
                            st.json(
                                {
                                    "label": label,
                                    "confidence": prob,
                                    "model_run_id": info.get("run_id", "—"),
                                    "registry_alias": info.get(
                                        "registry_alias", "production"
                                    ),
                                    "architecture": info.get("arch", "VGG-11"),
                                }
                            )

                    elif resp.status_code == 403:
                        st.error(
                            "🔒 Clé API invalide — vérifiez la variable `API_KEY`."
                        )
                    else:
                        st.error(f"❌ Erreur API ({resp.status_code})")

                except requests.exceptions.ConnectionError:
                    st.markdown(
                        """
<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:12px;
            padding:1.5rem;text-align:center;">
  <div style="font-size:1.5rem;margin-bottom:.5rem;">🔌</div>
  <div style="font-weight:600;color:#991b1b;">API hors ligne</div>
  <div style="color:#b91c1c;font-size:.85rem;margin-top:.4rem;">
    Démarrez l'API : <code>uvicorn radiocovid.inference.api:app --port 8000</code>
  </div>
</div>""",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")

        if uploaded:
            st.markdown(
                """
<div class="disclaimer">
  ⚠️ <strong>Avertissement médical</strong> — Ce résultat est produit par un modèle d'IA
  à des fins de recherche. Il ne constitue pas un diagnostic médical.
  Consultez un professionnel de santé qualifié.
</div>""",
                unsafe_allow_html=True,
            )

# ===========================================================================
# PAGE 3 — MONITORING
# ===========================================================================
elif page == "📊 Monitoring":
    st.markdown(
        """
<div style="margin-bottom:1.5rem;">
  <h1 style="margin:0 0 .3rem;color:#0f172a;">📊 Monitoring du modèle</h1>
  <p style="color:#64748b;margin:0;">
    Surveillance continue des prédictions et détection de dérive en production.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

    df = load_predictions()
    reference = load_reference()

    if df.empty:
        st.warning(
            "Aucun log de prédictions trouvé. "
            "Activez `ENABLE_INFERENCE_LOGGING=1` et effectuez des prédictions."
        )
        st.stop()

    # Window
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df7 = df[df["timestamp"] >= cutoff].copy()
    df7["date"] = df7["timestamp"].dt.date

    # ---- Status banner ----
    n_total = len(df7)
    avg_conf = df7["confidence"].mean() if n_total else 0
    n_normal = (df7["label"] == "NORMAL").sum()
    n_abnormal = n_total - n_normal

    # Simple drift flag: if avg confidence drops > 5% below reference
    ref_conf = reference.get("features", {}).get("confidence", {}).get("mean", avg_conf)
    drift_flag = avg_conf < ref_conf - 0.05

    if drift_flag:
        st.markdown(
            '<div style="background:#fef9c3;border:1px solid #fde047;border-radius:10px;'
            'padding:.8rem 1.2rem;margin-bottom:1rem;">'
            "⚠️ <strong>Alerte drift potentiel</strong> — "
            "La confiance moyenne a baissé de plus de 5 points par rapport au référentiel. "
            "Vérifiez la qualité des images entrantes.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#dcfce7;border:1px solid #86efac;border-radius:10px;'
            'padding:.8rem 1.2rem;margin-bottom:1rem;">'
            "✅ <strong>Aucun drift détecté</strong> — "
            "Le modèle se comporte de manière stable sur les 7 derniers jours.</div>",
            unsafe_allow_html=True,
        )

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prédictions (7j)", n_total)
    k2.metric(
        "Confiance moyenne",
        f"{avg_conf:.1%}",
        delta=f"{(avg_conf - ref_conf):+.1%} vs référentiel",
        delta_color="normal",
    )
    k3.metric("NORMAL", f"{n_normal}", f"{n_normal/n_total:.0%}" if n_total else "")
    k4.metric(
        "ABNORMAL", f"{n_abnormal}", f"{n_abnormal/n_total:.0%}" if n_total else ""
    )

    st.markdown("---")

    # ---- Confidence timeline ----
    st.markdown("#### Confidence moyenne par jour")
    daily = df7.groupby("date")["confidence"].agg(["mean", "count"]).reset_index()
    daily.columns = ["date", "mean", "n"]

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(range(len(daily)), daily["mean"], alpha=0.12, color="#0284c7")
    ax.plot(
        range(len(daily)),
        daily["mean"],
        marker="o",
        lw=2.5,
        color="#0284c7",
        markersize=6,
        label="Confiance moyenne",
    )
    if reference:
        ref_m = ref_conf
        ax.axhline(ref_m, color="#94a3b8", lw=1.5, ls="--", label="Référentiel")
        ax.fill_between(
            range(len(daily)), ref_m - 0.05, ref_m + 0.05, alpha=0.06, color="#94a3b8"
        )
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels(
        [str(d) for d in daily["date"]], rotation=25, ha="right", fontsize=8
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Confiance", fontsize=9)
    ax.legend(fontsize=8)
    for i, row in daily.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (i, row["mean"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="#64748b",
        )
    _style_chart(ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ---- Feature histograms ----
    st.markdown("#### Distribution des features image — courant vs référentiel")

    FEATURES = {
        "img_mean": "Luminosité moyenne",
        "img_std": "Contraste",
        "img_entropy": "Entropie",
        "confidence": "Confiance modèle",
    }
    available = [f for f in FEATURES if f in df7.columns]
    cols = st.columns(len(available))

    for i, feat in enumerate(available):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(4, 2.8))
            vals = df7[feat].dropna()
            ax.hist(
                vals,
                bins=22,
                alpha=0.75,
                color="#0284c7",
                density=True,
                label=f"Courant (n={len(vals)})",
            )
            if reference and feat in reference.get("features", {}):
                r = reference["features"][feat]
                mu, sigma = r["mean"], r["std"]
                x = np.linspace(r["min"], r["max"], 200)
                y = (
                    1
                    / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                )
                ax.plot(x, y, color="#94a3b8", lw=2, label="Référentiel")
            ax.legend(fontsize=7)
            ax.set_yticks([])
            _style_chart(ax, FEATURES[feat])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")

    # ---- Label distribution + Recent predictions ----
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### Répartition des labels")
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        sizes = [n_normal, n_abnormal]
        colors = ["#10b981", "#ef4444"]
        wedges, _, autotexts = ax.pie(
            sizes,
            labels=["NORMAL", "ABNORMAL"],
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("white")
            at.set_fontweight("bold")
        ax.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        st.markdown("#### 10 dernières prédictions")
        recent = (
            df7.sort_values("timestamp", ascending=False)
            .head(10)[["timestamp", "label", "confidence"]]
            .copy()
        )
        recent["timestamp"] = recent["timestamp"].dt.strftime("%d/%m %H:%M")
        recent["confidence"] = recent["confidence"].map(lambda x: f"{x:.1%}")
        recent = recent.rename(
            columns={
                "timestamp": "Horodatage",
                "label": "Résultat",
                "confidence": "Confiance",
            }
        )
        st.dataframe(recent, hide_index=True, use_container_width=True)

    st.caption(
        "Données lues depuis `data/inference_logs/predictions.jsonl`. "
        "Activez `ENABLE_INFERENCE_LOGGING=1` en production pour alimenter ce tableau de bord en temps réel."
    )
