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

import hashlib
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
DRIFT_RESULT_FILE = ROOT / "reports" / "drift_result_latest.json"

# ---------------------------------------------------------------------------
# Auth: credentials stored as sha256 hashes
# ---------------------------------------------------------------------------
_USERS = {
    "admin": {
        "hash": hashlib.sha256(b"admin2026").hexdigest(),
        "role": "admin",
        "display": "Administrateur",
    },
    "user": {
        "hash": hashlib.sha256(b"user2026").hexdigest(),
        "role": "user",
        "display": "Utilisateur",
    },
}


def _check_credentials(username: str, password: str) -> dict | None:
    entry = _USERS.get(username.strip().lower())
    if entry and hashlib.sha256(password.encode()).hexdigest() == entry["hash"]:
        return entry
    return None


def _init_session():
    for key, default in [
        ("logged_in", False),
        ("role", None),
        ("username", None),
        ("display", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarNav"] { display: none; }

[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #64748b; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0f172a; }

.rc-card-normal {
    padding: 2rem; border-radius: 12px; text-align: center;
    background: #064e3b; border: 1px solid #10b981;
}
.rc-card-abnormal {
    padding: 2rem; border-radius: 12px; text-align: center;
    background: #7f1d1d; border: 1px solid #ef4444;
}
.rc-label { font-size: 2rem; font-weight: 700; color: white; }
.rc-conf  { font-size: 0.95rem; color: rgba(255,255,255,0.8); margin-top: 0.4rem; }

.rc-disclaimer {
    background: #f8fafc; border-left: 3px solid #cbd5e1;
    padding: 0.7rem 1rem; border-radius: 0 6px 6px 0;
    color: #64748b; font-size: 0.8rem; margin-top: 1.2rem;
}

.rc-status-ok   { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                  padding:.7rem 1rem; color:#166534; margin-bottom:1rem; font-size:.9rem; }
.rc-status-warn { background:#fffbeb; border:1px solid #fde68a; border-radius:8px;
                  padding:.7rem 1rem; color:#92400e; margin-bottom:1rem; font-size:.9rem; }

.rc-login-box {
    max-width: 380px; margin: 6rem auto 0; padding: 2.5rem 2rem;
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,.06);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Init session
# ---------------------------------------------------------------------------
_init_session()

# ===========================================================================
# LOGIN PAGE
# ===========================================================================
if not st.session_state.logged_in:
    # Hide sidebar on login screen
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none}</style>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div style="text-align:center;margin-top:4rem;">
  <div style="font-size:1.8rem;font-weight:700;color:#0f172a;">RadioCovid</div>
  <div style="color:#64748b;font-size:.95rem;margin-top:.3rem;">
    Système de classification de radiographies thoraciques
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("Identifiant", placeholder="Identifiant")
        password = st.text_input(
            "Mot de passe", type="password", placeholder="Mot de passe"
        )

        if st.button("Se connecter", use_container_width=True, type="primary"):
            entry = _check_credentials(username, password)
            if entry:
                st.session_state.logged_in = True
                st.session_state.role = entry["role"]
                st.session_state.username = username.strip().lower()
                st.session_state.display = entry["display"]
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")

    st.stop()

# ===========================================================================
# AUTHENTICATED: Sidebar
# ===========================================================================
role = st.session_state.role

with st.sidebar:
    st.markdown("### RadioCovid")
    st.markdown(
        "<span style='font-size:.82rem;color:#94a3b8;'>"
        "Aide à la décision en radiologie</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if role == "admin":
        nav_options = ["Accueil", "Prédiction", "Monitoring"]
    else:
        nav_options = ["Prédiction"]

    page = st.radio(
        "Navigation",
        nav_options,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        api_ok = r.status_code == 200
    except Exception:
        api_ok = False

    status_color = "#4ade80" if api_ok else "#f87171"
    status_text = "API en ligne" if api_ok else "API hors ligne"
    st.markdown(
        f"<div style='font-size:.8rem;color:{status_color};'>"
        f"&#9679; {status_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:.8rem;color:#94a3b8;'>"
        f"Connecté : {st.session_state.display}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Déconnexion", use_container_width=True):
        for key in ["logged_in", "role", "username", "display"]:
            st.session_state[key] = None if key != "logged_in" else False
        st.rerun()

    st.markdown(
        "<div style='position:absolute;bottom:1.5rem;left:1.5rem;"
        "font-size:.72rem;color:#475569;'>v1.0 &nbsp;·&nbsp; 2026</div>",
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


@st.cache_data(ttl=300)
def load_reference() -> dict:
    if not REFERENCE_FILE.exists():
        return {}
    try:
        return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_drift_result() -> dict:
    if not DRIFT_RESULT_FILE.exists():
        return {}
    try:
        return json.loads(DRIFT_RESULT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _style_chart(ax, title=""):
    ax.set_title(title, fontsize=10, fontweight="600", pad=8, color="#1e293b")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.set_facecolor("#f8fafc")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#e2e8f0")


# ===========================================================================
# ACCUEIL  (admin only)
# ===========================================================================
if page == "Accueil":
    st.markdown(
        """
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
            padding:2.5rem 2rem 2rem;border-radius:12px;margin-bottom:2rem;">
  <div style="font-size:.78rem;color:#7dd3fc;letter-spacing:.08em;
              text-transform:uppercase;margin-bottom:.5rem;">
    Projet MLOps · Promotion 2025/2026
  </div>
  <h1 style="color:white;font-size:2.2rem;font-weight:700;margin:0 0 .5rem;">RadioCovid</h1>
  <p style="color:#94a3b8;font-size:1rem;margin:0 0 1.4rem;max-width:560px;line-height:1.6;">
    Classification automatique de radiographies thoraciques avec pipeline MLOps complet,
    de la donnée brute jusqu'à la détection de dérive en production.
  </p>
  <div style="display:flex;gap:.7rem;flex-wrap:wrap;">
    <span style="background:rgba(255,255,255,.08);color:#cbd5e1;padding:3px 12px;
                 border-radius:4px;font-size:.78rem;">VGG-11 · PyTorch</span>
    <span style="background:rgba(255,255,255,.08);color:#cbd5e1;padding:3px 12px;
                 border-radius:4px;font-size:.78rem;">FastAPI · Docker · NGINX</span>
    <span style="background:rgba(255,255,255,.08);color:#cbd5e1;padding:3px 12px;
                 border-radius:4px;font-size:.78rem;">Airflow · W&B · DVC</span>
    <span style="background:rgba(255,255,255,.08);color:#cbd5e1;padding:3px 12px;
                 border-radius:4px;font-size:.78rem;">Evidently · Prometheus</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Images d'entraînement", "21 165", "COVID-19 Radiography Database")
    c2.metric("Temps d'inférence", "< 2 s", "Par radiographie")
    c3.metric("Classes", "2", "Normal / Anormal")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Fonctionnement**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1.2rem;">
  <div style="font-weight:600;color:#0f172a;margin-bottom:.4rem;">1. Acquisition</div>
  <div style="color:#475569;font-size:.88rem;line-height:1.5;">
    Le professionnel de santé dépose une radiographie thoracique via l'interface.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1.2rem;">
  <div style="font-weight:600;color:#0f172a;margin-bottom:.4rem;">2. Classification</div>
  <div style="color:#475569;font-size:.88rem;line-height:1.5;">
    Le modèle VGG-11 analyse l'image et retourne une classification
    avec un score de confiance en moins de 2 secondes.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1.2rem;">
  <div style="font-weight:600;color:#0f172a;margin-bottom:.4rem;">3. Surveillance</div>
  <div style="color:#475569;font-size:.88rem;line-height:1.5;">
    Chaque prédiction est loggée. Le système détecte automatiquement
    toute dérive du comportement du modèle.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Architecture**")
    st.markdown(
        """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:1.2rem 1.5rem;font-size:.82rem;color:#334155;line-height:1.8;">
  <table style="border-collapse:collapse;width:100%;">
    <tr>
      <td style="padding:.3rem .8rem;font-weight:600;color:#0284c7;width:180px;">Données</td>
      <td style="padding:.3rem .8rem;">DVC + Google Drive · versionage du dataset</td>
    </tr>
    <tr style="background:#f1f5f9;">
      <td style="padding:.3rem .8rem;font-weight:600;color:#0284c7;">Pipeline ML</td>
      <td style="padding:.3rem .8rem;">Airflow · ETL · Entraînement · Promotion W&B Model Registry</td>
    </tr>
    <tr>
      <td style="padding:.3rem .8rem;font-weight:600;color:#0284c7;">Inférence</td>
      <td style="padding:.3rem .8rem;">FastAPI · authentification · NGINX gateway · rate limiting</td>
    </tr>
    <tr style="background:#f1f5f9;">
      <td style="padding:.3rem .8rem;font-weight:600;color:#0284c7;">Observabilité</td>
      <td style="padding:.3rem .8rem;">Prometheus · Grafana · logs JSONL · Evidently drift detection</td>
    </tr>
    <tr>
      <td style="padding:.3rem .8rem;font-weight:600;color:#0284c7;">Intégration</td>
      <td style="padding:.3rem .8rem;">Docker · GitHub Actions CI/CD · Docker Hub</td>
    </tr>
  </table>
</div>""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PREDICTION  (all roles)
# ===========================================================================
elif page == "Prédiction":
    if role == "user":
        st.markdown(
            "<h2 style='margin-bottom:.3rem;color:#0f172a;'>Analyse de radiographie</h2>"
            "<p style='color:#64748b;margin:0 0 1.5rem;font-size:.95rem;'>"
            "Déposez une radiographie thoracique pour obtenir une classification automatique.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h2 style='margin-bottom:.3rem;color:#0f172a;'>Analyse de radiographie</h2>"
            "<p style='color:#64748b;margin:0 0 1.5rem;font-size:.95rem;'>"
            "Déposez une radiographie thoracique pour obtenir une classification automatique.</p>",
            unsafe_allow_html=True,
        )

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown(
            "<p style='font-weight:600;color:#374151;margin-bottom:.4rem;'>"
            "Image à analyser</p>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "PNG, JPG ou JPEG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded:
            uploaded.seek(0)
            img_bytes = uploaded.read()
            if len(img_bytes) == 0:
                st.error("Fichier vide. Essayez de re-uploader l'image.")
            else:
                st.image(img_bytes, caption=uploaded.name, use_container_width=True)

    with col_res:
        st.markdown(
            "<p style='font-weight:600;color:#374151;margin-bottom:.4rem;'>"
            "Résultat</p>",
            unsafe_allow_html=True,
        )
        if not uploaded:
            st.markdown(
                """
<div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:10px;
            padding:3rem 1rem;text-align:center;color:#94a3b8;font-size:.9rem;">
  Déposez une image pour lancer l'analyse
</div>""",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyse en cours"):
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
                        card_cls = (
                            "rc-card-normal"
                            if label == "NORMAL"
                            else "rc-card-abnormal"
                        )
                        st.markdown(
                            f'<div class="{card_cls}">'
                            f'<div class="rc-label">{label}</div>'
                            f'<div class="rc-conf">Confiance : {pct}%</div>'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(prob, text=f"Score de confiance : {pct}%")

                        if role == "admin":
                            with st.expander("Détails"):
                                try:
                                    info = requests.get(
                                        f"{API_URL}/info", timeout=5
                                    ).json()
                                except Exception:
                                    info = {}
                                st.json(
                                    {
                                        "label": label,
                                        "confidence": prob,
                                        "model_run_id": info.get("run_id", ""),
                                        "registry_alias": info.get(
                                            "registry_alias", "production"
                                        ),
                                        "architecture": info.get("arch", "VGG-11"),
                                    }
                                )

                    elif resp.status_code == 403:
                        st.error("Clé API invalide.")
                    else:
                        st.error(f"Erreur API ({resp.status_code})")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Service d'inférence non joignable. "
                        "Vérifiez que l'API est démarrée sur le port 8000."
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")

        if uploaded:
            st.markdown(
                '<div class="rc-disclaimer">'
                "Ce résultat est produit par un modèle d'apprentissage automatique "
                "à des fins de recherche. Il ne constitue pas un diagnostic médical. "
                "Consultez un professionnel de santé qualifié."
                "</div>",
                unsafe_allow_html=True,
            )


# ===========================================================================
# MONITORING  (admin only)
# ===========================================================================
elif page == "Monitoring":
    AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")
    AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
    AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS", "admin")
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
    WANDB_API_KEY_VAL = os.environ.get("WANDB_API_KEY", "")

    st.markdown(
        "<h2 style='margin-bottom:.3rem;color:#0f172a;'>Monitoring</h2>"
        "<p style='color:#64748b;margin:0 0 1.5rem;font-size:.95rem;'>"
        "État du modèle, des pipelines et des prédictions en production.</p>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 1. Modèle en production (W&B)
    # -----------------------------------------------------------------------
    st.markdown("**Modèle en production**")

    @st.cache_data(ttl=120)
    def load_wandb_production():
        try:
            import wandb

            api = wandb.Api()
            registry = os.environ.get("WANDB_REGISTRY", "Radiocovid-classifier")
            model_name = os.environ.get("WANDB_REGISTRY_MODEL", "radiocovid-classifier")
            alias = os.environ.get("WANDB_REGISTRY_ALIAS", "production")
            collection = api.artifact_collection(
                type_name="model",
                name=f"wandb-registry-{registry}/{model_name}",
            )
            for art in collection.versions():
                if alias in (art.aliases or []):
                    run = art.logged_by()
                    summary = run.summary if run else {}
                    return {
                        "version": art.version,
                        "created_at": art.created_at,
                        "run_id": run.id if run else "—",
                        "f1": summary.get("best_val_score", summary.get("val_f1", "—")),
                        "accuracy": summary.get("val_accuracy", "—"),
                        "loss": summary.get("val_loss", "—"),
                    }
        except Exception:
            pass
        # Fallback: fetch metrics directly from the production run in W&B
        try:
            import wandb

            entity = os.environ.get("WANDB_ENTITY", "")
            project = os.environ.get("WANDB_PROJECT", "")
            run_id = os.environ.get("WANDB_RUN_ID", "")
            if entity and project and run_id:
                api2 = wandb.Api()
                run = api2.run(f"{entity}/{project}/{run_id}")
                summary = run.summary
                return {
                    "version": "production",
                    "created_at": run.created_at,
                    "run_id": run.id,
                    "f1": summary.get("best_val_score", summary.get("val_f1", "—")),
                    "accuracy": summary.get("val_accuracy", "—"),
                    "loss": summary.get("val_loss", "—"),
                }
        except Exception:
            pass
        # Last resort: read from /info endpoint
        try:
            info = requests.get(f"{API_URL}/info", timeout=4).json()
            return {
                "version": info.get("registry_alias", "production"),
                "created_at": "—",
                "run_id": info.get("run_id", "—"),
                "f1": "—",
                "accuracy": "—",
                "loss": "—",
            }
        except Exception:
            return None

    prod = load_wandb_production()
    if prod:
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Version", prod["version"])
        w2.metric(
            "Run ID",
            (
                str(prod["run_id"])[:10] + "…"
                if len(str(prod["run_id"])) > 10
                else str(prod["run_id"])
            ),
        )
        f1_val = (
            f"{prod['f1']:.3f}" if isinstance(prod["f1"], float) else str(prod["f1"])
        )
        acc_val = (
            f"{prod['accuracy']:.1%}"
            if isinstance(prod["accuracy"], float)
            else str(prod["accuracy"])
        )
        w3.metric("F1 (val)", f1_val)
        w4.metric("Accuracy (val)", acc_val)
        if prod["created_at"] != "—":
            st.caption(f"Enregistré le {prod['created_at']}")
    else:
        st.info(
            "Informations WInformations W&B non disponibles — vérifiezB non disponibles. Verifiez WANDB_API_KEY et WANDB_ENTITY."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 2. État des pipelines Airflow
    # -----------------------------------------------------------------------
    st.markdown("**État des pipelines**")

    @st.cache_data(ttl=30)
    def load_dag_status(dag_id):
        try:
            r = requests.get(
                f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns",
                auth=(AIRFLOW_USER, AIRFLOW_PASS),
                params={"limit": 1, "order_by": "-execution_date"},
                timeout=4,
            )
            if r.status_code == 404:
                return {"state": "not_deployed", "start": "—", "end": "—"}
            if r.status_code == 200:
                runs = r.json().get("dag_runs", [])
                if not runs:
                    return {"state": "no_runs", "start": "—", "end": "—"}
                run = runs[0]
                return {
                    "state": run["state"],
                    "start": run.get("start_date", "—")[:16].replace("T", " "),
                    "end": (
                        run.get("end_date", "—")[:16].replace("T", " ")
                        if run.get("end_date")
                        else "—"
                    ),
                }
        except Exception:
            pass
        return None

    def _state_badge(state):
        colors = {
            "success": "#166534",
            "failed": "#991b1b",
            "running": "#1e40af",
            "queued": "#92400e",
            "not_deployed": "#475569",
            "no_runs": "#475569",
        }
        bg = {
            "success": "#f0fdf4",
            "failed": "#fef2f2",
            "running": "#eff6ff",
            "queued": "#fffbeb",
            "not_deployed": "#f1f5f9",
            "no_runs": "#f1f5f9",
        }
        labels = {
            "not_deployed": "non déployé",
            "no_runs": "aucun run",
        }
        c = colors.get(state, "#334155")
        b = bg.get(state, "#f8fafc")
        display = labels.get(state, state)
        return f"<span style='background:{b};color:{c};padding:2px 10px;border-radius:4px;font-size:.82rem;font-weight:600;'>{display}</span>"

    dags = {
        "radiocovid_pipeline": "Pipeline principal (Ingest > Train > Promote)",
        "radiocovid_monitoring": "Monitoring drift (quotidien)",
    }
    for dag_id, label in dags.items():
        status = load_dag_status(dag_id)
        if status:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:1rem;padding:.5rem 0;"
                f"border-bottom:1px solid #f1f5f9;'>"
                f"<span style='color:#0f172a;font-size:.88rem;flex:1;'>{label}</span>"
                f"{_state_badge(status['state'])}"
                f"<span style='color:#94a3b8;font-size:.78rem;'>Dernier run : {status['start']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:1rem;padding:.5rem 0;"
                f"border-bottom:1px solid #f1f5f9;'>"
                f"<span style='color:#0f172a;font-size:.88rem;flex:1;'>{label}</span>"
                f"<span style='color:#94a3b8;font-size:.82rem;'>Airflow non joignable</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 3. Actions
    # -----------------------------------------------------------------------
    st.markdown("**Actions**")

    def _trigger_dag(dag_id):
        try:
            r = requests.post(
                f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns",
                auth=(AIRFLOW_USER, AIRFLOW_PASS),
                json={"conf": {}},
                timeout=5,
            )
            if r.status_code == 404:
                return False, f"DAG '{dag_id}' non déployé dans cet Airflow."
            return r.status_code in (200, 201), r.text
        except Exception as e:
            return False, str(e)

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        if st.button("Recharger le modèle", use_container_width=True):
            try:
                headers = {"X-API-Key": API_KEY} if API_KEY else {}
                r = requests.post(f"{API_URL}/reload", headers=headers, timeout=10)
                if r.status_code == 200:
                    st.success("Modèle rechargé.")
                else:
                    st.error(f"Erreur {r.status_code}")
            except Exception as e:
                st.error(str(e))

    with a2:
        if st.button("Vérification drift", use_container_width=True):
            ok, msg = _trigger_dag("radiocovid_monitoring")
            if ok:
                st.success("DAG lancé.")
            else:
                st.error(f"Erreur : {msg[:120]}")

    with a3:
        if st.button("Lancer réentraînement", use_container_width=True):
            ok, msg = _trigger_dag("radiocovid_pipeline")
            if ok:
                st.success("Réentraînement lancé.")
            else:
                st.error(f"Erreur : {msg[:120]}")

    with a4:
        if st.button("Vider les logs", use_container_width=True, type="secondary"):
            if PREDICTIONS_FILE.exists():
                PREDICTIONS_FILE.write_text("")
                st.cache_data.clear()
                st.success("Logs vidés.")
            else:
                st.info("Aucun log à vider.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 4. Drift & prédictions
    # -----------------------------------------------------------------------
    st.markdown("**Prédictions en production**")

    df = load_predictions()
    reference = load_reference()
    drift_result = load_drift_result()

    if df.empty:
        st.warning(
            "Aucun log disponible. "
            "Activez ENABLE_INFERENCE_LOGGING=1 pour alimenter ce tableau de bord."
        )
        st.stop()

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df7 = df[df["timestamp"] >= cutoff].copy()
    df7["date"] = df7["timestamp"].dt.date

    n_total = len(df7)
    avg_conf = df7["confidence"].mean() if n_total else 0
    n_normal = (df7["label"] == "NORMAL").sum()
    n_abnormal = n_total - n_normal
    ref_conf = reference.get("features", {}).get("confidence", {}).get("mean", avg_conf)

    # Prioritize Evidently result when available, fallback to simple confidence check
    if drift_result:
        drift_flag = drift_result.get("drift_detected", False)
        n_drifted = len(drift_result.get("drifted_features", []))
        n_features = len(drift_result.get("features", {}))
        checked_at = drift_result.get("checked_at", "")[:16].replace("T", " ")
        if drift_flag:
            st.markdown(
                '<div class="rc-status-warn">'
                f"<strong>Alerte drift (Evidently)</strong> : {n_drifted}/{n_features} features "
                f"ont dérivé significativement (test KS, p&lt;0.05). Dernière analyse : {checked_at} UTC."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="rc-status-ok">'
                f"<strong>Statut nominal (Evidently)</strong> : aucune dérive détectée "
                f"({n_drifted}/{n_features} features, test KS). Dernière analyse : {checked_at} UTC."
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        drift_flag = avg_conf < ref_conf - 0.05
        if drift_flag:
            st.markdown(
                '<div class="rc-status-warn">'
                "<strong>Alerte drift</strong> : la confiance moyenne a baissé de plus de 5 points "
                "par rapport au référentiel. Vérifiez la qualité des images entrantes."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="rc-status-ok">'
                "<strong>Statut nominal</strong> : le modèle se comporte de manière stable "
                "sur les 7 derniers jours."
                "</div>",
                unsafe_allow_html=True,
            )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prédictions (7 j)", n_total)
    k2.metric(
        "Confiance moyenne",
        f"{avg_conf:.1%}",
        delta=f"{(avg_conf - ref_conf):+.1%} vs référentiel",
        delta_color="normal",
    )
    k3.metric("Normal", f"{n_normal}", f"{n_normal/n_total:.0%}" if n_total else "")
    k4.metric(
        "Anormal", f"{n_abnormal}", f"{n_abnormal/n_total:.0%}" if n_total else ""
    )

    # Evidently per-feature KS results
    if drift_result and drift_result.get("features"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Analyse de dérive par feature (Evidently — test KS)**")
        FEAT_LABELS = {
            "img_mean": "Luminosité",
            "img_std": "Contraste",
            "img_entropy": "Entropie",
            "confidence": "Confiance",
        }
        feat_cols = st.columns(len(drift_result["features"]))
        for i, (feat, stats) in enumerate(drift_result["features"].items()):
            label = FEAT_LABELS.get(feat, feat)
            p_val = stats.get("drift_score", float("nan"))
            detected = stats.get("drift_detected", False)
            badge = "🔴 DRIFT" if detected else "🟢 OK"
            feat_cols[i].metric(
                label=label,
                value=badge,
                delta=f"p = {p_val:.3f}",
                delta_color="inverse" if detected else "normal",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Confiance moyenne par jour**")

    daily = df7.groupby("date")["confidence"].agg(["mean", "count"]).reset_index()
    daily.columns = ["date", "mean", "n"]

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(range(len(daily)), daily["mean"], alpha=0.1, color="#0284c7")
    ax.plot(
        range(len(daily)),
        daily["mean"],
        marker="o",
        lw=2,
        color="#0284c7",
        markersize=5,
        label="Confiance moyenne",
    )
    if reference:
        ax.axhline(ref_conf, color="#94a3b8", lw=1.5, ls="--", label="Référentiel")
        ax.fill_between(
            range(len(daily)),
            ref_conf - 0.05,
            ref_conf + 0.05,
            alpha=0.05,
            color="#94a3b8",
        )
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels(
        [str(d) for d in daily["date"]], rotation=25, ha="right", fontsize=8
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Confiance", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    for i, row in daily.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (i, row["mean"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="#94a3b8",
        )
    _style_chart(ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Distribution des features image**")
    st.caption("Courant (bleu) vs référentiel d'entraînement (gris)")

    FEATURES = {
        "img_mean": "Luminosité",
        "img_std": "Contraste",
        "img_entropy": "Entropie",
        "confidence": "Confiance",
    }
    available = [f for f in FEATURES if f in df7.columns]
    cols = st.columns(len(available))

    for i, feat in enumerate(available):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(4, 2.6))
            vals = df7[feat].dropna()
            ax.hist(vals, bins=22, alpha=0.7, color="#0284c7", density=True)
            if reference and feat in reference.get("features", {}):
                r = reference["features"][feat]
                mu, sigma = r["mean"], r["std"]
                x = np.linspace(r["min"], r["max"], 200)
                y = (
                    1
                    / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                )
                ax.plot(x, y, color="#94a3b8", lw=1.5)
            ax.set_yticks([])
            _style_chart(ax, FEATURES[feat])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("**Répartition des labels**")
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.pie(
            [n_normal, n_abnormal],
            labels=["Normal", "Anormal"],
            colors=["#22c55e", "#ef4444"],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9},
        )
        ax.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        st.markdown("**Dernières prédictions**")
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
        "Source : data/inference_logs/predictions.jsonl · "
        "Fenêtre d'analyse : 7 jours"
    )
