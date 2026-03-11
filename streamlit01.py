# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
import glob

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="RadioCovid – Version simplifiée",
    page_icon="🫁",
    layout="wide",
)

st.sidebar.title("🫁 Menu")
PAGES = [
    "Introduction & Problématique",
    "EDA",
    "Préprocessing",
    "Architecture du modèle",
    "Modèles",
    "Résultats",
    "Prédiction",
    "Conclusion",
]
page = st.sidebar.radio("Aller à :", PAGES)

# =========================
# CHEMINS
# =========================
ROOT = Path(".")
REPORTS = ROOT / "reports"


# -------------------------------------------------------------------
# 1) INTRODUCTION
# -------------------------------------------------------------------
if page == "Introduction & Problématique":
    st.title("📘 Introduction & Problématique")

    st.markdown("""
## 🎯 Objectif général

Le projet **RadioCovid** vise à analyser automatiquement des radiographies pulmonaires  
afin de détecter différentes pathologies :

- **COVID‑19**
- **Lung Opacity**
- **Viral Pneumonia**
- **Normal**

## 🚨 Problématique

Les radiographies présentent :

- une **grande variabilité** de qualité,
- des **masques pulmonaires parfois incorrects**,
- un **déséquilibre important** entre catégories,
- des **textures pulmonaires hétérogènes**.

Ces éléments rendent indispensable un processus de **nettoyage**,  
**prétraitement**, **rééquilibrage** et **standardisation** avant l’entraînement du modèle.
""")


# -------------------------------------------------------------------
# 2) EDA
# -------------------------------------------------------------------
elif page == "EDA":
    st.title("🔍 EDA — Analyse exploratoire")

    # ----------------------------
    # PARTIE 1 — VOLUMETRIE
    # ----------------------------
    st.header("1️⃣ Volumétrie & structure du dataset")

    dist_fig = REPORTS / "images.png"
    if dist_fig.exists():
        st.image(str(dist_fig), caption="Répartition du dataset par classe", use_container_width=True)
    else:
        st.info("Aucune figure trouvée dans reports/images.png")

    st.markdown("""
Le dataset est très déséquilibré :

- **Normal ~ 10 000**
- **Lung Opacity ~ 6 000**
- **COVID ~ 3 600**
- **Viral Pneumonia ~ 1 300**

➡️ Ce déséquilibre devra être corrigé au moment du rééquilibrage.
""")


    # ----------------------------
    # PARTIE 2 — HARALICK
    # ----------------------------
    
        
    st.header("2️⃣ Analyse des textures (Haralick)")

    # --- Buscar figuras harlick*.png ---
    harlick_figs = sorted(glob.glob(str(REPORTS / "harlick*.png")))

    if not harlick_figs:
        st.info("Aucune figure Haralick trouvée (reports/harlick*.png).")
    else:
        # Estado del carrusel
        if "harlick_pos" not in st.session_state:
            st.session_state["harlick_pos"] = 0

        n = len(harlick_figs)

        # Botones anteriores / siguientes
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            if st.button("◀️", key="harlick_prev"):
                st.session_state["harlick_pos"] = (st.session_state["harlick_pos"] - 1) % n

        with col3:
            if st.button("▶️", key="harlick_next"):
                st.session_state["harlick_pos"] = (st.session_state["harlick_pos"] + 1) % n

        # Imagen actual del carrusel
        current = harlick_figs[st.session_state["harlick_pos"]]
        caption = f"{Path(current).stem}  ({st.session_state['harlick_pos']+1}/{n})"

        col2.image(current, caption=caption, use_container_width=True)

        # Explicación ligera
        st.markdown("""
    Les **caractéristiques de Haralick** (contrast, entropy, homogeneity, energy, correlation…)  
    permettent d’analyser la **texture pulmonaire**.

    ### 🔍 Points clés issus du notebook :
    - **Contrast** est la feature la plus discriminante entre pathologies.  
    - **Entropy** et **Homogeneity** sont fortement **corrélées négativement**.  
    - Les images pathologiques présentent des textures **moins homogènes** que les normales.  
    - La normalisation est essentielle avant l'entraînement.
    """)

    
    # ----------------------------
    # PARTIE 3 — SYMÉTRIE & BORDS
    # ----------------------------
    
        # -------- 3️⃣ Analyse de l’aire pulmonaire (carrusel ciblé) --------
    st.header("3️⃣ Analyse de l’aire pulmonaire (asymétrie & bords)")

    st.markdown("""
    On calcule l’**indice d’asymétrie** entre poumon gauche et droit :

    \\[
    IA = \\frac{|A_g - A_d|}{A_g + A_d}
    \\]

    - IA ~ **0** → forte symétrie (profil sain)  
    - IA **élevé** → asymétrie marquée (potentiellement pathologique)  

    Certaines images présentent aussi des **masques hors cadre** (poumons coupés ou collés aux bords) qui doivent être retirées.
    """)

    # --- Carrusel simple con imágenes específicas: pulmon1,2,3,4,14 ---
    candidatos = [
        REPORTS / "pulmon1.png",
        REPORTS / "pulmon2.png",
        REPORTS / "pulmon3.png",
        REPORTS / "pulmon4.png",
        REPORTS / "pulmon14.png",
        REPORTS / "pulmon7.png",
        REPORTS / "pulmon9.png",
    ]

    pulmon_figs = [p for p in candidatos if p.exists()]

    if not pulmon_figs:
        st.info("ℹ️ Aucune des figures demandées n’a été trouvée (pulmon1/2/3/4/14).")
    else:
        # estado del carrusel (independiente de otros)
        key = "pulmon_focus_carousel"
        if key not in st.session_state:
            st.session_state[key] = 0  # índice 0..n-1

        n = len(pulmon_figs)
        col_prev, col_img, col_next = st.columns([1, 6, 1])

        with col_prev:
            if st.button("◀️", key=key + "_prev", use_container_width=True):
                st.session_state[key] = (st.session_state[key] - 1) % n

        with col_next:
            if st.button("▶️", key=key + "_next", use_container_width=True):
                st.session_state[key] = (st.session_state[key] + 1) % n

        idx = st.session_state[key]
        current = pulmon_figs[idx]
        caption = f"{current.stem}  ({idx+1}/{n})"

        col_img.image(str(current), caption=caption, use_container_width=True)

        # explicación mínima (ligero)
        with st.expander("📝 Interprétation (résumé)"):
            st.markdown("""
    - **Asymétrie** : IA élevé → dissymétrie entre poumons, souvent plus marquée dans les classes pathologiques.  
    - **Masques hors cadre** : poumons coupés/au bord → échantillons à **exclure**.  
    - Le nettoyage (retrait des outliers & masques invalides) **stabilise** le dataset pour l’entraînement.
    """)
# -------------------------------------------------------------------
# 3) PREPROCESSING
# -------------------------------------------------------------------
elif page == "Préprocessing":
    st.title("⚙️ Préprocessing")

    # --------- REMOVE OUTLIERS ---------
    st.header("1️⃣ Remove Outliers (clean.py)")

    st.markdown("""
Le script **clean.py** supprime trois types d’outliers :

### 🔸 1. Masques **hors cadre**
Les poumons touchent les bords → segmentation incorrecte.

### 🔸 2. **Asymétrie extrême**
Indice IA trop élevé → masque incohérent.

### 🔸 3. Outliers **Haralick / GLCM**
Textures anormales détectées via :
- contrast
- entropy
- homogeneity  
Puis filtrage via **IQR** (Inter‑Quartile Range).



➡️ À la fin, on génère un **manifest.parquet** propre.
""")
    
    
    
    # Lista fija de 6 imágenes + títulos en français
    images_outliers = [
        ("boxplot_outliers_haralick.png",  "Boxplot des outliers Haralick"),
        ("exemple_outliers_haralick.png",  "Exemple d’outliers Haralick"),
        ("boxplot_outliers_asymetrie.png", "Boxplot des outliers d’asymétrie"),
        ("exemple_outliers_asymetrie.png", "Exemple d’outliers d’asymétrie"),
        ("boxplot_outliers_bords.png",     "Boxplot des outliers de bords"),
        ("exemple_outliers_bords.png",     "Exemple d’outliers de bords"),
    ]

    # Construcción de rutas existentes
    files = []
    for fname, title in images_outliers:
        p = REPORTS / fname
        if p.exists():
            files.append((p, title))

    if not files:
        st.info("ℹ️ Aucune des images d’outliers demandées n’a été trouvée dans le dossier reports/.")
    else:
        # Estado del carrusel
        key = "outliers_carousel"
        if key not in st.session_state:
            st.session_state[key] = 0

        n = len(files)
        col_prev, col_img, col_next = st.columns([1, 6, 1])

        # Botón anterior
        with col_prev:
            if st.button("◀️", key=f"{key}_prev", use_container_width=True):
                st.session_state[key] = (st.session_state[key] - 1) % n

        # Botón siguiente
        with col_next:
            if st.button("▶️", key=f"{key}_next", use_container_width=True):
                st.session_state[key] = (st.session_state[key] + 1) % n

        # Mostrar imagen actual
        idx = st.session_state[key]
        img_path, title = files[idx]
        caption = f"{title} ({idx+1}/{n})"
        col_img.image(str(img_path), caption=caption, use_container_width=True)

        # Explicación
        with st.expander("Résultat du retrait des outliers"):
            st.markdown("""

    #### 🔸 Outliers Haralick  

    #### 🔸 Outliers d’asymétrie  

    #### 🔸 Outliers de bords  

    """)

    # =========================
    # PREPROCESSING — Sélecteur Binaire / Multiclasse
    # =========================

    # --------- REMOVE OUTLIERS ---------
    st.header("2️⃣ Split all in 2 modes")

    st.header("Sélection du mode de classification")

    # ------ Selector de modo ------
    mode = st.radio(
        "Choisir le mode de classification :",
        ["Mode Binaire (Sain vs Malade)", "Mode Multiclasse (3 pathologies)"],
        horizontal=True,
    )

    # ------ Datos de ejemplo basados en tu dataset real ------
    # Si quieres, puedes sustituir estos valores por los reales más tarde.
    counts_binaire = {
        "SAIN": 10192,
        "MALADE": 3616 + 6012 + 1345,   # COVID + Lung Opacity + Viral Pneumonia
    }

    counts_multiclasse = {
        "COVID": 3616,
        "Lung Opacity": 6012,
        "Viral Pneumonia": 1345,
    }

    # ------ Mostrar sección según modo ------
    if mode.startswith("Mode Binaire"):
        st.subheader("🟦 Mode binaire : SAIN vs MALADE")

        st.markdown("""
    Dans ce mode, le dataset est réduit à **2 classes** :

    - **SAIN** = Normal  
    - **MALADE** = COVID + Lung Opacity + Viral Pneumonia  

    Ce choix permet de simplifier le problème et d'améliorer la détection globale de maladie.
    """)

        # Mostrar números
        st.write("### 📊 Nombre d’images par classe")
        st.write(counts_binaire)

        # Mostrar gráfico (grafico1.png)
        g1 = REPORTS / "grafico2.png"
        if g1.exists():
            st.image(str(g1), caption="Distribution binaire des classes", use_container_width=True)
        else:
            st.warning("⚠️ Impossible de trouver `grafico1.png` dans le dossier reports/.")

    else:
        st.subheader("🟩 Mode multiclasse : 3 pathologies")

        st.markdown("""
    Dans ce mode, le dataset conserve toutes les pathologies séparées :

    - **COVID**
    - **Lung Opacity**
    - **Viral Pneumonia**

    Ce mode permet d’obtenir une classification plus fine,  
    au prix d’un déséquilibre plus important entre classes.
    """)

        # Mostrar números
        st.write("### 📊 Nombre d’images par classe")
        st.write(counts_multiclasse)

        # Mostrar gráfico (grafico2.png)
        g2 = REPORTS / "grafico1.png"
        if g2.exists():
            st.image(str(g2), caption="Distribution multiclasse des pathologies", use_container_width=True)
        else:
            st.warning("⚠️ Impossible de trouver `grafico2.png` dans le dossier reports/.")
# -------------------------------------------------------------------
# 4) ARCHITECTURE DU MODÈLE
# -------------------------------------------------------------------
elif page == "Architecture du modèle":
    st.title("🧠 Architecture du modèle")

    st.markdown("""
Le script **train.py** orchestre toute l’architecture d’entraînement.

## 1️⃣ Rééquilibrage
Oversampling / undersampling selon le mode (binaire ou multiclasse).

## 2️⃣ Découpage des données
Création de :

- **Train**
- **Validation**
- **Test**

## 3️⃣ Transformations / Data Augmentation
Le modèle applique plusieurs transformations :

- zoom in / zoom out  
- ajustement de luminosité  
- variations de contraste  
- normalisation  

## 4️⃣ Entraînement du modèle
Le modèle utilisé est basé sur **VGG11**.

Le script :
- instancie le modèle  
- configure les callbacks  
- lance `trainer.fit()`  
- puis `trainer.test()`  

➡️ Pipeline simple, robuste et adapté aux radiographies.
""")


