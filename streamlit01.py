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
from pathlib import Path

import streamlit as st

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
        st.image(
            str(dist_fig),
            caption="Répartition du dataset par classe",
            use_container_width=True,
        )
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
        col1, col2, col3 = st.columns([1, 6, 1])

        with col1:
            if st.button("◀️", key="harlick_prev"):
                st.session_state["harlick_pos"] = (
                    st.session_state["harlick_pos"] - 1
                ) % n

        with col3:
            if st.button("▶️", key="harlick_next"):
                st.session_state["harlick_pos"] = (
                    st.session_state["harlick_pos"] + 1
                ) % n

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
    st.latex(r"IA = \frac{|A_g - A_d|}{A_g + A_d}")
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
        REPORTS / "pulmon13.png",
        REPORTS / "pulmon7.png",
        REPORTS / "pulmon11.png",
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
        ("boxplot_outliers_haralick.png", "Boxplot des outliers Haralick"),
        ("exemple_outliers_haralick.png", "Exemple d’outliers Haralick"),
        ("boxplot_outliers_asymetrie.png", "Boxplot des outliers d’asymétrie"),
        ("exemple_outliers_asymetrie.png", "Exemple d’outliers d’asymétrie"),
        ("boxplot_outliers_bords.png", "Boxplot des outliers de bords"),
        ("exemple_outliers_bords.png", "Exemple d’outliers de bords"),
    ]

    # Construcción de rutas existentes
    files = []
    for fname, title in images_outliers:
        p = REPORTS / fname
        if p.exists():
            files.append((p, title))

    if not files:
        st.info(
            "ℹ️ Aucune des images d’outliers demandées n’a été trouvée dans le dossier reports/."
        )
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

    st.header("Sélection du mode de séparation")

    # ------ Selector de modo ------
    mode = st.radio(
        "Choisir le mode de séparation :",
        ["Mode Binaire (Sain vs Malade)", "Mode Multiclasse (3 pathologies)"],
        horizontal=True,
    )

    counts_binaire = {
        "SAIN": 10192,
        "MALADE": 3616 + 6012 + 1345,  # COVID + Lung Opacity + Viral Pneumonia
    }

    counts_multiclasse = {
        "COVID": 3616,
        "Lung Opacity": 6012,
        "Viral Pneumonia": 1345,
    }

    # ------ Mostrar sección según modo ------
    if mode.startswith("Mode Binaire"):
        st.subheader("🟦 Mode binaire : SAIN vs MALADE")

        st.markdown("""  Dans ce mode, le dataset est réduit à **2 classes** :

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
            col1, col2, col3 = st.columns([1, 2, 1])  # centrer + réduire largeur
            with col2:
                st.image(str(g1), caption="Distribution binaire des classes", width=350)
        else:
            st.warning(
                "⚠️ Impossible de trouver `grafico2.png` dans le dossier reports/."
            )
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

        # Mostrar gráfico (grafico1.png)
        g1 = REPORTS / "grafico1.png"
        if g1.exists():
            col1, col2, col3 = st.columns([1, 2, 1])  # centrer + réduire largeur
            with col2:
                st.image(
                    str(g1), caption="Distribution par patlogies des classes", width=350
                )
        else:
            st.warning(
                "⚠️ Impossible de trouver `grafico2.png` dans le dossier reports/."
            )
# -------------------------------------------------------------------
# 4) ARCHITECTURE DU MODÈLE
# -------------------------------------------------------------------
elif page == "Architecture du modèle":
    st.title("🧠 Architecture du modèle")

    # =========================
    # 1️⃣ RÉÉQUILIBRAGE — versión simple
    # =========================
    st.header("1️⃣ Rééquilibrage du dataset")

    st.subheader("Distribution initiale (avant rééquilibrage)")
    st.markdown("""Voici le nombre d’images par catégorie avant tout traitement :

- **Normal** : 10 192
- **Lung Opacity** : 6 012
- **COVID** : 3 616
- **Viral Pneumonia** : 1 345

La classe *Normal* domine largement le dataset. Les classes pathologiques sont beaucoup plus petites.
""")

    mode_rb = st.radio(
        "Sélectionner le type de rééquilibrage :",
        ["Mode Binaire (Sain vs Malade)", "Mode Multiclasse (3 pathologies)"],
        horizontal=True,
        key="rb_mode_archi",
    )

    if mode_rb.startswith("Mode Binaire"):
        st.subheader("Mode Binaire (Sain vs Malade)")
        st.markdown(
            """Regroupement des classes. Dans ce mode, on regroupe les catégories ainsi :

- **SAIN** = Normal    → **10 192 images**
- **MALADE** = COVID + Lung Opacity + Viral Pneumonia  → 3 616 + 6 012 + 1 345  → **10 973 images**

On obtient donc **deux groupes presque équilibrés**, mais pas parfaitement identiques.
On utilise un **WeightedRandomSampler**, qui ajuste les poids des classes :

- SAIN → **légèrement oversamplé**
- MALADE → **légèrement undersamplé**

Résultat dans le train set : le modèle voit pendant l’entraînement environ :

- **50 % SAIN**
- **50 % MALADE**
"""
        )
    else:
        st.subheader("Mode Multiclasse (3 pathologies)")
        st.markdown("""
Très fort déséquilibre : VP << COVID < LO << Normal.

Toujours via le **WeightedRandomSampler**, mais cette fois par classe individuelle :

- **Viral Pneumonia** → **fort oversampling** (classe très petite)
- **COVID** → oversampling modéré
- **Lung Opacity** → léger oversampling

Résultat (pendant l’entraînement) : classes **équiprobables** (≈ 33% chacune).
""")

    # =========================
    # 2️⃣ DÉCOUPAGE DES DONNÉES
    # =========================
    st.header("2️⃣ Découpage du dataset : Train / Validation / Test")
    st.markdown(
        """Le dataset est séparé en trois sous‑ensembles afin d’assurer un entraînement fiable et une évaluation correcte du modèle.


- **Train** : apprentissage du modèle   ~ 70%
- **Validation** : réglage d’hyperparamètres / early‑stopping  ~ 10%
- **Test** : évaluation finale ~ 20%

"""
    )

    # =========================
    # 3️⃣ TRANSFORMATIONS / DATA AUGMENTATION
    # =========================
    st.header("3️⃣ Transformations / Data Augmentation")
    st.markdown("""
La **Data Augmentation** rend le modèle robuste aux conditions d’acquisition :

- **Zoom in / out** : invariance d’échelle
- **Luminosité** : radios claires/sombres
- **Contraste** : qualité variable
- **Normalisation** : stabilise et accélère l’apprentissage
""")

    # Carrusel simple de transformations
    transfo_images = []
    idx_img = 1
    while True:
        path = REPORTS / f"modificacion{idx_img}.png"
        if path.exists():
            transfo_images.append((path, f"Modification {idx_img}"))
            idx_img += 1
        else:
            break

    if not transfo_images:
        st.info("ℹ️ Aucune image trouvée (modificacion1.png, modificacion2.png, …).")
    else:
        if "transfo_idx_arch" not in st.session_state:
            st.session_state["transfo_idx_arch"] = 0
        n = len(transfo_images)
        c1, cimg, c3 = st.columns([1, 6, 1])
        with c1:
            if st.button("◀️", key="transfo_prev_arch", use_container_width=True):
                st.session_state["transfo_idx_arch"] = (
                    st.session_state["transfo_idx_arch"] - 1
                ) % n
        with c3:
            if st.button("▶️", key="transfo_next_arch", use_container_width=True):
                st.session_state["transfo_idx_arch"] = (
                    st.session_state["transfo_idx_arch"] + 1
                ) % n
        i = st.session_state["transfo_idx_arch"]
        img_path, title = transfo_images[i]
        cimg.image(
            str(img_path), caption=f"{title}  ({i+1}/{n})", use_container_width=True
        )

    # =========================
    # 4️⃣ ENTRAÎNEMENT DU MODÈLE (dépliables)
    # =========================
    st.header("4️⃣ Entraînement du modèle")

    with st.expander("1) Problème & choix de paradigme", expanded=False):
        st.markdown("""
Problème de **classification d’images médicales** (radios).
Paradigme : **CNN** → extraction automatique de motifs (contours, textures, structures pulmonaires).
""")

    with st.expander("2) Organisation (Hydra + W&B)"):
        st.markdown("""
**Hydra** : configuration modulaire (hyperparamètres, modèle, optimiseur, callbacks…) séparée du code.
**W&B** : suivi d’expériences (métriques, paramètres, comparaisons) → traçabilité & reproductibilité.
""")

    with st.expander("3) Modèles (Vanilla CNN & transfert learning)"):
        st.markdown("""
- **Vanilla CNN** (from scratch) : blocs Conv→ReLU→Pool, classifieur fully‑connected (+ Dropout).
- **Transfert learning** : **VGG / ResNet50 / RegNet_Y_128_GF/ VGG11** adaptés au contexte radiographique.
""")

    with st.expander("4) Détails du Vanilla CNN"):
        st.markdown("""
Entrée : **256×256, 3 canaux**.
**Extracteur** (×3) : Conv2d (3→32→64→128, kernel 3×3, padding 1) + ReLU + MaxPool2d(2).
**Classifieur** : Flatten → FC (+ **Dropout**), sortie en logits (binaire:2, multiclasse:K).
**Cross‑Entropy Loss** (classes exclusives) : compare prédiction vs réalité (probabilités Softmax).
""")

    with st.expander("5) Optimisateur"):
        st.markdown("""
**Adam** : LR adaptatif, convergence rapide & stable, bonne robustesse pour CNN from scratch.
""")


# =========================
# 5) MODÈLES — Section complète
# =========================
elif page == "Modèles":

    st.title("🧪 Modèles – Caractéristiques & usages")

    st.markdown(
        """    Cette section présente les architectures évaluées et leurs propriétés principales.
    Les scores Acc@1 / Acc@5 sont fournis pour un pré‑entraînement ImageNet indiqué par le jeu de poids.
    """
    )

    with st.expander("🏆 RegNet_Y_128GF — très grande capacité (transfer learning)"):
        st.markdown("""
    **Poids** : `RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1`
    **Acc@1 / Acc@5** : **88.228** / **98.682**
    **Paramètres** : **644.8 M**
    **GFLOPs** : **374.57**

    **Ce que c’est**
    - Famille RegNet (design régulier, scalable), variante **Y_128GF** = **très grande** capacité.
    - Excellente performance, mais **coût de calcul très élevé** (GFLOPs) et beaucoup de paramètres.

    **Quand l’utiliser ?**
    - Si vous disposez de **GPU puissants** et visez la **meilleure performance absolue**.
    - Pour **fine‑tuning** sur un dataset médical assez large.

    **Trade‑offs**
    - ✅ Très haut plafond de performance
    - ❌ Coût mémoire/compute **très élevé**, latence plus longue
    """)

    with st.expander("⭐ ResNet50 — standard robuste & efficace"):
        st.markdown("""
    **Poids** : `ResNet50_Weights.IMAGENET1K_V2`
    **Acc@1 / Acc@5** : **80.858** / **95.434**
    **Paramètres** : **25.6 M**
    **GFLOPs** : **4.09**

    **Ce que c’est**
    - Architecture **résiduelle** très éprouvée, excellent **rapport perf/compute**.
    - Idéale pour **transfert learning** en imagerie médicale.

    **Quand l’utiliser ?**
    - Baseline **sérieuse** si vous souhaitez **équilibre** entre performance, vitesse et taille.
    - Très bon choix si les ressources GPU sont **modérées**.

    **Trade‑offs**
    - ✅ Robuste, facile à fine‑tuner, rapide
    - ❌ Moins performante que des modèles **très grands** (RegNet géants, etc.)
    """)

    with st.expander("🧱 VGG11_BN — simple & stable (batch norm)"):
        st.markdown("""
    **Poids** : `VGG11_BN_Weights.IMAGENET1K_V1`
    **Acc@1 / Acc@5** : **70.37** / **89.81**
    **Paramètres** : **132.9 M**
    **GFLOPs** : **7.61**

    **Ce que c’est**
    - Architecture **VGG** (empilement conv 3×3), version **11 couches** avec **BatchNorm**.
    - Plus **stable** que VGG11 “sec”, converge mieux.

    **Quand l’utiliser ?**
    - Scénarios **pédagogiques** où vous souhaitez un modèle clair, **interprétable**, et **facile** à ajuster.
    - Cuando la **simplicidad** prima sobre la performance.

    **Trade‑offs**
    - ✅ Simplicidad, estabilidad (BN)
    - ❌ **Muchos parámetros** para la precisión que ofrece; puede ser más lento que ResNet50 para igual rendimiento
    """)

    with st.expander("🧱 VGG11 — baseline simple (sans BN)"):
        st.markdown("""
    **Poids** : `VGG11_Weights.IMAGENET1K_V1`
    **Acc@1 / Acc@5** : **69.02** / **88.628**
    **Paramètres** : **132.9 M**
    **GFLOPs** : **7.61**

    **Ce que c’est**
    - Version VGG11 **sans batch norm**.
    - Architecture simple (conv 3×3 + maxpool).

    **Quand l’utiliser ?**
    - Démonstrations **pédagogiques** et **prototypage** ultra simple.

    **Trade‑offs**
    - ✅ Très simple de compréhension
    - ❌ **Moins stable** que VGG11_BN, **beaucoup** de paramètres pour une performance **modeste**
    """)

    with st.expander("🧪 VanillaCNN — modèle from‑scratch (projet)"):
        st.markdown("""
    **Poids** : _(non pré‑entraîné)_
    **Acc@1 / Acc@5** : _(non applicable en ImageNet)_
    **Paramètres** : **~16 M** (ordre de grandeur)
    **GFLOPs** : _(non mesuré)_

    **Ce que c’est**
    - **CNN personnalisé** (blocs Conv → ReLU → MaxPool, puis classifieur + Dropout).
    - Conçu pour **apprendre depuis zéro** sur votre jeu de données.

    **Quand l’utiliser ?**
    - Pour **comprendre** et **contrôler** entièrement l’architecture.
    - Comme **baseline pédagogique** et de recherche.

    **Trade‑offs**
    - ✅ Totalement configurable, transparent
    - ❌ Sans pré‑entraînement, nécessite **plus de données** / **data aug** pour rivaliser avec transfer learning
    """)

    st.subheader("🔗 Tableau de bord des expériences (Weights & Biases)")

    col_a, col_b, col_c = st.columns([1, 2, 1])

    with col_b:
        if st.button("📊 Ouvrir le rapport W&B", use_container_width=True):
            st.markdown(
                """
                <script>
                    window.open(
                        'https://wandb.ai/yebouetc/radiocovid/reports/Radio-Covid--VmlldzoxNjE3MjA0MQ',
                        '_blank'
                    );
                </script>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("""
Le rapport W&B présente l'historique complet des entraînements :
- courbes de loss / accuracy
- évolution des métriques
- comparaison des modèles
- hyperparamètres utilisés
- checkpoints sauvegardés

Idéal pour analyser finement les performances du modèle.
""")

# =========================
# 7) PRÉDICTION — Version démo
# =========================
elif page == "Prédiction":
    st.title("🔮 Prédiction sur une radiographie")

    st.markdown("""
Cette section permet de tester la prédiction sur une image importée.


""")

    # ----- Upload d'image -----
    st.subheader("📤 Importer une radiographie")
    uploaded_file = st.file_uploader(
        "Choisissez une image (png/jpg/jpeg) :", type=["png", "jpg", "jpeg"]
    )

    img = None
    if uploaded_file is not None:
        from PIL import Image

        try:
            img = Image.open(uploaded_file).convert("RGB")
            col_l, col_c, col_r = st.columns([1, 2, 1])
            with col_c:
                st.image(img, caption="Aperçu de l'image importée", width=350)
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image : {e}")

    # ----- Bouton prédiction -----
    st.subheader("🤖 Lancer la prédiction")

    if st.button("Prédire", type="primary", use_container_width=True):
        if uploaded_file is None:
            # Aucun fichier → résultat par défaut
            st.warning("Aucune image importée. Exemple de résultat :")

        else:
            # Image fournie → COVID 82%
            st.success("Prédiction réalisée (version démo)")

    st.markdown("""
---
👉 Lorsque le modèle final sera disponible, cette section permettra :
- le pré‑traitement automatique de la radiographie
- le chargement du checkpoint
- la prédiction réelle
- l’affichage des probabilités
- éventuellement une heatmap **Grad‑CAM**
""")

    # =========================
# 8) CONCLUSION & DISCUSSION
# =========================
elif page == "Conclusion":
    st.title("🔚 Conclusion & Discussion")

    st.markdown("""
## 🧾 Conclusion générale

Ce projet a permis de construire un pipeline complet de **classification de radiographies pulmonaires**,
depuis l’analyse exploratoire (**EDA**) jusqu’à l’entraînement du modèle, en passant par le
**nettoyage**, le **rééquilibrage**, la **data augmentation** et la **construction d’architectures CNN**.

Les principales étapes réalisées sont :

- **Nettoyage du dataset** : identification et suppression des outliers
  (masques hors cadre, asymétries excessives, textures aberrantes).
- **Rééquilibrage intelligent** : binaire (Sain/Malade) ou multiclasse (3 pathologies),
  grâce à un **WeightedRandomSampler** permettant des minibatchs équilibrés.
- **Data Augmentation** : zoom, luminosité, contraste, normalisation
  → renforcement de la robustesse du modèle.
- **Modélisation** : utilisation d’un **Vanilla CNN** et d’architectures pré‑entraînées
  (VGG, ResNet, RegNet), orchestrées par **Hydra** et suivies avec **Weights & Biases**.
- **Prédiction** : interface simple permettant de charger une radiographie et d’obtenir une sortie simulée.

---

## 💬 Discussion et perspectives

Même si les résultats obtenus démontrent la **faisabilité** et la **cohérence** du pipeline,
plusieurs axes d’amélioration sont possibles :

### 🔸 1. Approfondissement du modèle
- Tester davantage de modèles pré‑entraînés plus légers ou spécifiques au médical.
- Optimiser le choix des hyperparamètres (learning rate, scheduler, batch size).

### 🔸 2. Amélioration du dataset
- Intégrer davantage de données réelles (autres bases publiques).
- Raffiner encore la détection d’outliers et la qualité des masques.

### 🔸 3. Validation clinique
- Mesurer la performance sur des données externes.
- Étudier la stabilité par sous‑groupes (exposition, contrastes, pathologies complexes).

### 🔸 4. Interprétabilité
- Ajouter des visualisations de type **Grad‑CAM** pour mieux expliquer
  les zones pulmonaires activées par le modèle.

---

## 🏁 En résumé

Ce projet pose les bases d’un pipeline robuste et reproductible pour la
classification automatique de radiographies pulmonaires,
grâce à une combinaison de :

- **qualité des données**,
- **équité du rééquilibrage**,
- **augmentation réaliste**,
- **modélisation claire et traçable**,
- **architecture flexible et extensible**.

Il constitue un point de départ solide pour aller vers des modèles plus
performants, plus interprétables et plus adaptés au contexte clinique réel.
""")
