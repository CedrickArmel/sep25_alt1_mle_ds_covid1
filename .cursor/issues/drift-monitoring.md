# Chantier : Drift Monitoring (covariate shift sur les images)

**Objectif :** Détecter les dérives sur les **images entrantes** (covariate shift) en surveillant des statistiques visuelles simples et l'évolution de la confidence du modèle — deux signaux indépendants de la prévalence réelle des maladies, donc adaptés au contexte médical.

**Pourquoi pas la distribution NORMAL/ABNORMAL ?**
La proportion de cas anormaux varie naturellement (pandémie, grippe saisonnière). Surveiller cette distribution génèrerait des fausses alarmes en période d'épidémie et masquerait un vrai drift en période creuse. Ce qu'on veut vraiment détecter : les images qui ne ressemblent plus à ce sur quoi le modèle a été entraîné (changement de scanner, de protocole, de preprocessing…).

**Ce qu'on surveille (4 features) :**
- `img_mean` — luminosité moyenne (changement scanner/exposition)
- `img_std` — contraste global (changement de protocole)
- `img_entropy` — complexité de l'histogramme (bruit, qualité image)
- `confidence` — certitude du modèle (proxy de drift indépendant de la prévalence)

**Architecture deux DAGs (recommandation mentor) :**
```
DAG 1 : radiocovid_pipeline (@weekly)
        ingest >> etl >> train >> promote   ← inchangé

DAG 2 : radiocovid_monitoring (@daily)
        drift_check
        └── si drift détecté ET RETRAIN_ON_DRIFT=1
            → déclenche DAG 1 via API Airflow
```

**Stockage :**
- Logs de prédictions → **W&B Table** `inference_logs` (source de vérité, accessible via SDK)
- Rapports HTML Evidently → **W&B Artifact** + copie locale `reports/`
- Référentiel de training → **W&B Artifact** `reference_distribution:latest`

**Ordre d'exécution :** DRIFT-01 → DRIFT-02 → DRIFT-03 → DRIFT-04 → DRIFT-05 → DRIFT-06

---

## DRIFT-01 : Logger chaque prédiction + features image dans une W&B Table

**Composant :** INFERENCE
**Fichiers concernés :**
- `radiocovid-inference/src/radiocovid/inference/inference_logger.py` — nouveau module
- `radiocovid-inference/src/radiocovid/inference/api.py` — appel au logger dans `/predict`
- `.env.example` — `ENABLE_WANDB_LOGGING` (défaut `0`), `WANDB_INFERENCE_LOG_TABLE` (défaut `inference_logs`)

**Dépendances :** aucune

**Pourquoi :** Chaque appel `/predict` extrait 3 stats de l'image brute + la confidence et les logge dans une W&B Table. Ce store alimente le drift check périodique et la page Streamlit.

**Features extraites (image PIL brute, avant transform) :**
```python
import numpy as np
from scipy.stats import entropy as scipy_entropy

def extract_image_features(image: Image.Image) -> dict:
    arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=256, range=(0, 1), density=True)
    return {
        "img_mean":    float(arr.mean()),
        "img_std":     float(arr.std()),
        "img_entropy": float(scipy_entropy(hist + 1e-10)),
    }
```

**Schéma de la W&B Table (une ligne par prédiction) :**

| Colonne | Type | Description |
|---|---|---|
| `timestamp` | str ISO 8601 | Horodatage de la requête |
| `label` | str | Prédiction (`NORMAL` / `ABNORMAL`) |
| `confidence` | float | Score de confiance (0–1) |
| `img_mean` | float | Luminosité moyenne |
| `img_std` | float | Écart-type des pixels |
| `img_entropy` | float | Entropie de l'histogramme |
| `model_run_id` | str | `run_id` du modèle en production |

**Critères d'acceptation :**
- [ ] Après 3 appels à `/predict`, la W&B Table contient 3 lignes avec les 7 colonnes
- [ ] `ENABLE_WANDB_LOGGING=0` → aucun appel W&B (tests unitaires non cassés)
- [ ] Log non-bloquant : échec W&B = warning, pas d'erreur 500 sur `/predict`
- [ ] Aucune nouvelle dépendance lourde (`scipy` et `numpy` déjà présents)

**Taille :** S

---

## DRIFT-02 : Construire le référentiel (distribution val/test set)

**Composant :** INFERENCE / SCRIPTS
**Fichiers concernés :**
- `scripts/build_reference.py` — script à créer
- Sorties : `data/reference_distribution.json` + artefact W&B `reference_distribution:latest`

**Dépendances :** aucune (s'exécute une fois sur le val/test set existant, puis après chaque retrain)

**Pourquoi :** Le référentiel est la "photo" de la distribution des images d'entraînement. Evidently la compare aux images entrantes pour détecter si elles ont changé. Il faut le recalculer à chaque nouveau modèle promu en production.

**Format du référentiel (`reference_distribution.json`) :**
```json
{
  "n_samples": 1250,
  "created_at": "2026-08-13T14:00:00Z",
  "model_run_id": "abc123",
  "features": {
    "img_mean":    {"mean": 0.48, "std": 0.09, "p5": 0.32, "p95": 0.63},
    "img_std":     {"mean": 0.21, "std": 0.05, "p5": 0.13, "p95": 0.30},
    "img_entropy": {"mean": 5.12, "std": 0.41, "p5": 4.40, "p95": 5.78},
    "confidence":  {"mean": 0.87, "std": 0.11, "p5": 0.63, "p95": 0.98}
  }
}
```

**Critères d'acceptation :**
- [ ] `python scripts/build_reference.py --data-dir data/train_folder/val` génère le JSON + l'artefact W&B
- [ ] Couvre au minimum les 4 features sur ≥ 100 images
- [ ] `make build-reference` : cible Makefile qui lance le script
- [ ] Réexécutable après retrain avec `--overwrite` pour mettre à jour le référentiel

**Taille :** S

---

## DRIFT-03 : Script de détection de drift (`run_drift_check.py`)

**Composant :** SCRIPTS
**Fichiers concernés :**
- `scripts/run_drift_check.py` — point d'entrée CLI
- `radiocovid-inference/src/radiocovid/inference/drift_check.py` — logique Evidently (séparé pour testabilité)

**Dépendances :** DRIFT-01, DRIFT-02

**Pourquoi :** Ce script est le cœur du chantier. Il charge les prédictions récentes depuis W&B, charge le référentiel, lance Evidently (KS test sur les 4 features), et publie le résultat dans W&B.

**Logique Evidently :**
- `ColumnDriftMetric` sur chaque feature : `img_mean`, `img_std`, `img_entropy`, `confidence`
- Test statistique : **KS test** (Kolmogorov-Smirnov) — adapté aux distributions continues
- `drift_detected = True` si ≥ 2 features sur 4 dérivent (seuil configurable via `DRIFT_THRESHOLD_FEATURES`)

**Critères d'acceptation :**
- [ ] `python scripts/run_drift_check.py --window 7d --min-samples 50` :
  - Charge les prédictions de la fenêtre depuis W&B
  - Si < `min-samples` → exit 0 + message "Not enough data, skipping"
  - Si ≥ `min-samples` → Evidently KS test → logge dans W&B : `drift_detected`, `drifted_features`, score par feature
  - Rapport HTML dans `reports/drift_<YYYYMMDD>.html` ET pushé comme artefact W&B
- [ ] `DRIFT_FAIL_ON_DETECT=1` → exit code 1 si drift détecté
- [ ] `DRIFT_FAIL_ON_DETECT=0` (défaut) → exit 0 toujours

**Dépendance à ajouter :** `evidently>=0.4`

**Taille :** M

---

## DRIFT-04 : Image Docker `radiocovid-drift:0.1.0`

**Composant :** INFRA
**Fichiers concernés :**
- `docker/drift/Dockerfile` — Python 3.10-slim + evidently + wandb + pandas + scipy
- `docker/drift/entrypoint.sh` — appelle `scripts/run_drift_check.py`
- `docker-compose.yml` — service `drift` (profil `drift`)
- `Makefile` — cibles `drift-build`, `drift-run`

**Dépendances :** DRIFT-03

**Critères d'acceptation :**
- [ ] `docker build -t radiocovid-drift:0.1.0 docker/drift/` sans erreur
- [ ] `docker run --rm -e WANDB_API_KEY=… radiocovid-drift:0.1.0` exécute le drift check
- [ ] Service `drift` dans `docker-compose.yml` sous profil `drift`
- [ ] `make drift-build` + `make drift-run` fonctionnels

**Taille :** S

---

## DRIFT-05 : Nouveau DAG Airflow `radiocovid_monitoring`

**Composant :** INFRA
**Fichiers concernés :**
- `dags/radiocovid_monitoring.py` — **nouveau fichier DAG** (ne pas modifier `radiocovid_pipeline.py`)
- `docker-compose.yml` — variables d'env drift passées au scheduler Airflow
- `.env.example` — nouvelles variables drift
- `README.md` — section "Drift monitoring — DAG radiocovid_monitoring"

**Dépendances :** DRIFT-04

**Pourquoi un second DAG ?**
Séparer les responsabilités : le DAG d'entraînement tourne `@weekly`, le monitoring peut tourner `@daily`. On peut les activer/désactiver indépendamment. C'est la pratique standard en MLOps (recommandation mentor).

**DAG `radiocovid_monitoring` :**
```
drift_check (DockerOperator, radiocovid-drift:0.1.0)
    │
    └── si drift_detected=True ET RETRAIN_ON_DRIFT=1
        → TriggerDagRunOperator → radiocovid_pipeline
```

**Critères d'acceptation :**
- [ ] `dags/radiocovid_monitoring.py` existe avec `schedule="@daily"`, `dag_id="radiocovid_monitoring"`
- [ ] Tâche `drift_check` via `DockerOperator` (même image `radiocovid-drift:0.1.0`)
- [ ] Tâche `trigger_retrain` via `TriggerDagRunOperator` avec condition `RETRAIN_ON_DRIFT=1`
- [ ] `RETRAIN_ON_DRIFT=0` par défaut dans `.env.example` (non-bloquant)
- [ ] `DRIFT_FAIL_ON_DETECT` et `RETRAIN_ON_DRIFT` documentés dans le README
- [ ] `radiocovid_pipeline.py` **inchangé** (les deux DAGs sont indépendants)

**Taille :** S

---

## DRIFT-06 : Page "Monitoring" dans l'app Streamlit

**Composant :** APP
**Fichiers concernés :**
- `radiocovid-app/src/radiocovid/app/app.py` — page monitoring

**Dépendances :** DRIFT-01 (W&B Table remplie)

**Ce que la page affiche :**
1. **Confidence dans le temps** — courbe de la confidence moyenne par jour (7 derniers jours)
2. **Histogrammes features image** — `img_mean`, `img_std`, `img_entropy` : courant (coloré) vs référentiel (gris)
3. **Badge statut drift** — issu du dernier run W&B : `✅ Aucun drift` ou `⚠️ Drift sur : img_mean, confidence`

**Critères d'acceptation :**
- [ ] Page "📊 Monitoring" dans la sidebar Streamlit
- [ ] Courbe confidence moyenne par jour
- [ ] Histogrammes superposés (référentiel vs courant) pour les 3 features image
- [ ] Badge drift basé sur le dernier artefact W&B `drift_detected`
- [ ] Fallback gracieux si W&B inaccessible ou table vide : "Pas encore de données"

**Taille :** S

---

## Résumé visuel complet

```
[API FastAPI /predict]
   │  extract_image_features() → img_mean, img_std, img_entropy
   │  + confidence, label, timestamp, model_run_id
   ▼
[W&B Table "inference_logs"]        ← DRIFT-01
   │
   ├── [scripts/build_reference.py] ← DRIFT-02
   │   Référentiel val/test → W&B Artifact "reference_distribution:latest"
   │
   └── [scripts/run_drift_check.py] ← DRIFT-03
       Evidently KS test (4 features)
       → W&B : drift_detected, scores
       → W&B Artifact : rapport HTML
       → reports/drift_YYYYMMDD.html (copie locale)

[Docker radiocovid-drift:0.1.0]     ← DRIFT-04

[DAG 1 : radiocovid_pipeline @weekly]   ← INCHANGÉ
ingest >> etl >> train >> promote

[DAG 2 : radiocovid_monitoring @daily]  ← DRIFT-05 (nouveau fichier)
drift_check → (si drift + RETRAIN_ON_DRIFT=1) → TriggerDagRun → DAG 1

[Streamlit page "Monitoring"]       ← DRIFT-06
- confidence timeline
- histogrammes (courant vs référentiel)
- badge drift OK / alerte
```

---

## Tableau récap

| Ticket | Composant | Ce que ça fait | Statut | Taille |
|---|---|---|---|---|
| DRIFT-01 | INFERENCE | Logger features image + confidence → JSONL local | ✅ Fait | S |
| DRIFT-02 | SCRIPTS | Référentiel distribution val/test set → JSON + W&B Artifact | ✅ Fait | S |
| DRIFT-03 | SCRIPTS | Drift check Evidently (KS test, 4 features) + rapport W&B | ✅ Fait | M |
| DRIFT-04 | INFRA | Image Docker `radiocovid-drift:0.1.0` | À faire | S |
| DRIFT-05 | INFRA | Nouveau DAG `radiocovid_monitoring` (@daily, TriggerDagRun) | À faire | S |
| DRIFT-06 | APP | Page Streamlit "Monitoring" (confidence + histogrammes + badge) | À faire | S |

---

## Variables d'environnement nouvelles

| Variable | Défaut | Rôle |
|---|---|---|
| `ENABLE_WANDB_LOGGING` | `0` | Active le log des prédictions vers W&B (`1` en prod) |
| `WANDB_INFERENCE_LOG_TABLE` | `inference_logs` | Nom de la W&B Table de logs |
| `DRIFT_WINDOW_DAYS` | `7` | Fenêtre temporelle d'analyse (jours) |
| `DRIFT_MIN_SAMPLES` | `50` | Minimum de prédictions pour lancer l'analyse |
| `DRIFT_FAIL_ON_DETECT` | `0` | `1` = le container drift fail si drift détecté |
| `DRIFT_REPORT_DIR` | `reports/` | Dossier pour les rapports HTML Evidently |
| `RETRAIN_ON_DRIFT` | `0` | `1` = déclenche automatiquement `radiocovid_pipeline` si drift détecté |
