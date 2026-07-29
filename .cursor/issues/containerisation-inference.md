# Chantier : Containerisation de l'inférence

**Objectif :** Packager `radiocovid-inference` comme un vrai package Python (cohérent avec ETL et Core), le publier sur le registre PyPI privé, et mettre à jour le Dockerfile pour l'utiliser.

**Ordre d'exécution :** INFER-00 → INFER-01 → INFER-02 → INFER-03 → INFER-04 → INFRA-01 → INFRA-02

---

## INFER-00 : Ajouter `radiocovid-inference` au workspace uv

**Composant :** INFERENCE
**Fichiers concernés :**
- `pyproject.toml` (racine) — ajouter `"radiocovid-inference"` dans `[tool.uv.workspace].members` et `[tool.uv.sources]`

**Dépendances :** aucune

**Critères d'acceptation :**
- [ ] `uv sync` depuis la racine ne génère pas d'erreur liée à `radiocovid-inference`
- [ ] `radiocovid-inference` apparaît dans `uv workspace list`

**Taille :** XS

---

## INFER-01 : Créer le `pyproject.toml` de `radiocovid-inference`

**Composant :** INFERENCE
**Fichiers concernés :**
- `radiocovid-inference/pyproject.toml` — à créer

**Dépendances :** aucune

**Critères d'acceptation :**
- [ ] Package nommé `radiocovid-inference`, version `0.1.0`, `requires-python = ">=3.10,<3.11"`
- [ ] `build-backend = "uv_build"` avec `namespace = true` et `module-name = "radiocovid.inference"`
- [ ] Dépendances déclarées : `torch`, `torchvision`, `fastapi`, `uvicorn[standard]`, `pillow`, `python-dotenv`, `python-multipart`, `wandb`
- [ ] `pip install -e radiocovid-inference/` s'exécute sans erreur

**Taille :** XS

---

## INFER-02 : Déplacer les fichiers source sous `src/radiocovid/inference/` et mettre à jour les imports

**Composant :** INFERENCE
**Fichiers concernés :**
- `radiocovid-inference/src/radiocovid/inference/wandb_download_ckpt.py` — déplacé depuis la racine, aucun import à changer
- `radiocovid-inference/src/radiocovid/inference/predict.py` — déplacé + `from wandb_download_ckpt import ...` → `from radiocovid.inference.wandb_download_ckpt import ...`
- `radiocovid-inference/src/radiocovid/inference/api.py` — déplacé + `from predict import ...` → `from radiocovid.inference.predict import ...`
- `radiocovid-inference/api.py`, `predict.py`, `wandb_download_ckpt.py` (racine) — à supprimer
- **Pas de `__init__.py`** — `namespace = true` dans `pyproject.toml` s'en occupe

**Dépendances :** INFER-01

**Critères d'acceptation :**
- [ ] `from radiocovid.inference.api import app` fonctionne dans un interpréteur Python
- [ ] `from radiocovid.inference.predict import load_model` fonctionne
- [ ] Plus aucun fichier `.py` à la racine de `radiocovid-inference/`

**Taille :** S

---

## INFER-03 : Mettre à jour les imports dans les tests

**Composant :** INFERENCE
**Fichiers concernés :**
- `radiocovid-inference/tests/test_predict.py` — `from predict import ...` → `from radiocovid.inference.predict import ...`
- `radiocovid-inference/tests/test_api.py` — `from api import ...` → `from radiocovid.inference.api import ...`
- `radiocovid-inference/tests/test_wandb_download_ckpt.py` — idem
- `radiocovid-inference/tests/conftest.py` — idem si concerné

**Dépendances :** INFER-02

**Critères d'acceptation :**
- [ ] `pytest radiocovid-inference/tests/` s'exécute sans erreur d'import
- [ ] Aucun test n'importe directement un module sans le namespace `radiocovid.inference`

**Taille :** XS

---

## INFER-04 : Publier `radiocovid-inference` sur le registre PyPI privé

**Composant :** INFERENCE
**Fichiers concernés :** aucun fichier à modifier — commandes à exécuter

**Dépendances :** INFER-01, INFER-02

**Critères d'acceptation :**
- [ ] `uv build --package radiocovid-inference` génère un wheel dans `dist/`
- [ ] `uv publish --package radiocovid-inference` publie sur le registre sans erreur
- [ ] `pip install radiocovid-inference==0.1.0 --index-url <url-registre>` fonctionne depuis un environnement vierge

**Taille :** XS *(retrouver le token et l'URL du registre — 2 commandes une fois récupérés)*

---

## INFRA-01 : Créer `docker/inference/entrypoint.sh`

**Composant :** INFRA
**Fichiers concernés :**
- `docker/inference/entrypoint.sh` — à créer

**Dépendances :** INFER-02

**Critères d'acceptation :**
- [ ] Le script commence par `set -e`
- [ ] Si `WANDB_API_KEY` est vide ou absent, affiche un message d'erreur explicite et retourne un code non-zéro
- [ ] Si la variable est présente, lance `uvicorn radiocovid.inference.api:app --host 0.0.0.0 --port 8000`
- [ ] Le script est rendu exécutable (`chmod +x`) dans le Dockerfile

**Taille :** XS

---

## INFRA-02 : Mettre à jour `docker/inference/Dockerfile`

**Composant :** INFRA
**Fichiers concernés :**
- `docker/inference/Dockerfile` — réécriture complète

**Dépendances :** INFER-04, INFRA-01

**Critères d'acceptation :**
- [ ] `pip install --no-cache-dir radiocovid-inference==0.1.0 --index-url <url-registre>` remplace les installations manuelles
- [ ] `COPY entrypoint.sh /entrypoint.sh` + `RUN chmod +x /entrypoint.sh`
- [ ] `ENTRYPOINT ["/entrypoint.sh"]` remplace le `CMD`
- [ ] Plus aucun `COPY` de fichiers `.py`
- [ ] `docker compose --profile inference build` s'exécute sans erreur

**Taille :** XS
