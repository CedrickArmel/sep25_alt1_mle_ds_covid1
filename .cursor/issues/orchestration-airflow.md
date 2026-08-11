# Chantier : Orchestration Airflow

## C'est quoi Airflow ? (explication simple)

Imagine que tu dois exécuter plusieurs tâches dans un ordre précis, à intervalles réguliers.
Par exemple : chaque semaine, il faut d'abord nettoyer les données, *puis* entraîner le modèle.
Si tu le fais à la main, tu oublies, tu te trompes d'ordre, tu ne sais pas si ça a planté.

**Airflow, c'est un chef d'orchestre automatique.**

Concrètement :
- Tu décris tes tâches et leur ordre dans un fichier Python appelé un **DAG** (Directed Acyclic Graph = graphe de tâches sans boucle)
- Airflow les exécute automatiquement selon un calendrier (ex: `@weekly`)
- Il a une **interface web** pour voir ce qui s'est bien passé, ce qui a planté, relancer une tâche en échec
- Si une étape échoue, la suivante ne démarre pas (ex: si l'ETL plante, on n'entraîne pas sur des données corrompues)

## Dans notre projet

Pipeline cible orchestré par Airflow :

```
[1] Ingest + versionnage (Drive → DVC/Git)
        ↓
[2] ETL / preprocessing
        ↓
[3] Train  (+ test intégré, métriques W&B)
        ↓
[4] Promote  (register_model.py → @production si meilleur)
```

L'inférence reste hors Airflow : l'API charge déjà le modèle flagué `@production`.
L'évaluation séparée n'est pas une tâche Airflow : elle est déjà faite en fin de train (`test: true`) ; la promotion se base sur `best_val_score` (validation).

Sans Airflow : on lancerait chaque étape à la main.
Avec Airflow : on configure une fois, et la chaîne tourne quand de nouvelles données arrivent (ou sur schedule).

## Ce qui est déjà fait vs ce qui reste

**Ordre d'exécution :** INFRA-AF-01 → AF-02 → AF-03 → AF-04 → AF-05 → AF-06 → AF-09 → AF-10 → AF-07
*(AF-08 = obsolète car AF-09 déplace l'ingest dans Airflow)*

---

## INFRA-AF-01 ✅ : Créer l'image Docker Airflow custom

**Composant :** INFRA
**Fichiers concernés :**
- `docker/airflow/Dockerfile` — image basée sur `apache/airflow:2.9.2-python3.10` + provider Docker

**Dépendances :** aucune

**Pourquoi :** Airflow a besoin du plugin `apache-airflow-providers-docker` pour pouvoir lancer des containers Docker depuis un DAG. L'image officielle ne l'inclut pas par défaut.

**Critères d'acceptation :**
- [x] `docker/airflow/Dockerfile` existe et installe `apache-airflow-providers-docker==3.9.2`
- [x] L'image se build sans erreur (`docker build -t radiocovid-airflow:2.9.2 docker/airflow`)

**Taille :** XS

**Notes (déjà fait) :**
- Image `apache/airflow:2.9.2-python3.10` + `pip install apache-airflow-providers-docker==3.9.2`
- Référencée dans `docker-compose.yml` comme `radiocovid-airflow:2.9.2`

---

## INFRA-AF-02 ✅ : Écrire le DAG ETL → Train

**Composant :** INFRA
**Fichiers concernés :**
- `dags/radiocovid_pipeline.py` — DAG avec 2 tâches `DockerOperator`

**Dépendances :** INFRA-AF-01

**Pourquoi :** C'est le fichier central du chantier — il décrit le pipeline ML que doit orchestrer Airflow.

**Critères d'acceptation :**
- [x] DAG `radiocovid_pipeline` défini avec `schedule="@weekly"`
- [x] Tâche `etl` : lance `radiocovid-etl:0.1.0` avec les bons volumes et variables d'env
- [x] Tâche `train` : lance `radiocovid-train:0.1.1` avec volumes data/models/logs + `WANDB_API_KEY`
- [x] Dépendance `etl >> train` correctement définie (train ne démarre que si ETL réussit)
- [x] `HOST_PROJECT_DIR` utilisé pour les chemins de bind mount (pas de chemin hardcodé)

**Taille :** S

**Notes (déjà fait) :**
- `DockerOperator` avec `docker_url="unix://var/run/docker.sock"` (DinD via socket)
- `shm_size=2GB` pour PyTorch sur la tâche train
- `auto_remove="force"` pour nettoyer les containers après exécution

---

## INFRA-AF-03 ✅ : Intégrer la stack Airflow dans docker-compose.yml

**Composant :** INFRA
**Fichiers concernés :**
- `docker-compose.yml` — 4 nouveaux services sous le profil `airflow`

**Dépendances :** INFRA-AF-01

**Pourquoi :** Airflow nécessite plusieurs services : une base de données (Postgres), un service d'initialisation, un serveur web et un planificateur. Tous sont déclarés dans le même `docker-compose.yml`, isolés via le profil `airflow` pour ne pas interférer avec ETL et Train.

**Critères d'acceptation :**
- [x] `airflow-postgres` : base PostgreSQL avec healthcheck
- [x] `airflow-init` : migre la DB + crée un utilisateur admin au premier démarrage
- [x] `airflow-webserver` : UI accessible sur `http://localhost:8080`
- [x] `airflow-scheduler` : planificateur qui déclenche les tâches du DAG
- [x] Socket Docker `/var/run/docker.sock` monté sur webserver et scheduler (pour `DockerOperator`)
- [x] Dossier `./dags` monté sur les services qui en ont besoin
- [x] Tous les services sont sous le profil `airflow` (ne démarrent pas par défaut)

**Taille :** S

**Notes (déjà fait) :**
- Volume nommé `airflow-postgres-data` pour persister la DB entre les redémarrages
- `AIRFLOW__CORE__LOAD_EXAMPLES: 'false'` pour ne pas polluer l'UI

---

## INFRA-AF-04 : Ajouter les targets Makefile pour Airflow

**Composant :** INFRA
**Fichiers concernés :**
- `Makefile` — nouvelles cibles `airflow-init`, `airflow-up`, `airflow-down`, `airflow-logs`

**Dépendances :** INFRA-AF-03

**Pourquoi :** Sans targets Makefile, il faut retenir les commandes `docker compose --profile airflow ...` à la main. Les autres chantiers (ETL, train) ont leurs cibles Make — Airflow doit avoir les siennes pour rester cohérent.

**Critères d'acceptation :**
- [x] `make airflow-build` : build l'image `radiocovid-airflow:2.9.2`
- [x] `make airflow-init` : lance `airflow-init` (migration DB + user admin) — à n'appeler qu'une fois
- [x] `make airflow-up` : démarre webserver + scheduler en arrière-plan
- [x] `make airflow-down` : arrête tous les services Airflow
- [x] `make airflow-logs` : affiche les logs du scheduler en temps réel
- [x] Les cibles sont documentées dans le `README.md` section Airflow

**Taille :** XS

**Notes d'implémentation (2026-07-30) :**
- Targets ajoutés dans `Makefile` ; doc README section « Airflow — orchestration »
- INFRA-AF-04 **terminé**

---

## INFRA-AF-05 : Documenter et vérifier HOST_PROJECT_DIR

**Composant :** INFRA
**Fichiers concernés :**
- `.env.example` — ajout de `HOST_PROJECT_DIR`
- `README.md` — section Airflow : expliquer pourquoi cette variable est nécessaire

**Dépendances :** INFRA-AF-03

**Pourquoi :** Le `DockerOperator` lance des containers depuis Airflow (qui tourne lui-même dans Docker). Pour que les bind mounts fonctionnent, les chemins doivent être ceux de **l'hôte**, pas du container Airflow. Si `HOST_PROJECT_DIR` n'est pas définie, les mounts échoueront silencieusement.

**Critères d'acceptation :**
- [x] `HOST_PROJECT_DIR` présent dans `.env.example` avec une valeur d'exemple et un commentaire explicatif
- [x] README explique qu'on doit définir `HOST_PROJECT_DIR=/chemin/absolu/vers/le/projet` dans `.env` avant de lancer Airflow
- [x] Le `docker-compose.yml` utilise `${HOST_PROJECT_DIR:-${PWD}}` comme fallback (déjà le cas — vérifié)

**Taille :** XS

**Notes d'implémentation (2026-07-30) :**
- `.env.example` + README ; fallback compose déjà en place sur webserver/scheduler
- INFRA-AF-05 **terminé**

---

## INFRA-AF-06 ✅ : Valider le DAG en run manuel (test end-to-end)

**Composant :** INFRA
**Fichiers concernés :**
- Aucune modification de code — validation opérationnelle

**Dépendances :** INFRA-AF-04 + INFRA-AF-05 + images `radiocovid-etl:0.1.0` et `radiocovid-train:0.1.1` buildées

**Pourquoi :** Le DAG est écrit mais n'a jamais été exécuté via Airflow. Il faut valider que le `DockerOperator` arrive bien à lancer les containers, que les mounts sont corrects et que la chaîne ETL → Train se termine avec succès.

**Critères d'acceptation :**
- [ ] `make airflow-init && make airflow-up` démarre sans erreur
- [ ] L'UI Airflow sur `http://localhost:8080` affiche le DAG `radiocovid_pipeline`
- [ ] Déclenchement manuel depuis l'UI → les 2 tâches passent en vert (status `success`)
- [ ] Les logs de chaque tâche dans l'UI montrent la sortie du container Docker correspondant
- [ ] `make airflow-down` arrête proprement tous les services

**Taille :** S

---

## INFRA-AF-07 : Ajouter des tests unitaires pour le DAG

**Composant :** INFRA
**Fichiers concernés :**
- `tests/test_dag_radiocovid_pipeline.py` — nouveau fichier de tests (à créer à la racine ou dans un dossier `tests/`)

**Dépendances :** INFRA-AF-06

**Pourquoi :** Les DAGs Airflow peuvent contenir des erreurs silencieuses (mauvais ordre de tâches, variable manquante, import cassé). Les tests unitaires permettent de les détecter sans lancer toute la stack.

**Critères d'acceptation :**
- [ ] Test : le DAG s'importe sans erreur (`from dags.radiocovid_pipeline import dag`)
- [ ] Test : le DAG contient exactement 2 tâches (`etl` et `train`)
- [ ] Test : la dépendance `etl >> train` est bien définie (vérifier `dag.task_dict["train"].upstream_task_ids`)
- [ ] Test : le schedule est `@weekly`
- [ ] `tox` ou `pytest tests/` passe en CI

**Taille :** S

---

## INFRA-AF-08 : (Optionnel) Déclencher le DAG via l'API Airflow après un ingest GitHub Actions

**Composant :** INFRA
**Fichiers concernés :**
- `.github/workflows/data_ingest.yml` — ajouter un step `trigger airflow dag` en fin de workflow
- `README.md` — documenter le flux complet Ingest → Airflow

**Dépendances :** INFRA-AF-06 + INFRA-04c (workflow data_ingest)

**Pourquoi :** Variante si l'ingest reste dans GitHub Actions (INFRA-04c) et qu'Airflow ne fait que ETL → Train → Promote. Si INFRA-AF-09 est retenu (ingest dans Airflow), ce ticket devient **obsolète**.

**Flux cible (variante GHA) :**
```
Drive incoming_images/
       │
GitHub Actions (data_ingest.yml)
  1. Sync Drive → local
  2. dvc add + push
  3. git tag data-vX.Y
       │
  4. POST /api/v1/dags/radiocovid_pipeline/dagRuns
       │
Airflow
  ETL → TRAIN → PROMOTE
```

**Critères d'acceptation :**
- [ ] Un step `curl` (ou action GitHub) envoie un POST à l'API Airflow après un ingest réussi
- [ ] La variable `AIRFLOW_URL` et un token d'auth sont stockés en secrets GitHub
- [ ] Le DAG se déclenche bien côté Airflow (visible dans l'UI)
- [ ] Si le trigger échoue (Airflow injoignable), le workflow CI ne bloque pas (step `continue-on-error: true`)
- [ ] README documente le flux complet

**Taille :** M

---

## INFRA-AF-09 ✅ : Orchestrer l'ingest + versionnage des données dans Airflow

**Composant :** INFRA
**Fichiers concernés :**
- `docker/ingest/Dockerfile` — image Python 3.10-slim + git + dvc[gdrive] + pydrive2
- `docker/ingest/entrypoint.sh` — configure git identity / credentials, appelle `ingest_and_version_data.py`
- `dags/radiocovid_pipeline.py` — tâche `ingest` en amont de `etl` (`ingest >> etl >> train`)
- `docker-compose.yml` — service `ingest` (build) + env vars ingest sur webserver/scheduler
- `.env.example` — ajout `GH_PAT`, `INGEST_SKIP_PUSH`, `GIT_USER_NAME`, `GIT_USER_EMAIL`
- `Makefile` — `ingest-build` + `airflow-build` inclut maintenant l'image ingest

**Dépendances :** INFRA-AF-06 + script `scripts/ingest_and_version_data.py` (déjà existant)

**Pourquoi :** Le mentor demande d'automatiser au maximum. L'ingest était déclenché via GitHub Actions (`workflow_dispatch`). Avec Airflow, le DAG orchestre toute la chaîne sans intervention manuelle.

**Flux implémenté :**
```
Airflow (@weekly ou trigger manuel)
  [1] ingest  → lance radiocovid-ingest:0.1.0 (monte le repo complet à /workspace)
               → scripts/ingest_and_version_data.py --source=local|gdrive
               → dvc add + git commit + git tag (INGEST_SKIP_PUSH=1 par défaut)
  [2] etl     → radiocovid-etl:0.1.0 (préprocessing)
  [3] train   → radiocovid-train:0.1.1 (entraînement + W&B)
```

**Variables clés :**
| Variable | Défaut | Rôle |
|---|---|---|
| `INCOMING_SOURCE` | `local` | Source des images (`local` ou `gdrive`) |
| `INGEST_SKIP_PUSH` | `1` | `1` = pas de push distant (dev local) ; `0` = push vers GitHub + DVC remote |
| `GH_PAT` | vide | PAT GitHub pour `git push` des tags (requis si `INGEST_SKIP_PUSH=0`) |
| `GIT_USER_NAME` | `airflow-bot` | Identité git pour le commit automatique |

**Critères d'acceptation :**
- [x] Une tâche `ingest` existe dans le DAG, en amont de `etl`
- [x] Image dédiée `radiocovid-ingest:0.1.0` (git + DVC + pydrive2)
- [x] Credentials GDrive / DVC via variables d'env, sans secret hardcodé
- [x] `INGEST_SKIP_PUSH=0` + `GH_PAT` → push tags + DVC remote
- [x] `.env.example` documente toutes les variables

**Taille :** M

**Notes d'implémentation (2026-08-11) :**
- INFRA-AF-09 **terminé**
- AF-08 (bridge GHA → Airflow) devient **obsolète** : l'ingest est désormais dans Airflow

---

## INFRA-AF-10 ✅ : Ajouter la promotion du meilleur modèle au DAG

**Composant :** INFRA
**Fichiers concernés :**
- `dags/radiocovid_pipeline.py` — ajouter une tâche `promote` après `train` (`etl >> train >> promote` ou `ingest >> etl >> train >> promote`)
- `scripts/register_model.py` — réutilisé tel quel (`--promote`)
- évent. petit container / image Python légère, ou BashOperator avec `uv run`
- `docker-compose.yml` — passer `WANDB_*` à la tâche promote
- `README.md` — documenter la promotion dans le DAG

**Dépendances :** INFRA-AF-06 (+ script `scripts/register_model.py` déjà existant)

**Pourquoi :** Après un train, les métriques sont dans W&B (`best_val_score`). La promotion vers `@production` se fait aujourd'hui à la main via `register_model.py`. Airflow doit l'automatiser : comparer le candidat au modèle prod et ne promouvoir que s'il est meilleur.

**Critères d'acceptation :**
- [x] Tâche `promote` après `train` dans le DAG (`ingest >> etl >> train >> promote`)
- [x] Image dédiée `radiocovid-promote:0.1.0` (Python + wandb)
- [x] Appelle `scripts/register_model.py` avec `--promote` si `PROMOTE_APPLY=1`, sinon dry-run
- [x] Si le nouveau modèle n'est pas meilleur, pas de changement de `@production` (géré par le script)
- [x] `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_REGISTRY*` disponibles pour la tâche
- [x] `PROMOTE_APPLY` documenté dans `.env.example` (défaut `0` = dry-run safe)

**Taille :** S

**Notes d'implémentation (2026-08-11) :**
- `docker/promote/Dockerfile` + `entrypoint.sh`
- Variable `PROMOTE_APPLY` : `0` → affiche comparaison sans changer l'alias ; `1` → promeut si candidat meilleur
- INFRA-AF-10 **terminé**

---

## Résumé visuel

```
INFRA-AF-01 ✅  Image Docker Airflow custom
      │
INFRA-AF-02 ✅  DAG radiocovid_pipeline (ETL >> Train)
      │
INFRA-AF-03 ✅  Stack Airflow dans docker-compose.yml
      │
    ┌─┴─────────────────────┐
    │                       │
INFRA-AF-04 ✅           INFRA-AF-05 ✅
Makefile targets       HOST_PROJECT_DIR
    │                       │
    └──────────┬────────────┘
               │
         INFRA-AF-06
         Run end-to-end (ETL → Train)
               │
         INFRA-AF-07
         Tests unitaires DAG
               │
         INFRA-AF-09
         Ingest + versionnage dans Airflow
               │
         INFRA-AF-10 ✅
         Promote meilleur modèle (@production)

    (optionnel / alternatif à AF-09)
         INFRA-AF-08
         Trigger API depuis GitHub Actions
```

---

## Tableau récap

| Ticket | Étape métier | Ce que ça fait | Statut | Taille | Durée moy. | Fichiers impactés |
|---|---|---|---|---|---|---|
| INFRA-AF-01 | Infra Airflow | Image Docker custom + provider Docker | ✅ Fait | XS | ~30 min | `docker/airflow/Dockerfile` |
| INFRA-AF-02 | Preprocessing + Train | DAG `etl >> train` via DockerOperator | ✅ Fait | S | ~2 h | `dags/radiocovid_pipeline.py` |
| INFRA-AF-03 | Infra Airflow | Stack Postgres / init / webserver / scheduler | ✅ Fait | S | ~2 h | `docker-compose.yml` |
| INFRA-AF-04 | Outillage | Targets Makefile `airflow-*` | ✅ Fait | XS | ~30 min | `Makefile`, `README.md` |
| INFRA-AF-05 | Config | Documenter `HOST_PROJECT_DIR` | ✅ Fait | XS | ~20 min | `.env.example`, `README.md` |
| INFRA-AF-06 | Validation | Run manuel end-to-end ETL→Train | ✅ Fait | S | ~2–3 h | Aucun (ops) |
| INFRA-AF-07 | Qualité | Tests unitaires du DAG | À faire | S | ~2 h | `tests/test_dag_radiocovid_pipeline.py` |
| INFRA-AF-08 | Pont GHA→Airflow | Trigger API après ingest GHA | ~~Obsolète~~ (AF-09 fait) | M | — | — |
| INFRA-AF-09 | Ingest + versionnage | Image `radiocovid-ingest` + tâche `ingest` dans le DAG | ✅ Fait | M | ~1 j | `docker/ingest/`, `dags/radiocovid_pipeline.py`, `docker-compose.yml`, `.env.example`, `Makefile` |
| INFRA-AF-10 | Promotion modèle | Tâche `promote` via `register_model.py --promote` | ✅ Fait | S | ~2–3 h | `docker/promote/`, `dags/radiocovid_pipeline.py`, `docker-compose.yml`, `.env.example` |
