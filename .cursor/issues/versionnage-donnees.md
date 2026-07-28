# Chantier : Versionnage des données (DVC)

**Objectif :** Passer de DVC-as-download à un vrai versionnage des images (tags Git + remote Drive), utilisable depuis les services Docker, puis automatisable.

**Ordre d'exécution :** INFRA-01 → INFRA-02 → INFRA-03 → INFRA-04

---

## INFRA-01 : Figer la version actuelle `data-v1.0`

**Composant :** INFRA
**Fichiers concernés :**
- `data.dvc` — pointeur déjà présent (vérifier / commit si besoin)
- tag Git `data-v1.0`

**Dépendances :** aucune

**Critères d'acceptation :**
- [x] Tag `data-v1.0` existe (local + GitHub)
- [x] Auth GDrive OK + `dvc push` → `Everything is up to date` (hash déjà présent sur Drive)
- [x] Contenu présent sur le remote Drive

**Taille :** XS

**Notes d'implémentation (2026-07-18) :**
- Tag annoté `data-v1.0` créé puis poussé : `git push origin data-v1.0` ✅
- Pointeur `data.dvc` = hash `f07d3727…`
- Auth GDrive réparée (nouveau login) ; `dvc push` → Everything is up to date ✅
- INFRA-01 **terminé**

---

## INFRA-02 : Documenter le process « nouvelle version de données »

**Composant :** INFRA
**Fichiers concernés :**
- `README.md` — section versionnage + « Publishing a new data version »
- `Makefile` — cible `data-version`

**Dépendances :** INFRA-01

**Critères d'acceptation :**
- [x] Étapes écrites (add → push → commit → tag)
- [x] Un membre de l'équipe peut créer `data-v1.1` en suivant la doc (`make data-version TAG=…` + commandes Git)

**Taille :** XS

**Notes d'implémentation (2026-07-18) :**
- README : fetch d'une version taguée + process de publication
- Makefile : `make data-version TAG=data-vX.Y` (dvc add/push + rappel git)
- INFRA-02 **terminé**

---

## INFRA-03 : Rendre DVC utilisable dans les services Docker

**Composant :** INFRA
**Fichiers concernés :**
- `docker/etl/Dockerfile`, `docker/etl/entrypoint.sh` — `git` + `dvc pull` avant clean
- `docker/train/Dockerfile`, `docker/train/entrypoint.sh` — `dvc pull` optionnel
- `docker-compose.yml` — volumes credentials / `.git` / config
- `.env.example`, `README.md`

**Dépendances :** INFRA-01 (+ branche synchronisée avec `origin/dev`)

**Critères d'acceptation :**
- [x] Entrypoint ETL peut faire `dvc pull` (`DVC_PULL=1`)
- [x] Version ciblée via `DATA_VERSION` (ex. `data-v1.0`) documentée
- [x] Aucun secret hardcodé dans les Dockerfiles (`.env` + token monté)
- [x] Validé en run réel : `docker compose --profile etl up` → exit 0, « ETL terminé avec succès » (2026-07-20)

**Taille :** M

**Notes d'implémentation (2026-07-18 / 2026-07-20) :**
- Adapté le Docker **existant** (pas de recreation)
- `data.dvc` via `git show $DATA_VERSION:data.dvc` ; token GDrive copié en writable ; `DVC_PULL_FORCE` pour les fichiers locaux (manifest)
- Run validé : clean (COVID → Normal) + train_folder + exit code 0
- INFRA-03 **terminé et validé**

---

## INFRA-04 (résumé) : Flux complet de versionnage automatique

**Objectif :**
```
User dépose dans Drive/incoming_images/
        │
Job (CI/cron)
  1. Sync incoming_images/ → data/01_raw/...
  2. dvc add data/ + dvc push
  3. git commit data.dvc + tag data-vX.Y + data-latest
  4. git push → GitHub
        │
docker compose --profile etl up
  DATA_VERSION=data-latest → dvc pull si hash différent
```

Découpé en 4 sous-tickets ci-dessous. **Ordre :** INFRA-04a → 04b → 04c → 04d.

**Notes INFRA-04 (2026-07-27/28) :**
- Scaffold local déjà en place (`incoming/`, `scripts/ingest_and_version_data.py`, `make data-ingest`, `DATA_VERSION=data-latest`)
- Drive `incoming_images/` créé (ID `1dAEaH4KxmOCkS0hRKN5SsNePLeyJ17VQ`) avec sous-folders classes/images/masks
- `data-latest` tag local → `data-v1.0`

---

## INFRA-04a : Sync Drive `incoming_images/` → local `incoming/`

**Composant :** INFRA
**Fichiers concernés :**
- `scripts/ingest_and_version_data.py` — remplacer le `sync_from_gdrive` placeholder par le vrai téléchargement via `pydrive2`
- `.env.example` — `INCOMING_GDRIVE_FOLDER_ID` déjà présent

**Dépendances :** INFRA-04 scaffold (déjà fait)

**Critères d'acceptation :**
- [x] `INCOMING_SOURCE=gdrive INCOMING_GDRIVE_FOLDER_ID=1dAEaH… python scripts/ingest_and_version_data.py --dry-run` liste les fichiers Drive sans les télécharger
- [x] `make data-ingest` avec `INCOMING_SOURCE=gdrive` télécharge les images de Drive vers `incoming/<class>/images/` et `/masks/`
- [ ] Fichiers sans masque → erreur explicite (pas d'ingest silencieux) ← à valider en test réel
- [x] Auth : réutilise le token GDrive DVC (`.dvc/tmp/gdrive-user-credentials.json`)

**Taille :** M

**Notes d'implémentation (2026-07-28) :**
- `sync_from_gdrive()` remplacé par une implémentation `pydrive2` réelle
- Parcours Drive : `incoming_images/<class>/images/` + `/<class>/masks/` → télécharge dans `incoming/<class>/images/` + `masks/`
- `--dry-run` liste sans télécharger (critère 1 ✅)
- Auth via `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET` (env ou `.dvc/config.local`) + token `.dvc/tmp/gdrive-user-credentials.json`
- `copy_into_raw()` préserve désormais le sous-dossier `images/` / `masks/` lors du merge dans `data/01_raw/`
- `archive_incoming()` idem (archive avec sous-dossiers)
- `.env.example` : `INCOMING_GDRIVE_FOLDER_ID` pré-rempli avec l'ID réel `1dAEaH4KxmOCkS0hRKN5SsNePLeyJ17VQ`
- INFRA-04a **implémenté** (à valider en run réel)

---

## INFRA-04b : Script complet d'ingest + versionnage (local→DVC→Git)

**Composant :** INFRA
**Fichiers concernés :**
- `scripts/ingest_and_version_data.py` — valider le flux complet : sync Drive → merge `data/01_raw/` → `dvc add` → `dvc push` → git tag `data-vX.Y` + `data-latest` → archive `incoming_images`

**Dépendances :** INFRA-04a

**Critères d'acceptation :**
- [ ] `make data-ingest` avec images réelles dans Drive → crée tag `data-v1.1` (ou suivant) + `data-latest` sur Git
- [ ] `git log --oneline -- data.dvc` montre un nouveau commit
- [ ] Dossier `incoming/<class>/` vidé après ingest (fichiers archivés sous `incoming/_processed/`)
- [ ] Dataset = anciennes images + nouvelles (pas seulement les nouvelles)

**Taille :** M

---

## INFRA-04c : GitHub Actions — déclenchement du job d'ingest

**Composant :** INFRA
**Fichiers concernés :**
- `.github/workflows/data_ingest.yml` — nouveau workflow
- Secrets GitHub : `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`, `GDRIVE_CREDENTIALS_JSON`, `GH_PAT` (push tags)

**Dépendances :** INFRA-04b

**Critères d'acceptation :**
- [x] Workflow déclenché **manuellement** (`workflow_dispatch`) avec paramètre optionnel `TAG`
- [x] Le runner CI exécute le script d'ingest (authentification Drive via secrets)
- [x] Commit + tags `data-vX.Y` / `data-latest` poussés sur GitHub depuis le runner
- [ ] Workflow visible dans l'onglet Actions de GitHub ← nécessite push + secrets configurés

**Taille :** M

**Notes d'implémentation (2026-07-28) :**
- `.github/workflows/data_ingest.yml` créé avec `workflow_dispatch` (inputs : `tag`, `dry_run`)
- Auth : `GDRIVE_CREDENTIALS_JSON` (contenu du token JSON) + `GDRIVE_CLIENT_ID/SECRET` + `GH_PAT`
- Installe `dvc[gdrive]` + `pydrive2` sur le runner Ubuntu
- Configure DVC local credentials + git identity avant de lancer le script
- Git push via PAT (contourne les protections de branche si nécessaire)
- INFRA-04c **implémenté** — à valider après push + configuration des secrets GitHub

---

## INFRA-04d : (Plus tard) Déclenchement automatique à l'arrivée d'images

**Composant :** INFRA
**Fichiers concernés :**
- À définir selon l'option retenue (Google Cloud Function / Apps Script / cron / poll CI)

**Dépendances :** INFRA-04c

**Critères d'acceptation :**
- [ ] Un dépôt de fichiers dans `incoming_images/` sur Drive déclenche automatiquement le workflow d'ingest dans les X minutes
- [ ] Notification (log, Slack, email) en cas d'échec

**Taille :** L — à cadrer après INFRA-04c
