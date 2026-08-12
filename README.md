# RadioCovid

**Detecting COVID-19 from chest X-rays with deep learning.**

---

## What this project does

Every day, radiologists around the world read thousands of chest X-rays to identify lung conditions such as COVID-19, viral pneumonia, and lung opacity. This is time-consuming and requires scarce specialist expertise.

This project provides a fully reproducible machine-learning pipeline that takes a dataset of labelled chest X-rays and automatically trains a classifier capable of distinguishing COVID-19 (and other abnormal lung findings) from healthy lungs. The goal is **not** to replace a radiologist — it is to offer a research-grade baseline that teams can build on: for academic benchmarking, for rapid prototyping of clinical-decision support tools, or simply for learning how an end-to-end medical imaging pipeline is built.

The pipeline covers three stages: cleaning raw images, preparing them for training, and running the actual model training — all driven by configuration so that experiments are reproducible and easy to share.

> **Disclaimer — research use only.** This software is not a certified medical device. It must not be used for clinical diagnosis or treatment decisions.

---

## How it works

```
Raw X-ray images
      │
      ▼
┌─────────────────────────────────────┐
│  ETL  (radiocovid-etl)              │
│  1. Remove outlier images           │
│     via Haralick texture features   │
│  2. Organise images into class      │
│     sub-folders for training        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Training  (radiocovid-core)        │
│  PyTorch Lightning · VGG-11         │
│  Focal loss · F-beta metric         │
│  Hydra config · W&B / TensorBoard   │
└──────────────────┬──────────────────┘
                   │
                   ▼
     Trained model checkpoint
     + W&B model artifact
                   │
                   ▼
┌─────────────────────────────────────┐
│  Model Registry  (W&B)              │
│  scripts/register_model.py          │
│  promote best run → @production     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Inference  (radiocovid-inference)  │
│  FastAPI · loads @production        │
│  POST /predict on chest X-rays      │
└─────────────────────────────────────┘
```

---

## Repository layout

| Path | Contents |
|---|---|
| `radiocovid-core/` | Modeling library — datamodule, VGG-11 backbone, focal loss, training loop |
| `radiocovid-etl/` | Data preparation — outlier removal (Haralick GLCM), ImageFolder builder |
| `radiocovid-inference/` | Inference package — model loading from W&B Model Registry, FastAPI server |
| `scripts/` | Operational scripts — `register_model.py` (promote best run to registry) |
| `docker/` | Dockerfiles and entrypoint scripts for each pipeline stage |
| `data/` | Raw and processed X-ray images (tracked by DVC, stored on Google Drive) |
| `models/` | Saved model checkpoints |
| `notebooks/` | Exploratory data analysis notebooks |
| `references/` | Research paper that informs the modeling choices |
| `reports/` | Generated figures and reports |
| `docker-compose.yml` | Orchestrates all containerised pipeline stages |
| `.env.example` | Template for W&B credentials and Model Registry settings (copy to `.env`) |

```text
├── data.dvc
├── LICENSE
├── Makefile
├── models
├── mypy.ini
├── notebooks
│   ├── 1_0_eda_radiography.ipynb
│   ├── 1_1_audit_dataloader_output.ipynb
│   ├── 1.0_cay_eda.ipynb
│   └── 1.0_ta_eda_.ipynb
├── pyproject.toml
├── radiocovid-app
│   ├── pyproject.toml
│   └── src
│       └── radiocovid
│           └── app
│               ├── __init__.py
│               └── app.py
├── radiocovid-core
│   ├── pyproject.toml
│   └── src
│       └── radiocovid
│           └── core
│               ├── __init__.py
│               ├── configs
│               ├── data
│               ├── losses
│               ├── models
│               ├── train.py
│               └── utils
├── radiocovid-etl
│   ├── pyproject.toml
│   └── src
│       └── radiocovid
│           └── etl
│               ├── __init__.py
│               ├── clean.py
│               ├── configs
│               ├── preprocessings.py
│               ├── train_folder.py
│               └── utils.py
├── README.md
├── references
│   └── 2208.02046v1.pdf
├── reports
│   └── figures
├── tox.ini
└── uv.lock
```

---

## For ML engineers — vanilla run

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.10 | [python.org](https://www.python.org/downloads/) or `make cpusetup` (see below) |
| uv | latest | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| DVC | bundled via uv | — |
| Google Drive access | — | Ask a project maintainer to share the DVC remote with you |

> **Host bootstrap (optional):** The `Makefile` provides `make cpusetup`, `make gpusetup`, and `make tpusetup` targets that install `pyenv`, `uv`, and the relevant environment variables for CPU, GPU (CUDA), or TPU runs respectively. You only need these if you are setting up a fresh machine.

### Dev container (recommended)

Use a dev container when you want the **same Linux environment** on Windows, macOS, or Linux without installing Python and system libraries by hand. Your project files stay on your machine; only the tools (Python, `uv`, OpenCV libs, etc.) run inside Docker.

| Requirement | Notes |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Must be running before opening the container |
| **Dev Containers** extension | **Cursor:** install **Dev Containers** by **Anysphere** (`anysphere.remote-containers`). **VS Code:** install **Dev Containers** by Microsoft. Do **not** rely on **Container Tools** alone — it does not open the project in a dev container. |

**First-time setup:**

1. Clone this repo and open the **repository root** in Cursor or VS Code.
2. Command Palette → **Dev Containers: Rebuild and Reopen in Container** (first run only; later use *Reopen in Container*).
3. Wait for the post-create step to finish (`uv sync --all-groups` and `pre-commit install`). The first build can take **10–20 minutes** (PyTorch and dependencies).
4. Optional: set `WANDB_API_KEY` in your shell environment if you need online Weights & Biases logging. The container defaults to `WANDB_MODE=offline`.

**Quick check** (inside the container terminal):

```shell
python --version          # Python 3.10.x
uv run python -c "import torch; print(torch.__version__)"
uv run pytest radiocovid-core/tests/test_imports.py -q
```

Configuration lives in `.devcontainer/` (`Dockerfile` + `devcontainer.json`). **Git** (`commit`, `push`) works the same as on your host: the repo is mounted into the container, not copied.

To leave the container: **Dev Containers: Reopen Folder Locally**.

---

### Step 1 — Clone and install

```shell
git clone <repo-url>
cd sep25_alt1_mle_ds_covid1

uv sync --group dev
```

`uv sync` reads the lockfile and creates a `.venv` with every dependency pinned. All three CLI commands (`radiocovid-clean`, `radiocovid-train-folder`, `radiocovid-train`) are available inside that environment.

---

### Step 2 — Fetch the data

The dataset lives on Google Drive and is version-controlled with DVC.

- **Google Drive (via DVC)** stores the heavy image files.
- **Git** stores `data.dvc` (a small pointer + hash) and optional tags such as `data-v1.0`.

```shell
dvc fetch
# or, to also check out files into ./data :
dvc pull
```

**Fetch a specific data version** (after cloning):

```shell
git checkout data-v1.0
dvc pull
```

**First-time Google Drive setup:** If this is your first connection to the remote, you need a `client_id` and `client_secret` from the project's Google Cloud project. Follow the [DVC GDrive setup guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended), then run:

```shell
dvc remote modify --local data gdrive_client_id     [YOUR-CLIENT-ID]
dvc remote modify --local data gdrive_client_secret [YOUR-CLIENT-ID-SECRET]
dvc remote modify --local data gdrive_user_credentials_file .dvc/tmp/gdrive-user-credentials.json
dvc fetch
```

A browser window will open for Google login. On success, DVC writes a local token file (gitignored) under `.dvc/tmp/`.

---

### Publishing a new data version

Do this only when the contents of `data/` change (new/removed/corrected images).

```
modify data/  →  dvc add  →  dvc push  →  git commit data.dvc  →  git tag  →  git push
```

1. Update images under `data/`.
2. Register the new hash and upload to Drive:

   ```shell
   dvc add data/
   dvc push
   ```

3. Commit the pointer and tag the version (example: `data-v1.1`):

   ```shell
   git add data.dvc
   git commit -m "chore: bump dataset to data-v1.1"
   git tag -a data-v1.1 -m "Dataset version data-v1.1"
   git push origin HEAD data-v1.1
   ```

Helper (same DVC steps, then prints the git commands):

```shell
make data-version TAG=data-v1.1
```

Current baseline immutable tag: **`data-v1.0`**.
Floating pointer used by Docker by default: **`data-latest`** (updated whenever you ingest new images).

---

### Ingest new images automatically (local incoming folder)

Until the shared Drive upload folder is available, drop new images under `incoming/<class>/` then run:

```shell
make data-ingest
# dry-run: python scripts/ingest_and_version_data.py --dry-run
```

This copies files into `data/01_raw/COVID-19_Radiography_Dataset/`, runs DVC add/push, creates `data-vX.Y`, and moves the floating tag `data-latest`. See `incoming/README.md`.

When you have the Drive **incoming** folder id, set `INCOMING_SOURCE=gdrive` and `INCOMING_GDRIVE_FOLDER_ID=...` in `.env` — the download step will be wired to the same script.

---

### Step 3 — Clean the data

Remove texture-based outliers from the raw images and produce a manifest file:

```shell
uv run radiocovid-clean \
  data_dir=./data/01_raw \
  'folders=[COVID,Lung_Opacity,Normal,"Viral Pneumonia"]' \
  clean.dmax=29 \
  clean.output=./data/manifest.parquet \
  'clean.features=[contrast]'
```

This reads the four class folders under `./data/01_raw/`, filters images using the Haralick contrast feature (dropping those with a score above `dmax=29`), and writes a parquet manifest that maps every kept image to its class, file path, and mask path.

---

### Step 4 — Build the training folder

Create the class sub-folder structure that PyTorch's `ImageFolder` loader expects:

```shell
uv run radiocovid-train-folder \
  symlink.manifest_path=./data/manifest.parquet \
  symlink.dst_dir=./data/train_folder \
  'symlink.classes={COVID: 1, Lung_Opacity: 1, Normal: 0, "Viral Pneumonia": 1}'
```

Images are symlinked (not copied) into `./data/train_folder/0/` and `./data/train_folder/1/` according to the mapping — here the task is framed as binary: **1 = abnormal lung** (COVID, Lung Opacity, Viral Pneumonia), **0 = healthy lung** (Normal). Adjust `symlink.classes` to change the class grouping.

---

### Step 5 — Smoke test (one mini-batch)

Verify the full pipeline runs end-to-end before committing to a long training run:

```shell
uv run radiocovid-train \
  debug=fast_dev_run \
  datamodule.dataset.root=./data/train_folder
```

A successful run prints a Lightning progress bar and exits without error.

---

### Step 6 — Full training run

```shell
uv run radiocovid-train \
  datamodule.dataset.root=./data/train_folder
```

Checkpoints are saved to `models/` and logs to `logs/`. To log to Weights & Biases or TensorBoard, append `loggers=wandb` or `loggers=tensorboard`.

---

### Step 7 — Exploring configuration

Every command is powered by [Hydra](https://hydra.cc). Run `--help` to see all available options:

```shell
uv run radiocovid-train --help
```

Example output:

```text
train is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

callbacks: early_stopping, model_checkpoint, model_summary, multiple, rich_progress_bar
debug: barebones, default, fast_dev_run, limit, overfit
experiment: default
loggers: multiple, tensorboard, wandb
module: default
module/loss: focal_loss
module/metric: fbeta_score
module/optimizer: adamw, base_optimizer, sgd
module/scheduler: cosine, cosine_wr, linear, multistep, sequential
profiler: advanced, pytorch, simple, xla
strategy: auto, ddp, tpu
tuner: optuna


== Config ==
Override anything in the config (foo.bar=value)

paths:
  root_dir: ${hydra:runtime.cwd}
  data_dir: null
  log_dir: ${paths.root_dir}/logs/
  output_dir: ${hydra:runtime.output_dir}
  work_dir: ${hydra:runtime.cwd}
extras:
  ignore_warnings: false
  enforce_tags: true
  print_config: true
module:
  _target_: radiocovid.core.RadioCovidModule
  net:
    _target_: torchvision.models.vgg11
    num_classes: 2
    init_weights: true
    dropout: 0.2
  trainable_layers:
    classifier: null
  priors:
  - 0.5004
  - 0.4996
```

Key config groups:

| Group | Notable options |
|---|---|
| `debug` | `fast_dev_run`, `limit`, `overfit` |
| `loggers` | `tensorboard`, `wandb`, `multiple` |
| `strategy` | `auto`, `ddp`, `tpu` |
| `module/optimizer` | `adamw`, `sgd` |
| `module/scheduler` | `cosine`, `cosine_wr`, `linear` |
| `tuner` | `optuna` (hyperparameter search) |
| `callbacks` | `early_stopping`, `model_checkpoint` |

Override any value with `foo.bar=value`. For repeatable experiments, create a YAML file under `radiocovid-core/src/radiocovid/core/configs/experiment/` and load it with `experiment=your_experiment_name`.

---

## Running with Docker

The pipeline can be run without installing Python or any dependencies locally — everything runs inside Docker containers orchestrated by Docker Compose.

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | latest | Must be running |

### Pipeline stages

| Compose profile | What it runs | Default model |
|---|---|---|
| `etl` | `radiocovid-clean` → `radiocovid-train-folder` | — |
| `train` | `radiocovid-train` | ResNet50 binary |
| `inference` | FastAPI server on port 8000 | W&B Model Registry `@production` |
| `airflow` | Airflow UI + scheduler (orchestrates ETL → Train) | — |

### Step 1 — Fetch the data (optional on the host)

You can either:

- **Pull on the host** (classic): see [Step 2](#step-2--fetch-the-data) above, then run containers with `DVC_PULL=0`, or
- **Pull inside the ETL container** (default): set `GDRIVE_CLIENT_*` in `.env`, keep `DVC_PULL=1`. Default `DATA_VERSION=data-latest` resolves the newest dataset tag before cleaning.

Prerequisite for in-container pull: a one-time interactive `dvc pull`/`dvc push` on the host so `.dvc/tmp/gdrive-user-credentials.json` exists (browser login). That file is **mounted** into the container — it is never baked into the image.

### Step 2 — Configure W&B

Copy the environment template and fill in your personal values:

```shell
cp .env.example .env
```

| Variable | Required for | Notes |
|---|---|---|
| `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET` | ETL/train with `DVC_PULL=1` | Same values as `.dvc/config.local` |
| `DATA_VERSION` | ETL/train with `DVC_PULL=1` | Default `data-latest` (or pin `data-v1.0`, …) |
| `DVC_PULL` | ETL/train | `1` = pull inside container; `0` = use mounted `./data` |
| `WANDB_API_KEY` | Training (online), inference | Leave empty for offline training only |
| `WANDB_ENTITY` | Training (online), inference, promotion | Your W&B team or user slug |
| `WANDB_PROJECT` | Training, promotion | Default `radiologist` (Hydra project name) |
| `WANDB_REGISTRY*` | Inference | Model Registry name, collection, alias — see `.env.example` |
| `HOST_PROJECT_DIR` | Airflow | Absolute host path to this repo (see [Airflow](#airflow--orchestration)) |

The `.env` file is git-ignored — each developer keeps their own locally. Docker Compose reads it automatically for variable substitution.

### Step 3 — Run the ETL

```shell
docker compose --profile etl up
```

This will:
1. Build the `radiocovid-etl:0.1.0` image on first run (subsequent runs reuse the cached image)
2. If `DVC_PULL=1`: resolve `data.dvc` from `DATA_VERSION` and `dvc pull` into `./data`
3. Clean the raw images from `./data/01_raw/` and write `./data/manifest.parquet`
4. Build the training folder structure under `./data/train_folder/`

The container exits automatically when both steps complete. Your `./data/` folder is mounted into the container — outputs are written directly to your machine.

### Step 4 — Run the training

```shell
docker compose --profile train up
```

This trains a ResNet50 binary classifier by default. Checkpoints are saved to `./models/` and logs to `./logs/`.

To override any Hydra config parameter:

```shell
# Different experiment
docker compose --profile train run train experiment=train_resnet50_binary_2

# Custom hyperparameters
docker compose --profile train run train \
  module/optimizer=sgd \
  module/scheduler=cosine \
  trainer.max_epochs=50
```

### Step 5 — Run inference

The inference container serves a REST API that loads the model tagged `@production` in the W&B Model Registry. It installs `radiocovid-inference` from the local source tree (no PyPI publish required when you change inference code).

**Prerequisites:** `WANDB_API_KEY` and `WANDB_ENTITY` must be set in `.env`. The registry collection `Radiocovid-classifier/radiocovid-classifier` must already exist in your W&B entity.

```shell
docker compose --profile inference build
docker compose --profile inference up
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/info` | GET | Loaded model metadata (run id, registry alias, source artifact) |
| `/predict` | POST | Upload a chest X-ray image (`multipart/form-data`, field `file`) |
| `/reload` | POST | Re-download the model from the registry (after promotion) |

**Quick test** (with the server running):

```shell
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict -F "file=@path/to/xray.png"
```

After promoting a new model (see [Model Registry](#model-registry--promoting-a-model)), reload without restarting the container:

```shell
curl -X POST http://localhost:8000/reload
```

**Registry download fallback:** If the W&B registry artifact cannot be downloaded (permissions), the inference service automatically falls back to the source training artifact in `WANDB_REGISTRY_SOURCE_PROJECT` (default `radiocovid`). Set this in `.env` if your production model lives in a different source project.

**Override the registry path** (optional): set `WANDB_REGISTRY_ARTIFACT` in `.env` to the full path copied from the W&B registry Usage tab.

### Rebuilding an image after a package update

```shell
docker compose --profile etl build && docker compose --profile etl up
docker compose --profile train build && docker compose --profile train up
docker compose --profile inference build && docker compose --profile inference up
```

### Airflow — orchestration

Airflow runs the weekly DAG `radiocovid_pipeline` (`dags/radiocovid_pipeline.py`): **ETL → Train**, each step via `DockerOperator` (images `radiocovid-etl` / `radiocovid-train`).

**Prerequisite — `HOST_PROJECT_DIR`:** Airflow itself runs in Docker, then starts ETL/train containers on the host Docker daemon. Bind mounts must use the **host** absolute path of this repository (not a path inside the Airflow container). Set it in `.env` before starting Airflow:

```shell
# Linux / macOS
HOST_PROJECT_DIR=/absolute/path/to/sep25_alt1_mle_ds_covid1

# Windows (forward slashes are fine)
HOST_PROJECT_DIR=C:/Projects/Projet MLops/sep25_alt1_mle_ds_covid1
```

If unset, `docker-compose.yml` falls back to `${HOST_PROJECT_DIR:-${PWD}}`, which is often incorrect when Compose runs from Docker Desktop / a non-project PWD.

Also build the ETL and train images once (`docker compose --profile etl build` and `--profile train build`) so the DAG can pull them by tag.

| Make target | What it does |
|---|---|
| `make airflow-build` | Build image `radiocovid-airflow:2.9.2` |
| `make airflow-init` | One-shot: migrate Airflow DB + create user `admin` / `admin` |
| `make airflow-up` | Start Postgres + webserver + scheduler in the background |
| `make airflow-down` | Stop all Airflow profile services |
| `make airflow-logs` | Follow scheduler logs |

```shell
# First time
make airflow-build
make airflow-init

# Day to day
make airflow-up
# UI → http://localhost:8080  (admin / admin)
make airflow-logs
make airflow-down
```

---

## Model Registry — promoting a model

After training, each run with `log_model=True` produces a **model artifact** in W&B. The promotion script picks the best run (by `best_val_score`, i.e. validation F1 macro) and links it to the existing Model Registry collection.

**Script location:** `scripts/register_model.py` (run from the repository root).

### Workflow

```
Training runs (WANDB_PROJECT=radiologist)
        │
        ▼
scripts/register_model.py          ← dry-run: compare candidate vs @production
        │
        ▼  --promote
W&B Model Registry  @production
        │
        ▼
Inference container  (docker compose --profile inference)
        │
        ▼  POST /reload
Serving the new model
```

### Usage

```shell
# 1. Configure .env (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_REGISTRY*)
cp .env.example .env

# 2. Dry-run — shows current production vs best candidate, no W&B changes
uv run --with wandb python scripts/register_model.py

# 3. Apply promotion — links artifact and moves the @production alias
uv run --with wandb python scripts/register_model.py --promote

# 4. Reload the inference API (if already running)
curl -X POST http://localhost:8000/reload
```

The script is **dry-run by default**. It compares the candidate's `best_val_score` against the current production model and warns if promotion would be a downgrade. It skips promotion when the same source artifact is already in production.

**Registry variables** (in `.env`):

| Variable | Default | Role |
|---|---|---|
| `WANDB_REGISTRY` | `Radiocovid-classifier` | Registry name (without `wandb-registry-` prefix) |
| `WANDB_REGISTRY_MODEL` | `radiocovid-classifier` | Collection name inside the registry |
| `WANDB_REGISTRY_ALIAS` | `production` | Alias to update on promotion and to serve at inference |

---

## Hardware

| Target | Install | Notes |
|---|---|---|
| CPU | `uv sync --group dev` (default) | Works out of the box |
| GPU (CUDA) | `make gpusetup`, then `uv sync --extra gpu` | Sets CUDA env vars |
| TPU | `make tpusetup`, then `uv sync --extra xla` | Installs `torch-xla` + `libtpu`; use `strategy=tpu` |

---

## Development

```shell
uv run pre-commit install   # install git hooks (linting, formatting)
uv run tox                  # run the full test and lint suite
uv run mypy .               # type-check (config: mypy.ini)
```

---

## References & credits

- **Paper:** [arXiv 2208.02046](references/2208.02046v1.pdf) — the research work that motivates the modeling choices in this project.
- **Dataset:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle).
- **Authors:** [@CedrickArmel](https://github.com/CedrickArmel), [@samarita22](https://github.com/samarita22), [@TaxelleT](https://github.com/TaxelleT), [@Yeyecodes](https://github.com/Yeyecodes).
- **License:** MIT — see [LICENSE](LICENSE).
