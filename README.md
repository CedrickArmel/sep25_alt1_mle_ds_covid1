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
     + metrics & training logs
```

---

## Repository layout

| Path | Contents |
|---|---|
| `radiocovid-core/` | Modeling library — datamodule, VGG-11 backbone, focal loss, training loop |
| `radiocovid-etl/` | Data preparation — outlier removal (Haralick GLCM), ImageFolder builder |
| `docker/` | Dockerfiles and entrypoint scripts for each pipeline stage |
| `data/` | Raw and processed X-ray images (tracked by DVC, stored on Google Drive) |
| `models/` | Saved model checkpoints |
| `notebooks/` | Exploratory data analysis notebooks |
| `references/` | Research paper that informs the modeling choices |
| `reports/` | Generated figures and reports |
| `docker-compose.yml` | Orchestrates all containerised pipeline stages |

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

```shell
dvc fetch
```

**First-time Google Drive setup:** If this is your first connection to the remote, you need a `client_id` and `client_secret` from the project's Google Cloud project. Follow the [DVC GDrive setup guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended), then run:

```shell
dvc remote modify --local data gdrive_client_id     [YOUR-CLIENT-ID]
dvc remote modify --local data gdrive_client_secret [YOUR-CLIENT-ID-SECRET]
dvc fetch
```

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

### Step 1 — Fetch the data

The dataset must be fetched with DVC before running any container (see [Step 2](#step-2--fetch-the-data) in the vanilla run section above).

### Step 2 — Configure W&B (optional)

Training logs metrics to Weights & Biases. By default it runs in **offline mode** (no account needed). To enable online logging:

```shell
cp .env.example .env
# Edit .env and paste your W&B API key
```

The `.env` file is git-ignored — each developer keeps their own locally.

### Step 3 — Run the ETL

```shell
docker compose --profile etl up
```

This will:
1. Build the `radiocovid-etl:0.1.0` image on first run (subsequent runs reuse the cached image)
2. Clean the raw images from `./data/01_raw/` and write `./data/manifest.parquet`
3. Build the training folder structure under `./data/train_folder/`

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

### Rebuilding an image after a package update

```shell
docker compose --profile etl build && docker compose --profile etl up
docker compose --profile train build && docker compose --profile train up
```

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
