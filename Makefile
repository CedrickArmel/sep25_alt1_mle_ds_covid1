SHELL:=/bin/bash

UV_VERSION ?= 0.7.13
PYTHON_VERSION ?= 3.10.16
VENV ?= covid
PYENV_GIT_TAG ?= v2.6.3

define GPUENVVARS
# Hydra debug
export HYDRA_FULL_ERROR=1

# Set environment variables for GPU
export CUDA_HOME="/usr/local/cuda"
export CUDA_VERSION="12.5.1"
export CUDA_MAJOR_VERSION="12"
export CUDA_MINOR_VERSION="5"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"

export NVARCH="x86_64"

export NVIDIA_VISIBLE_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"

export NV_CUDA_CUDART_VERSION="12.5.82-1"
export NV_CUDA_CUDART_DEV_VERSION="12.5.82-1"
export NV_CUDA_LIB_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE="cuda-nsight-compute-12-5=12.5.1-1"
export NV_NVTX_VERSION="12.5.82-1"
export NV_NVPROF_VERSION="12.5.82-1"
export NV_NVPROF_DEV_PACKAGE="cuda-nvprof-12-5=12.5.82-1"

# cuDNN
export NV_CUDNN_VERSION="9.2.1.18-1"
export NV_CUDNN_PACKAGE="libcudnn9-cuda-12=9.2.1.18-1"
export NV_CUDNN_PACKAGE_DEV="libcudnn9-dev-cuda-12=9.2.1.18-1"

# cuBLAS
export NV_LIBCUBLAS_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE="libcublas-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE_NAME="libcublas-12-5"
export NV_LIBCUBLAS_DEV_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE="libcublas-dev-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE_NAME="libcublas-dev-12-5"

# NCCL (for multi-GPU)
export NCCL_VERSION="2.22.3-1"
export NV_LIBNCCL_PACKAGE="libnccl2=2.22.3-1+cuda12.5"
export NV_LIBNCCL_PACKAGE_NAME="libnccl2"
export NV_LIBNCCL_PACKAGE_VERSION="2.22.3-1"
export NV_LIBNCCL_DEV_PACKAGE="libnccl-dev=2.22.3-1+cuda12.5"
export NV_LIBNCCL_DEV_PACKAGE_NAME="libnccl-dev"
export NV_LIBNCCL_DEV_PACKAGE_VERSION="2.22.3-1"
endef


define TPUENVVARS
# Unset environment variables that are not needed
for var in MASTER_ADDR MASTER_PORT TPU_PROCESS_ADDRESSES XRT_TPU_CONFIG; do
    unset $var
done

# Hydra debug
export HYDRA_FULL_ERROR=1

# Set environment variables for TPU
export ISTPUVM=1
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG_LEVEL=1
export TF_CPP_MIN_LOG_LEVEL=2
export TPU_ACCELERATOR_TYPE=v5litepod-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,4,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434,8435,8436,8437,8438
export TPU_SKIP_MDS_QUERY=1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
endef


define PYENVINIT
# Pyenv setup

export PYENV_ROOT="$$HOME/.pyenv"
[[ -d $$PYENV_ROOT/bin ]] && export PATH="$$PYENV_ROOT/bin:$$PATH"
eval "$$(pyenv init - bash)"
eval "$$(pyenv virtualenv-init -)"
endef

define UVALIASES
# uv aliases

alias uvadd="uv add --active"
alias uvsync="uv sync --active"
endef

export GPUENVVARS
export TPUENVVARS
export PYENVINIT
export PYENV_GIT_TAG
export UVALIASES

.PHONY: tpusetup gpusetup remove-tf uv pyenv venv reload test data-version data-ingest \
	ingest-build build-reference \
	airflow-build airflow-init airflow-up airflow-down airflow-logs

test:
	@uv sync --group test
	@pytest -q

# ---------------------------------------------------------------------------
# Ingest — build the radiocovid-ingest:0.1.0 image (used by Airflow DAG)
# ---------------------------------------------------------------------------
# Run once before make airflow-up, or after updating docker/ingest/Dockerfile.
ingest-build:
	docker compose --profile airflow build ingest

# ---------------------------------------------------------------------------
# Airflow — orchestration (profile: airflow)
# ---------------------------------------------------------------------------
# First time: make airflow-build && make airflow-init
# Then:       make airflow-up   → UI http://localhost:8080 (admin / admin)
# Set HOST_PROJECT_DIR in .env to the absolute host path of this repo.

airflow-build:
	docker compose --profile airflow build airflow-init ingest promote

airflow-init:
	docker compose --profile airflow up airflow-init

airflow-up:
	docker compose --profile airflow up -d airflow-postgres airflow-webserver airflow-scheduler

airflow-down:
	docker compose --profile airflow down

airflow-logs:
	docker compose --profile airflow logs -f airflow-scheduler

# Publish a new dataset version after changing files under data/.
# Usage: make data-version TAG=data-v1.1
data-version:
ifndef TAG
	$(error TAG is required, e.g. make data-version TAG=data-v1.1)
endif
	dvc add data/
	dvc push
	@echo ""
	@echo "✅ DVC pointer updated and pushed to the remote."
	@echo "Next (Git):"
	@echo "  git add data.dvc"
	@echo "  git commit -m \"chore: bump dataset to $(TAG)\""
	@echo "  git tag -a $(TAG) -m \"Dataset version $(TAG)\""
	@echo "  git tag -f data-latest $(TAG)"
	@echo "  git push origin HEAD $(TAG)"
	@echo "  git push --force origin data-latest"

# Ingest images from incoming/<class>/ then publish a new DVC + Git version.
# Optional: TAG=data-v1.2  SKIP_PUSH=1  (see scripts/ingest_and_version_data.py)
data-ingest:
	python scripts/ingest_and_version_data.py \
		$(if $(TAG),--tag $(TAG),) \
		$(if $(filter 1,$(SKIP_PUSH)),--skip-push,)

# ---------------------------------------------------------------------------
# Drift monitoring — build reference distribution (DRIFT-02)
# ---------------------------------------------------------------------------
# Run once after ETL to build the reference from val/test images.
# Re-run after each model promotion to keep the reference up-to-date.
# Usage:
#   make build-reference                        # uses data/train_folder/val
#   make build-reference DATA_DIR=data/train_folder/test
#   make build-reference PUSH_WANDB=1           # also upload to W&B
DATA_DIR ?= data/train_folder/val

build-reference:
	uv run python scripts/build_reference.py \
		--data-dir $(DATA_DIR) \
		--overwrite \
		$(if $(filter 1,$(PUSH_WANDB)),--push-wandb,)

tpusetup: tpuenvs remove-tf uv pyenv venv reload
gpusetup: gpuenvs remove-tf uv pyenv venv reload
cpusetup: remove-tf uv pyenv venv reload

uv:
	@echo "Installing uv, Python 🐍 package 📦 manager..."
	@grep -q '# uv setup' ~/.bashrc || echo '# uv setup' >> ~/.bashrc
	@curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh
	@grep -q 'uv generate-shell-completion bash' ~/.bashrc || echo 'eval "$$(uv generate-shell-completion bash)"' >> ~/.bashrc
	@grep -q 'uvx --generate-shell-completion bash' ~/.bashrc || echo 'eval "$$(uvx --generate-shell-completion bash)"' >> ~/.bashrc
	@grep -q '# uv aliases' ~/.bashrc || echo "$$UVALIASES" >> ~/.bashrc
	@echo "✅ uv installation completed!"

pyenv:
	@echo "Installing Pyenv, Python 🐍 version manager..."
	@curl https://pyenv.run | bash
	@grep -q 'pyenv init' ~/.bashrc || echo "$$PYENVINIT" >> ~/.bashrc
	@echo "✅ Pyenv installation completed!"

venv:
	@echo "🔄 Creating virtual environment 📦..."
	@export PYENV_ROOT="$$HOME/.pyenv" && export PATH="$$PYENV_ROOT/bin:$$PATH" && eval "$$(pyenv init --path)" && eval "$$(pyenv init -)" && \
	if ! pyenv versions --bare | grep -q "^$(PYTHON_VERSION)$$"; \
	then pyenv install $(PYTHON_VERSION); \
	else  echo "✅ Python 🐍 $(PYTHON_VERSION) is already installed!"; fi
	@export PYENV_ROOT="$$HOME/.pyenv" && export PATH="$$PYENV_ROOT/bin:$$PATH" && eval "$$(pyenv init --path)" && eval "$$(pyenv init -)" && \
	if ! pyenv virtualenvs --bare | grep -q "^$(VENV)$$"; \
	then pyenv virtualenv $(PYTHON_VERSION) $(VENV); \
	else echo "✅ Virtual environment 📦 exists already!"; fi
	@echo "✅ Virtual environment creation completed!"

remove-tf:
	@echo "🔄 Removing tensoflow package family to avoid conflicts"
	@pip uninstall tensorflow tensorflow-tpu tensorboard -y
	@echo "✅ Packages removed successfully!"

tpuenvs:
	@echo "🔄 Setting up  TPU environment variables..."
	@grep -q '# Set environment variables for TPU' ~/.bashrc || echo "$$TPUENVVARS" >> ~/.bashrc
	@echo "✅ TPU environment variables added successfully!"

gpuenvs:
	@echo "🔄 Setting up  GPU environment variables..."
	@grep -q '# Set environment variables for GPU' ~/.bashrc || echo "$$GPUENVVARS" >> ~/.bashrc
	@echo "✅ GPU environment variables added successfully!"

reload:
	@echo '⏭️ Manually run `source ~/.bashrc` for changes to be active ✅'
