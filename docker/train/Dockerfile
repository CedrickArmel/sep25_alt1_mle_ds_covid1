FROM python:3.10-slim-bullseye

# OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Large CUDA wheels (~1GB+) — raise timeout/retries to avoid flaky ReadTimeoutError
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 --retries 10 \
        torch==2.7.0+cu128 \
        torchvision==0.22.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 radiocovid-core==0.1.2 \
    && pip install --no-cache-dir --default-timeout=1000 --retries 10 "dvc>=3.66.1" "dvc-gdrive>=3.0.1"

WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["experiment=train_resnet50_binary", "datamodule.dataset.root=/workspace/data/train_folder"]
