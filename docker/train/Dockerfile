FROM python:3.10-slim-bullseye

# OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir radiocovid-core==0.1.2

WORKDIR /workspace

ENTRYPOINT ["radiocovid-train"]
CMD ["experiment=train_resnet50_binary", "datamodule.dataset.root=/workspace/data/train_folder"]
