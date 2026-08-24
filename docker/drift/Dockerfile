FROM python:3.10-slim-bullseye

# Drift-only deps — torch/torchvision/fastapi not needed here
RUN pip install --no-cache-dir \
    "evidently>=0.4" \
    "pandas>=1.5" \
    "wandb>=0.19.11" \
    "numpy>=1.24" \
    "pillow>=10.0" \
    "python-dotenv>=1.0"

# Install radiocovid-inference without its heavy deps (torch, fastapi…)
# Only drift_check.py and inference_logger.py are used at runtime.
COPY radiocovid-inference /pkg/radiocovid-inference
RUN pip install --no-cache-dir --no-deps /pkg/radiocovid-inference

COPY docker/drift/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
