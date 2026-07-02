FROM python:3.10-slim-bullseye

RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi==0.136.3 \
    uvicorn[standard]==0.34.3 \
    pillow==12.1.1 \
    python-dotenv==1.1.0 \
    python-multipart==0.0.29 \
    wandb==0.19.11

WORKDIR /app

COPY radiocovid-inference/api.py .
COPY radiocovid-inference/predict.py .
COPY radiocovid-inference/wandb_download_ckpt.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
