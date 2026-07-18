FROM python:3.10-slim-bullseye

RUN pip install --no-cache-dir \
    torch==2.6.0+cpu torchvision==0.21.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY radiocovid-inference /pkg/radiocovid-inference
RUN pip install --no-cache-dir /pkg/radiocovid-inference omegaconf

WORKDIR /app

COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
