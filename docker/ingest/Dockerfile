FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# dvc[gdrive] pulls pydrive2 transitively; pin for reproducibility
# pathspec 1.x removed _DIR_MARK used by scmrepo — pin to last compatible version
RUN pip install --no-cache-dir "dvc[gdrive]==3.51.2" "pydrive2==1.20.0" "pathspec==0.11.2"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
