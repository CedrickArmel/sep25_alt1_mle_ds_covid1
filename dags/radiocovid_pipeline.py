# MIT License
#
# Copyright (c) 2025 @CedrickArmel, @samarita22, @TaxelleT & @Yeyecodes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

# Host path to the project root — set HOST_PROJECT_DIR in docker-compose or .env
PROJECT_DIR = os.environ.get(
    "HOST_PROJECT_DIR",
    "/home/ubuntu/projects/sep25_alt1_mle_ds_covid1",
)

with DAG(
    dag_id="radiocovid_pipeline",
    description="Ingest + ETL + Training pipeline for RadioCovid",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["radiocovid", "ml"],
) as dag:

    ingest = DockerOperator(
        task_id="ingest",
        image="radiocovid-ingest:0.1.0",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        mounts=[
            # Full repo mount: script needs .git/, .dvc/, data/, incoming/, scripts/
            Mount(source=PROJECT_DIR, target="/workspace", type="bind"),
        ],
        environment={
            "INCOMING_SOURCE": os.environ.get("INCOMING_SOURCE", "local"),
            "INCOMING_GDRIVE_FOLDER_ID": os.environ.get(
                "INCOMING_GDRIVE_FOLDER_ID", ""
            ),
            "GDRIVE_CLIENT_ID": os.environ.get("GDRIVE_CLIENT_ID", ""),
            "GDRIVE_CLIENT_SECRET": os.environ.get("GDRIVE_CLIENT_SECRET", ""),
            # Credentials file path inside the container (mounted via full repo)
            "DVC_GDRIVE_CREDENTIALS_PATH": "/workspace/.dvc/tmp/gdrive-user-credentials.json",
            # Git identity for the automated commit created by publish_version()
            "GIT_USER_NAME": os.environ.get("GIT_USER_NAME", "airflow-bot"),
            "GIT_USER_EMAIL": os.environ.get("GIT_USER_EMAIL", "airflow@localhost"),
            # GitHub PAT for git push / dvc push to remote. Leave empty to skip push.
            "GH_PAT": os.environ.get("GH_PAT", ""),
            # 1 = skip dvc push + git push (safe default for local dev).
            # Set to 0 + provide GH_PAT to publish tags to GitHub.
            "INGEST_SKIP_PUSH": os.environ.get("INGEST_SKIP_PUSH", "1"),
        },
    )

    etl = DockerOperator(
        task_id="etl",
        image="radiocovid-etl:0.1.0",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,  # required with docker.sock (DinD): host can't see /tmp inside Airflow
        mounts=[
            Mount(source=f"{PROJECT_DIR}/data", target="/data", type="bind"),
        ],
        environment={
            "DATA_DIR": "/data/01_raw/COVID-19_Radiography_Dataset",
            "MANIFEST_PATH": "/data/manifest.parquet",
            "TRAIN_FOLDER_DIR": "/data/train_folder",
            # Data already on host for orchestration tests; avoid Drive auth inside this task
            "DVC_PULL": "0",
            # Full ETL by default. Set SKIP_CLEAN=1 in .env only to reuse an existing
            # manifest after a partial failure (e.g. train_folder step crashed).
            "SKIP_CLEAN": os.environ.get("SKIP_CLEAN", "0"),
        },
    )

    train = DockerOperator(
        task_id="train",
        image="radiocovid-train:0.1.1",
        command=[
            "experiment=train_resnet50_binary",
            "datamodule.dataset.root=/workspace/data/train_folder",
        ],
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        shm_size=2 * 1024 * 1024 * 1024,  # 2 GB — necesario para PyTorch
        device_requests=[
            DeviceRequest(count=1, capabilities=[["gpu"]]),
        ],
        mounts=[
            # Symlinks in train_folder point to absolute /data/... paths (created by ETL).
            # Keep both mounts, same as docker-compose train service.
            Mount(source=f"{PROJECT_DIR}/data", target="/data", type="bind"),
            Mount(source=f"{PROJECT_DIR}/data", target="/workspace/data", type="bind"),
            Mount(
                source=f"{PROJECT_DIR}/models", target="/workspace/models", type="bind"
            ),
            Mount(source=f"{PROJECT_DIR}/logs", target="/workspace/logs", type="bind"),
        ],
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
            "DVC_PULL": "0",
        },
    )

    promote = DockerOperator(
        task_id="promote",
        image="radiocovid-promote:0.1.0",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        mounts=[
            # scripts/register_model.py must be accessible at /workspace/scripts/
            Mount(source=PROJECT_DIR, target="/workspace", type="bind"),
        ],
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "radiologist"),
            "WANDB_REGISTRY": os.environ.get("WANDB_REGISTRY", "Radiocovid-classifier"),
            "WANDB_REGISTRY_MODEL": os.environ.get(
                "WANDB_REGISTRY_MODEL", "radiocovid-classifier"
            ),
            "WANDB_REGISTRY_ALIAS": os.environ.get(
                "WANDB_REGISTRY_ALIAS", "production"
            ),
            # 0 = dry-run (compare only, no change). 1 = move @production alias if candidate is better.
            "PROMOTE_APPLY": os.environ.get("PROMOTE_APPLY", "0"),
        },
    )

    ingest >> etl >> train >> promote
