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
from docker.types import Mount

# Host path to the project root — set HOST_PROJECT_DIR in docker-compose or .env
PROJECT_DIR = os.environ.get(
    "HOST_PROJECT_DIR",
    "/home/ubuntu/projects/sep25_alt1_mle_ds_covid1",
)

with DAG(
    dag_id="radiocovid_pipeline",
    description="ETL + Training pipeline for RadioCovid",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["radiocovid", "ml"],
) as dag:

    etl = DockerOperator(
        task_id="etl",
        image="radiocovid-etl:0.1.0",
        container_name="radiocovid-etl-airflow",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(source=f"{PROJECT_DIR}/data", target="/data", type="bind"),
        ],
        environment={
            "DATA_DIR": "/data/01_raw/COVID-19_Radiography_Dataset",
            "MANIFEST_PATH": "/data/manifest.parquet",
            "TRAIN_FOLDER_DIR": "/data/train_folder",
        },
    )

    train = DockerOperator(
        task_id="train",
        image="radiocovid-train:0.1.1",
        container_name="radiocovid-train-airflow",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        shm_size=2 * 1024 * 1024 * 1024,  # 2 GB — necesario para PyTorch
        mounts=[
            Mount(source=f"{PROJECT_DIR}/data", target="/workspace/data", type="bind"),
            Mount(
                source=f"{PROJECT_DIR}/models", target="/workspace/models", type="bind"
            ),
            Mount(source=f"{PROJECT_DIR}/logs", target="/workspace/logs", type="bind"),
        ],
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
        },
    )

    etl >> train
