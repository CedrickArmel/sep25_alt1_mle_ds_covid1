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

"""DAG radiocovid_monitoring — daily covariate-shift detection.

Architecture:
    drift_check  (DockerOperator → radiocovid-drift:0.1.0)
        │
        └── trigger_retrain  (TriggerDagRunOperator, trigger_rule=all_failed)
                Only fires when drift_check exits 1, which happens only when
                RETRAIN_ON_DRIFT=1 (we pass RETRAIN_ON_DRIFT as DRIFT_FAIL_ON_DETECT
                to the container so the exit code reflects user intent).

Key env vars (set in docker-compose / .env):
    RETRAIN_ON_DRIFT      0 = passive alert only (default)  1 = trigger retrain
    DRIFT_WINDOW_DAYS     Days of predictions to analyse (default 7)
    DRIFT_MIN_SAMPLES     Min predictions needed (default 50)
    ENABLE_WANDB_LOGGING  Push drift report to W&B (default 0)
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_DIR = os.environ.get(
    "HOST_PROJECT_DIR",
    "/home/ubuntu/projects/sep25_alt1_mle_ds_covid1",
)

# When RETRAIN_ON_DRIFT=1 we also want the container to exit 1 on drift
# so that trigger_retrain (trigger_rule=all_failed) fires automatically.
_retrain_on_drift = os.environ.get("RETRAIN_ON_DRIFT", "0")

with DAG(
    dag_id="radiocovid_monitoring",
    description="Daily covariate-shift detection on inference predictions",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["radiocovid", "monitoring", "drift"],
) as dag:

    drift_check = DockerOperator(
        task_id="drift_check",
        image="radiocovid-drift:0.1.0",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=PROJECT_DIR, target="/workspace", type="bind"),
        ],
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "radiologist"),
            "ENABLE_WANDB_LOGGING": os.environ.get("ENABLE_WANDB_LOGGING", "0"),
            "INFERENCE_LOG_DIR": "/workspace/"
            + os.environ.get("INFERENCE_LOG_DIR", "data/inference_logs"),
            "DRIFT_WINDOW_DAYS": os.environ.get("DRIFT_WINDOW_DAYS", "7"),
            "DRIFT_MIN_SAMPLES": os.environ.get("DRIFT_MIN_SAMPLES", "50"),
            "DRIFT_REPORT_DIR": "/workspace/reports",
            # Exit 1 on drift only when the user wants automatic retraining.
            "DRIFT_FAIL_ON_DETECT": _retrain_on_drift,
        },
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="radiocovid_pipeline",
        wait_for_completion=False,
        # Only fires when drift_check failed (i.e. drift detected + RETRAIN_ON_DRIFT=1)
        trigger_rule="all_failed",
    )

    drift_check >> trigger_retrain
