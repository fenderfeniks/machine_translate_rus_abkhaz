# dags/system_maintenance.py
"""DAG: System Maintenance (Cleanup).

Очистка старых логов и артефактов MLflow.
"""

from typing import Any

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


IMAGE: str = Variable.get("PROJECT_IMAGE", default_var="my-company/decoder_template:trainer-latest")
NAMESPACE: str = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

DEFAULT_CONFIG: dict[str, Any] = {
    "schedule": "@daily",
    "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    "resources": {"requests": {"cpu": "0.5", "memory": "1Gi"}},
    "logs_mount_path": "/app/logs",
    "logs_pvc_name": "logs-pvc",
    "retention_days": 30,
}

CONFIG: dict[str, Any] = Variable.get(
    "maintenance_config", default_var=DEFAULT_CONFIG, deserialize_json=True
)

default_args: dict[str, Any] = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
    "email_on_failure": True,
    "retries": CONFIG["default_args"]["retries"],
    "retry_delay": pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    "system_maintenance",
    default_args=default_args,
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["maintenance", "nlp", "llm"],
) as dag:
    cleanup_logs = KubernetesPodOperator(
        task_id="cleanup_logs_and_mlruns",
        name="cleanup-logs-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        service_account_name="airflow-worker-sa",
        cmds=[
            "python",
            "-m",
            "src.jobs.maintenance",
            "--action",
            "cleanup",
            "--days",
            str(CONFIG["retention_days"]),
        ],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_vars=[
            k8s.V1EnvVar(name="LOGS_DIR", value=CONFIG["logs_mount_path"]),
        ],
        volume_mounts=[k8s.V1VolumeMount(name="logs-data", mount_path=CONFIG["logs_mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="logs-data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["logs_pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )
