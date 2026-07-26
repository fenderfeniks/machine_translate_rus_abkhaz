# dags/batch_analytics.py
"""DAG: Batch Analytics (LLM-as-an-Analyst)."""

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
    "resources": {
        "requests": {"cpu": "1", "memory": "4Gi"},
        "limits": {
            "cpu": "2",
            "memory": "7Gi",
            "nvidia.com/gpu": "1",
        },
    },
    "db_secret_name": "db-secrets",
}

CONFIG: dict[str, Any] = Variable.get(
    "analytics_config", default_var=DEFAULT_CONFIG, deserialize_json=True
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
    "batch_analytics_reporting",
    default_args=default_args,
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "analytics", "llm"],
) as dag:
    analyze_reviews = KubernetesPodOperator(
        task_id="run_batch_inference",
        name="analytics-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.batch_analytics"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        service_account_name="airflow-worker-sa",
        env_from=[
            k8s.V1EnvFromSource(
                config_map_ref=k8s.V1ConfigMapEnvSource(name="decoder-template-api-config")
            ),
        ],
        env_vars=[
            k8s.V1EnvVar(
                name="DB_CONN",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name=CONFIG["db_secret_name"], key="connection-string"
                    )
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )
