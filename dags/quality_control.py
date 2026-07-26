# dags/quality_control.py
"""DAG: Model Evaluation & Quality Drift Detection (Generative LLM)."""

from typing import Any

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from kubernetes.client import models as k8s


IMAGE: str = Variable.get("PROJECT_IMAGE", default_var="my-company/decoder_template:trainer-latest")
NAMESPACE: str = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

DEFAULT_CONFIG: dict[str, Any] = {
    "schedule": "@weekly",
    "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    "resources": {
        "requests": {"cpu": "1", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "7Gi", "nvidia.com/gpu": "1"},
    },
    "rouge1_threshold": 0.45,
}

CONFIG: dict[str, Any] = Variable.get(
    "evaluation_config", default_var=DEFAULT_CONFIG, deserialize_json=True
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
    "llm_quality_drift_detection",
    default_args=default_args,
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "monitoring", "llm"],
) as dag:
    evaluate_model = KubernetesPodOperator(
        task_id="evaluate_llm",
        name="evaluator-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.eval"],
        arguments=[
            f"metric_thresholds.rouge1={CONFIG['rouge1_threshold']}",
            "model.builder.mlflow_alias=Staging",
        ],
        env_from=[
            k8s.V1EnvFromSource(
                config_map_ref=k8s.V1ConfigMapEnvSource(name="decoder-template-api-config")
            ),
        ],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        get_logs=True,
        is_delete_operator_pod=True,
    )

    threshold_percent = int(CONFIG["rouge1_threshold"] * 100)

    notify_slack = SlackWebhookOperator(
        task_id="alert_if_drift",
        slack_webhook_conn_id="slack_conn",
        message=f"Внимание! Качество генерации (ROUGE-1) упало ниже порога {threshold_percent}%. Требуется анализ данных и дообучение.",
        trigger_rule="one_failed",
    )

    evaluate_model >> notify_slack
