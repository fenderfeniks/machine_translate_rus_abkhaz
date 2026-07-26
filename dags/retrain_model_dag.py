# dags/retrain_model_dag.py
"""DAG: Регулярное дообучение модели (CPT/SFT) и слияние LoRA-адаптеров."""

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
    "resources_gpu": {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "7Gi", "nvidia.com/gpu": "1"},
    },
    "resources_cpu": {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "7Gi"},
    },
    "mount_path": "/app/models",
    "pvc_name": "model-weights-pvc",
}

CONFIG: dict[str, Any] = Variable.get(
    "training_config", default_var=DEFAULT_CONFIG, deserialize_json=True
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
    "weekly_llm_finetuning",
    default_args=default_args,
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "training", "llm"],
) as dag:
    train_model_task = KubernetesPodOperator(
        task_id="run_lora_finetuning",
        name="llm-trainer-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.train"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        env_vars=[
            k8s.V1EnvVar(
                name="HUGGINGFACE_TOKEN",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="hf-secrets", key="token", optional=True
                    )
                ),
            ),
            k8s.V1EnvVar(
                name="KAGGLE_USERNAME",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="decoder-template-api-secrets", key="KAGGLE_USERNAME"
                    )
                ),
            ),
            k8s.V1EnvVar(
                name="KAGGLE_KEY",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="decoder-template-api-secrets", key="KAGGLE_KEY"
                    )
                ),
            ),
        ],
        volume_mounts=[k8s.V1VolumeMount(name="model-weights", mount_path=CONFIG["mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="model-weights",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    merge_weights_task = KubernetesPodOperator(
        task_id="merge_lora_weights",
        name="llm-merge-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.tools.merge_lora"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_cpu"]),
        volume_mounts=[k8s.V1VolumeMount(name="model-weights", mount_path=CONFIG["mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="model-weights",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    evaluate_staging_task = KubernetesPodOperator(
        task_id="evaluate_staging_model",
        name="llm-eval-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        service_account_name="airflow-worker-sa",
        cmds=["python", "-m", "src.eval"],
        arguments=[f"model.name_or_path={CONFIG['mount_path']}/staging/merged_model"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        volume_mounts=[k8s.V1VolumeMount(name="model-weights", mount_path=CONFIG["mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="model-weights",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    request_approval = SlackWebhookOperator(
        task_id="request_manual_approval",
        slack_webhook_conn_id="slack_conn",
        message="✅ Обучение и слияние весов завершено. Метрики посчитаны. Монолитная модель ждет в Staging.\n"
        "👉 Проверьте MLflow. Если качество устраивает, запустите ручной DAG `promote_to_prod`.",
    )

    train_model_task >> merge_weights_task >> evaluate_staging_task >> request_approval
