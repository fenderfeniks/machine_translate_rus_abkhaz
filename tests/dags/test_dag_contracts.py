import importlib

import pytest


KUBERNETES_TASKS = [
    ("dags.batch_analytics", "run_batch_inference"),
    ("dags.retrain_model_dag", "run_lora_finetuning"),
    ("dags.retrain_model_dag", "evaluate_staging_model"),
    ("dags.quality_control", "evaluate_llm"),
    ("dags.promote_to_prod", "promote_staging_to_prod"),
    ("dags.promote_to_prod", "restart_api_deployment"),
    ("dags.system_maintenance", "cleanup_logs_and_mlruns"),
]


def _is_delete_pod(task):
    """Совместимо с Airflow 2.9+ где атрибут стал приватным."""
    return getattr(task, "_is_delete_operator_pod", None) or getattr(
        task, "is_delete_operator_pod", None
    )


@pytest.mark.parametrize("module,task_id", KUBERNETES_TASKS)
def test_all_k8s_tasks_use_correct_namespace(module, task_id):
    mod = importlib.import_module(module)
    task = mod.dag.get_task(task_id)
    assert task.namespace == "ml-pipelines", (
        f"{task_id}: namespace = {task.namespace!r}, ожидалось 'ml-pipelines'"
    )


@pytest.mark.parametrize("module,task_id", KUBERNETES_TASKS)
def test_all_k8s_tasks_have_service_account(module, task_id):
    mod = importlib.import_module(module)
    task = mod.dag.get_task(task_id)
    assert task.service_account_name == "airflow-worker-sa", (
        f"{task_id}: service_account = {task.service_account_name!r}"
    )


@pytest.mark.parametrize("module,task_id", KUBERNETES_TASKS)
def test_all_k8s_tasks_delete_pod_on_success(module, task_id):
    mod = importlib.import_module(module)
    task = mod.dag.get_task(task_id)
    assert _is_delete_pod(task) is True, (
        f"{task_id}: is_delete_operator_pod = {_is_delete_pod(task)!r}"
    )


def test_retrain_kaggle_secret_present():
    import dags.retrain_model_dag as mod

    task = mod.dag.get_task("run_lora_finetuning")
    env_names = {e.name for e in task.env_vars}
    assert "KAGGLE_USERNAME" in env_names, "KAGGLE_USERNAME не прокинут в trainer pod"
    assert "KAGGLE_KEY" in env_names, "KAGGLE_KEY не прокинут в trainer pod"

    for env in task.env_vars:
        if env.name == "KAGGLE_USERNAME":
            assert env.value_from is not None, "KAGGLE_USERNAME должен браться из secretKeyRef"
            assert env.value_from.secret_key_ref.name == "decoder-template-api-secrets"
            assert env.value_from.secret_key_ref.key == "KAGGLE_USERNAME"
        if env.name == "KAGGLE_KEY":
            assert env.value_from is not None
            assert env.value_from.secret_key_ref.name == "decoder-template-api-secrets"
            assert env.value_from.secret_key_ref.key == "KAGGLE_KEY"


def test_batch_analytics_db_conn_from_secret():
    import dags.batch_analytics as mod

    task = mod.dag.get_task("run_batch_inference")
    db_env = next((e for e in task.env_vars if e.name == "DB_CONN"), None)
    assert db_env is not None, "DB_CONN не найден в env_vars"
    assert db_env.value_from is not None, (
        "DB_CONN должен браться из secretKeyRef, не передаваться как value="
    )
    assert db_env.value is None, "DB_CONN не должен быть хардкодом"


def test_promote_kubectl_image_is_pinned():
    import dags.promote_to_prod as mod

    task = mod.dag.get_task("restart_api_deployment")
    assert "latest" not in task.image, (
        f"kubectl image не должен использовать latest: {task.image!r}"
    )
    assert "kubectl" in task.image


def test_quality_control_slack_trigger_rule():
    import dags.quality_control as mod

    task = mod.dag.get_task("alert_if_drift")
    assert task.trigger_rule == "one_failed", f"Неверный trigger_rule: {task.trigger_rule!r}"
