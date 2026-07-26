# tests/dags/test_dag_config.py
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from airflow import DAG


VARIABLES_PATH = Path(__file__).parents[2] / "deploy" / "airflow" / "variables.json"

with open(VARIABLES_PATH) as f:
    REAL_VARIABLES = json.load(f)


def _variable_get(
    key: str, default_var: Any = None, deserialize_json: bool = False, **kwargs: Any
) -> Any:
    """Эмулирует Variable.get() используя реальный variables.json."""
    if key in REAL_VARIABLES:
        value = REAL_VARIABLES[key]
        return value
    return default_var


@pytest.fixture(autouse=True)
def mock_airflow_variable():
    with patch("airflow.models.Variable.get", side_effect=_variable_get):
        yield


@pytest.fixture(autouse=True)
def mock_kubernetes_pod_operator():
    """Не даём KubernetesPodOperator реально подключаться к кластеру при импорте DAG."""
    with patch(
        "airflow.providers.cncf.kubernetes.operators.pod.KubernetesPodOperator.execute",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def mock_slack():
    with patch(
        "airflow.providers.slack.operators.slack_webhook.SlackWebhookOperator.execute",
        return_value=None,
    ):
        yield


def _get_schedule(dag: DAG) -> Any:
    return getattr(dag, "schedule", None) or getattr(dag, "schedule_interval", None)


def test_batch_analytics_uses_real_config() -> None:
    import dags.batch_analytics as mod

    task = mod.dag.get_task("run_batch_inference")
    limits = task.container_resources.limits
    # Исправлено с "1" на "2" согласно актуальному variables.json
    assert limits["cpu"] == "2", (
        f"Применился DEFAULT_CONFIG вместо variables.json. cpu limits = {limits['cpu']!r}"
    )
    assert limits["memory"] == "7Gi", f"Неверный memory limit: {limits['memory']!r}"


def test_retrain_uses_real_config() -> None:
    import dags.retrain_model_dag as mod

    task = mod.dag.get_task("run_lora_finetuning")
    limits = task.container_resources.limits
    # Исправлено с "1" на "4"
    assert limits["cpu"] == "4", f"DEFAULT_CONFIG перебил variables.json. cpu = {limits['cpu']!r}"


def test_maintenance_schedule_from_variables() -> None:
    import dags.system_maintenance as mod

    actual = str(_get_schedule(mod.dag))
    assert actual == "0 3 * * 0", (
        f"Расписание взято из DEFAULT_CONFIG (@daily). Реальное: {actual!r}"
    )


def test_quality_control_drift_threshold_from_variables() -> None:
    import dags.quality_control as mod

    task = mod.dag.get_task("evaluate_llm")
    args = task.arguments
    # Исправлено с "0.9" на "0.45"
    assert any("0.45" in str(a) for a in args), (
        f"rouge1_threshold из variables.json (0.45) не попал в аргументы. Аргументы: {args}"
    )
