# tests/dags/test_dag_structure.py
"""Тесты структуры Airflow DAG-ов.

Проверяют dag_id, расписание, состав тасков и порядок зависимостей.
"""

import importlib
from typing import Any

import pytest
from airflow import DAG


def _get_schedule(dag: DAG) -> Any:
    """Совместимо с Airflow 2.4+ где schedule_interval убран."""
    return getattr(dag, "schedule", None) or getattr(dag, "schedule_interval", None)


@pytest.mark.parametrize(
    "module,expected",
    [
        (
            "dags.retrain_model_dag",
            {
                "dag_id": "weekly_llm_finetuning",
                "schedule": "@weekly",
                "task_order": [
                    ("run_lora_finetuning", "merge_lora_weights"),
                    ("merge_lora_weights", "evaluate_staging_model"),
                    ("evaluate_staging_model", "request_manual_approval"),
                ],
                "all_tasks": {
                    "run_lora_finetuning",
                    "merge_lora_weights",
                    "evaluate_staging_model",
                    "request_manual_approval",
                },
            },
        ),
        (
            "dags.promote_to_prod",
            {
                "dag_id": "promote_llm_to_prod",
                "schedule": None,
                "task_order": [
                    ("promote_staging_to_prod", "restart_api_deployment"),
                ],
                "all_tasks": {
                    "promote_staging_to_prod",
                    "restart_api_deployment",
                },
            },
        ),
        (
            "dags.quality_control",
            {
                "dag_id": "llm_quality_drift_detection",
                "schedule": "@weekly",
                "task_order": [
                    ("evaluate_llm", "alert_if_drift"),
                ],
                "all_tasks": {"evaluate_llm", "alert_if_drift"},
            },
        ),
        (
            "dags.batch_analytics",
            {
                "dag_id": "batch_analytics_reporting",
                "schedule": "0 8 * * *",
                "task_order": [],
                "all_tasks": {"run_batch_inference"},
            },
        ),
        (
            "dags.system_maintenance",
            {
                "dag_id": "system_maintenance",
                "schedule": "0 3 * * 0",
                "task_order": [],
                "all_tasks": {"cleanup_logs_and_mlruns"},
            },
        ),
    ],
)
def test_dag_structure(module: str, expected: dict[str, Any]) -> None:
    mod = importlib.import_module(module)
    dag = mod.dag

    assert dag.dag_id == expected["dag_id"]
    assert str(_get_schedule(dag)) == str(expected["schedule"])
    assert {t.task_id for t in dag.tasks} == expected["all_tasks"]

    for upstream_id, downstream_id in expected["task_order"]:
        upstream = dag.get_task(upstream_id)
        assert downstream_id in upstream.downstream_task_ids


def test_promote_is_manual_trigger_only() -> None:
    import dags.promote_to_prod as mod

    assert _get_schedule(mod.dag) is None


def test_retrain_does_not_auto_promote() -> None:
    import dags.retrain_model_dag as mod

    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "promote_llm_to_prod" not in task_ids
