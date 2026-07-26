import json
from pathlib import Path
from unittest.mock import patch

import pytest


VARIABLES_PATH = Path(__file__).parents[2] / "deploy" / "airflow" / "variables.json"

with open(VARIABLES_PATH) as f:
    REAL_VARIABLES = json.load(f)


def _variable_get(key, default_var=None, deserialize_json=False, **kwargs):
    """
    Эмулирует Variable.get() используя реальный variables.json.
    Так тесты проверяют именно тот конфиг который пойдёт в прод.
    """
    if key in REAL_VARIABLES:
        value = REAL_VARIABLES[key]
        return value  # уже dict, deserialize_json не нужен
    return default_var


@pytest.fixture(autouse=True)
def mock_airflow_variable():
    with patch("airflow.models.Variable.get", side_effect=_variable_get):
        yield


@pytest.fixture(autouse=True)
def mock_kubernetes_pod_operator():
    """
    Не даём KubernetesPodOperator реально подключаться к кластеру при импорте DAG.
    """
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
