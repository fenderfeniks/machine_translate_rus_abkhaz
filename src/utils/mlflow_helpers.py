# src/utils/mlflow_helpers.py
import logging
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def _ensure_tracking_uri() -> None:
    """Гарантирует что MLflow использует правильный tracking URI из env.

    Если переменная окружения ``MLFLOW_TRACKING_URI`` задана —
    явно выставляет URI в MLflow. Иначе логирует предупреждение
    о fallback на локальную директорию ``./mlruns``.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        logger.debug("MLflow tracking URI: %s", uri)
    else:
        logger.warning(
            "MLFLOW_TRACKING_URI не задан. MLflow будет использовать ./mlruns по умолчанию."
        )


def _resolve_peft_path(downloaded_path: str) -> str:
    """Нормализует путь к скачанным артефактам LoRA.

    ``mlflow.transformers.log_model`` сохраняет PEFT-веса в подпапку
    ``peft/``, тогда как ``PeftModel.from_pretrained`` ожидает папку
    с ``adapter_config.json`` напрямую. Функция прозрачно разрешает
    этот сдвиг.

    Args:
        downloaded_path: Путь к скачанной директории артефакта.

    Returns:
        Путь к директории с ``adapter_config.json``:
        либо ``<downloaded_path>/peft/``, либо ``downloaded_path`` как есть.
    """
    path = Path(downloaded_path.rstrip("\\/"))
    peft_subdir = path / "peft"
    if peft_subdir.exists() and (peft_subdir / "adapter_config.json").exists():
        logger.info("Найдена PEFT подпапка: %s", peft_subdir)
        return str(peft_subdir)
    return str(path)


def resolve_lora_resume_path(resume_cfg: DictConfig) -> str | None:
    """Разрешает путь к весам LoRA из MLflow на основе конфигурации.

    Поддерживает два режима загрузки:
    - по точному ``run_id``;
    - по алиасу из Model Registry (``model_name`` + ``alias``).

    Args:
        resume_cfg: Секция конфига ``model.lora_resume``.
            Ожидаемые ключи:

            - ``enabled`` (bool): флаг активации загрузки.
            - ``run_id`` (str, optional): точный run_id MLflow.
            - ``model_name`` (str, optional): имя модели в Registry.
            - ``alias`` (str, optional): алиас версии в Registry.
            - ``artifact_path`` (str): путь к артефакту внутри run.
              По умолчанию ``"lora_weights"``.

    Returns:
        Путь к директории с весами LoRA, либо ``None``
        если загрузка отключена (``enabled=False``).

    Raises:
        MlflowException:
            Если модель или алиас не найдены в MLflow Registry.
        ValueError:
            Если не переданы ни ``run_id``, ни пара
            ``model_name`` + ``alias``.
    """
    _ensure_tracking_uri()

    if not resume_cfg.get("enabled", False):
        return None

    run_id = resume_cfg.get("run_id")
    model_name = resume_cfg.get("model_name")
    alias = resume_cfg.get("alias")
    artifact_path = resume_cfg.get("artifact_path", "lora_weights")

    if run_id:
        logger.info("Скачивание LoRA адаптера по run_id: %s", run_id)
        downloaded = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        return _resolve_peft_path(downloaded)

    if model_name and alias:
        logger.info(
            "Поиск LoRA адаптера в Model Registry: %s (алиас='%s')",
            model_name,
            alias,
        )
        client = MlflowClient()
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            logger.error(
                "Не удалось найти модель '%s' с алиасом '%s'.",
                model_name,
                alias,
            )
            raise MlflowException(f"Ошибка загрузки из MLflow Registry: {e.message}") from e

        resolved_run_id = model_version.run_id
        logger.info(
            "Найдена версия %s (run_id: %s)",
            model_version.version,
            resolved_run_id,
        )
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=resolved_run_id, artifact_path=artifact_path
        )
        return _resolve_peft_path(downloaded)

    raise ValueError(
        "В model.lora_resume не указаны достаточные параметры. "
        "Передайте либо 'run_id', либо комбинацию 'model_name' и 'alias'."
    )
