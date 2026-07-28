# src/utils/mlflow.py
import gc
import logging
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf

# Подтягиваем нашу утилиту безопасности
from src.utils.torch_utils import register_safe_globals


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

_INFERENCE_GROUP: str = "inference-core"


# ==========================================
# БЛОК 1: УПРАВЛЕНИЕ ЗАВИСИМОСТЯМИ
# ==========================================
def _strip_version_specifier(requirement: str) -> str:
    name = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
    return name


def get_inference_pip_requirements(pyproject_path: str | Path) -> list[str]:
    pyproject_path = Path(pyproject_path)
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        declared = data["project"]["optional-dependencies"][_INFERENCE_GROUP]
    except KeyError:
        logger.warning(
            "Группа [project.optional-dependencies.%s] не найдена в %s. ",
            _INFERENCE_GROUP,
            pyproject_path,
        )
        return []

    pinned: list[str] = []
    for requirement in declared:
        pkg_name = _strip_version_specifier(requirement)
        try:
            installed_version = version(pkg_name)
            pinned.append(f"{pkg_name}=={installed_version}")
        except PackageNotFoundError:
            logger.warning("Пакет '%s' не установлен — пропускаю.", pkg_name)

    return pinned


# ==========================================
# БЛОК 2: ЗАГРУЗКА АДАПТЕРОВ (RESUME)
# ==========================================
def _ensure_tracking_uri() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        logger.warning("MLFLOW_TRACKING_URI не задан. Используется ./mlruns")


def _resolve_peft_path(downloaded_path: str) -> str:
    path = Path(downloaded_path.rstrip("\\/"))
    peft_subdir = path / "peft"
    if peft_subdir.exists() and (peft_subdir / "adapter_config.json").exists():
        return str(peft_subdir)
    return str(path)


def resolve_lora_resume_path(resume_cfg: DictConfig | dict) -> str | None:
    """Разрешает путь к весам LoRA на основе обновленного конфига."""
    _ensure_tracking_uri()

    # Обертка для безопасности
    if isinstance(resume_cfg, DictConfig):
        resume_cfg = OmegaConf.to_container(resume_cfg, resolve=True)

    if not resume_cfg or not resume_cfg.get("enabled", False):
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
        logger.info("Поиск LoRA в Registry: %s (alias='%s')", model_name, alias)
        client = MlflowClient()
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            raise MlflowException(f"Ошибка Registry: {e.message}") from e

        downloaded = mlflow.artifacts.download_artifacts(
            run_id=model_version.run_id, artifact_path=artifact_path
        )
        return _resolve_peft_path(downloaded)

    raise ValueError("Укажите 'run_id' или комбинацию 'model_name' и 'alias' для резьюма.")


# ==========================================
# БЛОК 3: СОХРАНЕНИЕ АДАПТЕРОВ (LOGGING)
# ==========================================
def _patch_peft_config_for_hydra(model: Any) -> None:
    if not hasattr(model, "peft_config"):
        return

    for _, peft_cfg in model.peft_config.items():
        for key, value in vars(peft_cfg).items():
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(peft_cfg, key, OmegaConf.to_container(value, resolve=True))


def log_lora_to_mlflow(
    cfg: Any,
    model_module: Any,
    tokenizer: Any,
    run_id: str,
    best_score: float | None = None,
) -> None:
    """Сохраняет LoRA-адаптер в MLflow, читая настройки из cfg.logger.registry."""
    logger.info("Подготовка к сохранению LoRA-адаптера в MLflow...")

    gc.collect()
    register_safe_globals()

    model_to_save = model_module.model
    client = MlflowClient()

    # 1. Достаем параметры из нового блока логгера
    registry_cfg = cfg.get("logger", {}).get("registry", {})
    base_model_name = registry_cfg.get("model_name", "GenerativeLLM")

    # Формируем имя для Registry с постфиксом
    reg_model_name = f"{base_model_name}_LoRA"

    artifact_path = registry_cfg.get("artifact_path", "lora_weights")
    register_on_success = registry_cfg.get("register_on_success", True)
    promote_to_staging = registry_cfg.get("promote_to_staging", True)

    with mlflow.start_run(run_id=run_id):
        _patch_peft_config_for_hydra(model_to_save)

        pyproject_path = Path(cfg.paths.root_dir) / "pyproject.toml"
        pip_requirements = get_inference_pip_requirements(pyproject_path)

        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model_to_save, "tokenizer": tokenizer},
            artifact_path=artifact_path,
            task="text-generation",
            signature=None,
            input_example=None,
            pip_requirements=pip_requirements if pip_requirements else None,
        )

        logger.info("LoRA адаптер сохранён в run_id: %s (путь: %s)", run_id, artifact_path)

        # 2. Проверяем флаг регистрации из конфига
        if not register_on_success:
            logger.info("Регистрация в Model Registry отключена (register_on_success=false).")
            return

        mv_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=reg_model_name,
        ).version

        # 3. Установка алиасов и тегов
        if promote_to_staging:
            client.set_registered_model_alias(
                name=reg_model_name, alias="Staging", version=mv_version
            )
            logger.info(
                "Модель '%s' версии %s помечена алиасом 'Staging'.", reg_model_name, mv_version
            )

        if best_score is not None:
            client.set_model_version_tag(reg_model_name, mv_version, "val_loss", str(best_score))
            mlflow.log_metric("promotion_candidate_val_loss", best_score)
