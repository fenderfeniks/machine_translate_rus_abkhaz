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


def _find_adapter_config(root: Path) -> Path | None:
    """Ищет adapter_config.json рекурсивно — на случай вложенных структур артефактов."""
    # Сначала проверяем корень и типичные подпапки
    for candidate in [root, root / "peft", root / "lora_weights", root / "adapter"]:
        if (candidate / "adapter_config.json").exists():
            return candidate
    # Рекурсивный поиск как последний вариант
    matches = list(root.rglob("adapter_config.json"))
    if matches:
        return matches[0].parent
    return None


def resolve_lora_resume_path(
    resume_cfg: DictConfig | dict,
    tracking_uri: str | None = None,
) -> str | None:
    """Разрешает путь к PEFT-адаптеру из MLflow artifacts.

    Поддерживает два режима:
    - run_id: скачивает артефакт напрямую из run
    - model_name + alias: ищет версию в Model Registry по алиасу

    Args:
        resume_cfg: конфиг с параметрами поиска адаптера
        tracking_uri: MLflow tracking URI — берётся из cfg, не из env.
                      Если None — откатывается к MLFLOW_TRACKING_URI из env.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        _ensure_tracking_uri()

    if isinstance(resume_cfg, DictConfig):
        resume_cfg = OmegaConf.to_container(resume_cfg, resolve=True)

    if not resume_cfg or not resume_cfg.get("enabled", False):
        return None

    run_id = resume_cfg.get("run_id")
    model_name = resume_cfg.get("model_name")
    alias = resume_cfg.get("alias")
    artifact_path = resume_cfg.get("artifact_path", "lora_weights")

    if run_id:
        logger.info("Скачивание LoRA адаптера по run_id=%s artifact_path=%s", run_id, artifact_path)
        downloaded = Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        )
    elif model_name and alias:
        logger.info("Поиск LoRA в Registry: %s (alias='%s')", model_name, alias)
        client = MlflowClient()
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            raise MlflowException(
                f"Алиас '{alias}' не найден для модели '{model_name}': {e.message}"
            ) from e

        logger.info(
            "Найдена версия %s (run_id=%s, source=%s), скачиваем...",
            model_version.version,
            model_version.run_id,
            model_version.source,
        )

        # run_id может быть None если модель зарегистрирована через register_model(artifact_uri)
        # В этом случае берём source URI напрямую из версии
        if model_version.run_id:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=model_version.run_id, artifact_path=artifact_path
                )
            )
        elif model_version.source:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(artifact_uri=model_version.source)
            )
        else:
            raise MlflowException(
                f"Версия {model_version.version} модели '{model_name}' "
                f"не содержит ни run_id, ни source URI."
            )
    else:
        raise ValueError("Укажите 'run_id' или комбинацию 'model_name' + 'alias' для резьюма.")

    adapter_dir = _find_adapter_config(downloaded)
    if adapter_dir is None:
        raise FileNotFoundError(
            f"adapter_config.json не найден в скачанных артефактах: {downloaded}\n"
            f"Содержимое: {list(downloaded.rglob('*'))}"
        )

    logger.info("PEFT-адаптер найден: %s", adapter_dir)
    return str(adapter_dir)


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
    """Сохраняет PEFT LoRA-адаптер в MLflow через save_pretrained + log_artifacts.

    Вместо mlflow.transformers.log_model (который тянет полную модель и не работает
    с PEFT из коробки) используем нативный PEFT-формат:
        adapter_config.json  ← конфиг LoRA (ranks, targets, ...)
        adapter_model.safetensors  ← только дельта-веса адаптера

    Это то что умеет читать PeftModel.from_pretrained() и resolve_lora_resume_path().
    """
    import tempfile

    logger.info("Подготовка к сохранению LoRA-адаптера в MLflow (run_id=%s)...", run_id)

    gc.collect()
    register_safe_globals()

    model_to_save = model_module.model
    client = MlflowClient()

    registry_cfg = cfg.get("logger", {}).get("registry", {})
    base_model_name = registry_cfg.get("model_name", "GenerativeLLM")
    reg_model_name = f"{base_model_name}_LoRA"
    artifact_path = registry_cfg.get("artifact_path", "lora_weights")
    register_on_success = registry_cfg.get("register_on_success", True)
    promote_to_staging = registry_cfg.get("promote_to_staging", True)

    # Патчим Hydra-типы в peft_config чтобы save_pretrained не упал на ListConfig
    _patch_peft_config_for_hydra(model_to_save)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Сохраняем только адаптер (adapter_config.json + adapter_model.safetensors)
        # НЕ всю модель — это несколько МБ вместо десятков ГБ
        logger.info("Сохранение PEFT-адаптера во временную директорию: %s", tmp_path)
        model_to_save.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)

        # Проверяем что adapter_config.json реально создался
        if not (tmp_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"save_pretrained не создал adapter_config.json в {tmp_path}. "
                f"Содержимое: {list(tmp_path.iterdir())}"
            )

        saved_files = [f.name for f in tmp_path.iterdir()]
        logger.info("Файлы адаптера: %s", saved_files)

        with mlflow.start_run(run_id=run_id):
            # Логируем всю папку как артефакт — сохраняется структура директории
            mlflow.log_artifacts(str(tmp_path), artifact_path=artifact_path)
            logger.info("LoRA адаптер сохранён в run_id=%s artifact_path=%s", run_id, artifact_path)

            if best_score is not None:
                mlflow.log_metric("promotion_candidate_val_loss", best_score)

            if not register_on_success:
                logger.info("Регистрация в Model Registry отключена (register_on_success=false).")
                return

            # Регистрируем артефакт как версию модели в Registry
            artifact_uri = mlflow.get_artifact_uri(artifact_path)
            mv_version = mlflow.register_model(
                model_uri=artifact_uri,
                name=reg_model_name,
            ).version
            logger.info("Зарегистрирована '%s' версия %s.", reg_model_name, mv_version)

            if promote_to_staging:
                client.set_registered_model_alias(
                    name=reg_model_name, alias="Staging", version=mv_version
                )
                logger.info(
                    "Модель '%s' версии %s помечена алиасом 'Staging'.", reg_model_name, mv_version
                )

            if best_score is not None:
                client.set_model_version_tag(
                    reg_model_name, mv_version, "val_loss", str(best_score)
                )
