# src/utils/mlflow_registry.py
import gc
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.mlflow_requirements import get_inference_pip_requirements
from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)


def _patch_peft_config_for_hydra(model: Any) -> None:
    """Конвертирует OmegaConf-объекты внутри PEFT-конфига в нативные типы Python.

    При использовании Hydra поля PEFT-конфига могут содержать
    ``ListConfig`` / ``DictConfig`` вместо стандартных ``list`` / ``dict``.
    MLflow не умеет сериализовывать OmegaConf-типы, поэтому перед
    сохранением артефакта необходимо привести их к нативным типам.

    Args:
        model: Объект модели с атрибутом ``peft_config``
            (``dict[str, PeftConfigMixin]``). Если атрибут отсутствует,
            функция завершается без изменений.
    """
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
    artifact_path: str = "lora_weights",
) -> None:
    """Сохраняет LoRA-адаптер в MLflow и опционально регистрирует его в Model Registry.

    Выполняет следующие шаги:
    - освобождает память перед сохранением;
    - регистрирует безопасные глобалы для сериализации чекпоинтов;
    - патчит PEFT-конфиг для совместимости с Hydra;
    - логирует трансформерную модель как MLflow-артефакт;
    - при необходимости регистрирует модель в Registry с алиасом ``Staging``.

    Args:
        cfg: Hydra-конфиг проекта.
        model_module: Lightning-модуль с атрибутом ``model``.
        tokenizer: Токенизатор для сохранения вместе с моделью.
        run_id: Идентификатор MLflow run, в который пишем артефакт.
        best_score: Лучшее значение val_loss. Если передано — тегируется
            на версии модели в Registry и логируется как метрика.
        artifact_path: Путь к артефакту внутри run.
            По умолчанию ``"lora_weights"``.
    """
    logger.info("Подготовка к сохранению LoRA-адаптера в MLflow...")

    gc.collect()
    register_safe_globals()

    model_to_save = model_module.model
    base_model_name = cfg.model.get("mlflow_model_name", "GenerativeLLM")
    reg_model_name = f"{base_model_name}_LoRA"
    client = MlflowClient()

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

        logger.info(
            "LoRA адаптер сохранён в run_id: %s (путь: %s)",
            run_id,
            artifact_path,
        )

        if not cfg.model.builder.get("register_in_mlflow", True):
            return

        mv_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=reg_model_name,
        ).version

        client.set_registered_model_alias(
            name=reg_model_name,
            alias="Staging",
            version=mv_version,
        )

        if best_score is not None:
            client.set_model_version_tag(reg_model_name, mv_version, "val_loss", str(best_score))
            mlflow.log_metric("promotion_candidate_val_loss", best_score)

        logger.info(
            "Модель '%s' версии %s помечена алиасом 'Staging'.",
            reg_model_name,
            mv_version,
        )
