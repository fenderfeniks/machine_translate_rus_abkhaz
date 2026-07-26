# scripts/train.py
"""Главный скрипт запуска обучения (Orchestrator) для Causal LM."""

import gc
import logging
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()

from src.core.models.generator import HFTextGenerator  # noqa E402
from src.utils.hydra_utils import setup_config  # noqa E402
from src.utils.logger import setup_logging  # noqa E402
from src.utils.mlflow_helpers import resolve_lora_resume_path  # noqa E402
from src.utils.mlflow_registry import log_lora_to_mlflow  # noqa E402
from src.utils.torch_utils import register_safe_globals  # noqa E402


setup_logging()
logger = logging.getLogger(__name__)


def build_callbacks(
    cfg: DictConfig, text_generator: Any, val_raw_dataset: list[Any]
) -> list[pl.Callback]:
    """Фабрика коллбэков с динамическим пробросом runtime-зависимостей.

    Args:
        cfg: Конфигурация Hydra (DictConfig).
        text_generator: Инстанс генератора для эвалюации.
        val_raw_dataset: Сырой валидационный датасет для генерации.

    Returns:
        Список проинициализированных коллбэков.
    """
    callbacks = []
    if not cfg.get("callbacks"):
        return callbacks

    for cb_name, cb_conf in cfg.callbacks.items():
        if cb_name == "generation_eval":
            cb = hydra.utils.instantiate(
                cb_conf, generator=text_generator, val_raw_dataset=val_raw_dataset
            )
            logger.info("Инициализирован сложный коллбэк: %s", cb_name)
        else:
            cb = hydra.utils.instantiate(cb_conf)
            logger.info("Инициализирован стандартный коллбэк: %s", cb_name)
        callbacks.append(cb)

    return callbacks


def _run_post_training_evaluation(
    trainer: pl.Trainer, model_module: pl.LightningModule, datamodule: pl.LightningDataModule
) -> float | None:
    """Изолированная логика тестирования лучшей модели после завершения обучения.

    Загружает лучшие веса (best_model_path) и запускает прогон
    по тестовой выборке.

    Args:
        trainer: Завершенный инстанс тренера Lightning.
        model_module: Модуль модели.
        datamodule: Модуль данных.

    Returns:
        Значение лучшей метрики (val_loss) или None, если чекпоинт не найден.
    """
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден, оценка пропущена.")
        return None

    register_safe_globals()
    logger.info("Загрузка лучших весов из %s для оценки...", best_ckpt_path)

    checkpoint = torch.load(best_ckpt_path, map_location=model_module.device, weights_only=False)
    lora_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
    model_module.load_state_dict(lora_state_dict, strict=False)

    logger.info("Запуск тестирования на отложенной выборке...")
    trainer.test(model=model_module, datamodule=datamodule)

    return float(trainer.checkpoint_callback.best_model_score or 0) or None


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Оркестратор обучения Causal LM.

    Инициализирует данные, модель, коллбэки и тренера. Поддерживает
    восстановление обучения из чекпоинтов и автоматическую регистрацию
    лучшей модели в MLflow после успешного завершения.

    Args:
        cfg: Конфигурация Hydra.
    """
    cfg = setup_config(cfg)
    logger.info("Старт скрипта обучения...")

    if cfg.trainer.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен GPU, но CUDA недоступна!")

    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)

    logger.info("Инициализация токенизатора и архитектуры...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # Делегирование загрузки адаптера из MLflow новой утилите
    resume_cfg = cfg.model.get("lora_resume", {})
    lora_resume_path = resolve_lora_resume_path(resume_cfg)

    base_model = hydra.utils.instantiate(
        cfg.model.builder, tokenizer=tokenizer, lora_resume_path=lora_resume_path
    ).build()

    logger.info("Инициализация DataModule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    logger.info("Инициализация компонентов генерации и модели...")
    text_generator = HFTextGenerator(
        model=base_model,
        tokenizer=tokenizer,
        generation_kwargs=cfg.get("generation_kwargs", {}),
    )
    model_module = hydra.utils.instantiate(cfg.model_module, model=base_model)

    if getattr(cfg.model, "compile", False):
        model_module.model = torch.compile(model_module.model)

    logger.info("Инициализация PyTorch Lightning Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # === 4. AUTO-RESUME (Поиск чекпоинтов) ===
    resume_path = None

    if cfg.get("resume_training", False):
        ckpt_dir = Path(cfg.paths.output_dir) / "checkpoints"
        last_ckpt = ckpt_dir / "last.ckpt"

        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            logger.info("Найден чекпоинт! Возобновление обучения с: %s", resume_path)
        else:
            logger.info(
                "Флаг resume_training=True, но last.ckpt не найден. Обучение начнется с нуля."
            )
    else:
        logger.info(
            "Флаг resume_training=False. Обучение начнется с нуля, старые чекпоинты игнорируются."
        )

    register_safe_globals()
    try:
        trainer.fit(model=model_module, datamodule=datamodule, ckpt_path=resume_path)
        print(type(trainer.logger))
        print(dir(trainer.logger))
        print(getattr(trainer.logger, "run_id", "НЕТ"))
        print(getattr(trainer.logger, "_run_id", "НЕТ"))
        print(getattr(trainer.logger, "experiment", "НЕТ"))
        logger.info("Обучение успешно завершено!")
    except KeyboardInterrupt:
        logger.warning("Обучение прервано (Ctrl+C)! Переход к сохранению артефактов...")
    except Exception:
        logger.exception("Критическая ошибка обучения:")
        raise

    finally:
        # 1. Загрузка лучших весов и оценка
        best_score = _run_post_training_evaluation(trainer, model_module, datamodule)

        # <-- ИЗВЛЕКАЕМ RUN_ID ИЗ ЛОГГЕРА ДО ЕГО УДАЛЕНИЯ -->
        mlflow_run_id = None
        if trainer.logger:
            # MLFlowLogger хранит run_id в публичном свойстве
            for attr in ("run_id", "_run_id", "runid"):
                val = getattr(trainer.logger, attr, None)
                if val:
                    mlflow_run_id = val
                    break
            # Fallback: достать через experiment (активный run MLflow)
            if mlflow_run_id is None:
                try:
                    import mlflow

                    active = mlflow.active_run()
                    if active:
                        mlflow_run_id = active.info.run_id
                except Exception:
                    pass

        logger.info("MLflow run_id для сохранения артефактов: %s", mlflow_run_id)

        # 2. Очистка ресурсов
        logger.info("Очистка памяти перед интеграцией с MLflow...")
        del trainer
        del datamodule
        del text_generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Сохранение и регистрация в MLflow
        if best_score is not None and mlflow_run_id is not None:
            log_lora_to_mlflow(
                cfg=cfg,
                model_module=model_module,
                tokenizer=tokenizer,
                run_id=mlflow_run_id,
                best_score=best_score,
            )
        else:
            logger.warning(
                "Не удалось сохранить модель в MLflow: best_score или mlflow_run_id отсутствуют."
            )


if __name__ == "__main__":
    train()
