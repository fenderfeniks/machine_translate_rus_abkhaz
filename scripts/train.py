# scripts/train.py
"""Оркестратор обучения Causal LM."""

import gc
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import PeftModel


load_dotenv()

from src.core.data.builder import NLPDataModule  # noqa: E402
from src.training.module import CausalLMLightningModule  # noqa: E402
from src.utils.hydra_utils import setup_config  # noqa: E402
from src.utils.logger import setup_logging  # noqa: E402
from src.utils.mlflow import log_lora_to_mlflow, resolve_lora_resume_path  # noqa: E402
from src.utils.torch_utils import register_safe_globals  # noqa: E402


setup_logging()
logger = logging.getLogger(__name__)


def _extract_mlflow_run_id(trainer: pl.Trainer) -> str | None:
    """Извлекает run_id из MLFlowLogger до удаления тренера.

    MLFlowLogger закрывает run при удалении объекта, поэтому run_id
    нужно сохранить до вызова del trainer.
    """
    if not trainer.logger:
        return None

    # Публичное свойство MLFlowLogger (lightning >= 2.0)
    for attr in ("run_id", "_run_id", "runid"):
        val = getattr(trainer.logger, attr, None)
        if val:
            return val

    # Fallback: активный run через mlflow SDK
    try:
        import mlflow

        active = mlflow.active_run()
        if active:
            return active.info.run_id
    except Exception:
        pass

    return None


def _run_post_training_evaluation(
    trainer: pl.Trainer,
    model_module: CausalLMLightningModule,
    datamodule: pl.LightningDataModule,
) -> float | None:
    """Загружает лучший чекпоинт и запускает тест на отложенной выборке.

    Returns:
        Значение best_model_score или None если чекпоинт не найден.
    """
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден — оценка пропущена.")
        return None

    register_safe_globals()
    logger.info("Загрузка лучших весов из %s...", best_ckpt_path)

    checkpoint = torch.load(best_ckpt_path, map_location=model_module.device, weights_only=False)
    lora_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
    model_module.load_state_dict(lora_state_dict, strict=False)

    logger.info("Тестирование на отложенной выборке...")
    trainer.test(model=model_module, datamodule=datamodule)

    score = trainer.checkpoint_callback.best_model_score
    return float(score) if score is not None else None


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Оркестратор обучения Causal LM.

    Порядок инициализации:
      1. Конфиг и seed
      2. Токенизатор → Модель (с опциональным LoRA-resume)
      3. DataModule
      4. LightningModule + torch.compile (опционально)
      5. Trainer (callbacks и logger подтягиваются из конфига Hydra)
      6. trainer.fit → post-training eval → MLflow регистрация
    """
    cfg = setup_config(cfg)
    logger.info("Старт обучения...")

    # Ранняя проверка: не тратить время на загрузку модели если GPU недоступен
    if cfg.trainer.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.trainer.accelerator='gpu', но CUDA недоступна. "
            "Используй environment=local для запуска на CPU."
        )

    pl.seed_everything(cfg.seed, workers=True)

    # ── 1. Токенизатор ───────────────────────────────────────────────────────
    logger.info("Загрузка токенизатора: %s", cfg.model.architecture.model_name_or_path)
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # ── 2. Модель ────────────────────────────────────────────────────────────
    # resolve_lora_resume_path: ищет адаптер в MLflow Registry или локальном пути
    lora_resume_path = resolve_lora_resume_path(cfg.model.get("lora_resume", {}))

    logger.info("Сборка модели...")
    # Инстанциируем только вложенный конфигуратор билдера
    builder = hydra.utils.instantiate(cfg.model.builder)
    builder.lora_resume_path = lora_resume_path

    # Модификаторы остались на верхнем уровне cfg.model, прокидываем их вручную
    builder.modifiers_cfg = cfg.model.get("modifiers")
    base_model = builder.build(tokenizer=tokenizer)

    # ── 3. DataModule ─────────────────────────────────────────────────────────
    logger.info("Инициализация DataModule...")

    # 1. Собираем коллатор отдельно, передавая ему runtime-токенизатор
    # collator = hydra.utils.instantiate(cfg.data.collator, tokenizer=tokenizer)

    # 2. Передаем готовый коллатор и токенизатор в датамодуль
    datamodule = NLPDataModule(data_cfg=cfg.data, tokenizer=tokenizer)

    # ── 4. LightningModule ────────────────────────────────────────────────────
    model_module = CausalLMLightningModule(
        model=base_model,
        # Инстанцируем конфиги здесь, чтобы они стали partial-функциями
        optimizer_cfg=hydra.utils.instantiate(cfg.optimizer),
        scheduler_cfg=hydra.utils.instantiate(cfg.scheduler) if "scheduler" in cfg else None,
    )

    if cfg.model.get("compile", False):
        logger.info("torch.compile включён — компиляция графа вычислений...")
        model_module.model = torch.compile(model_module.model)

    # ── 5. Trainer ────────────────────────────────────────────────────────────
    # Callbacks и logger объявлены в configs/trainer/ и подтягиваются Hydra.
    # instantiate рекурсивно создаёт все вложенные объекты (_target_ + параметры).
    logger.info("Инициализация Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # ── 6. Auto-resume ────────────────────────────────────────────────────────
    resume_path = None
    if cfg.get("resume_training", False):
        last_ckpt = Path(cfg.paths.log_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            logger.info("Resume: найден чекпоинт %s", resume_path)
        else:
            logger.warning("resume_training=True, но last.ckpt не найден — старт с нуля.")

    # ── 7. Обучение ───────────────────────────────────────────────────────────
    register_safe_globals()
    try:
        trainer.fit(model=model_module, datamodule=datamodule, ckpt_path=resume_path)
        logger.info("Обучение завершено.")
    except KeyboardInterrupt:
        logger.warning("Прервано (Ctrl+C) — переход к сохранению артефактов...")
    except Exception:
        logger.exception("Критическая ошибка во время обучения:")
        raise
    finally:
        # run_id извлекаем ДО del trainer — MLFlowLogger закрывает run при удалении
        mlflow_run_id = _extract_mlflow_run_id(trainer)
        logger.info("MLflow run_id: %s", mlflow_run_id)

        best_score = _run_post_training_evaluation(trainer, model_module, datamodule)

        logger.info("Очистка памяти GPU...")
        del trainer
        del datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Регистрируем только если обучение дало осмысленный результат
        is_peft = isinstance(base_model, PeftModel)
        if is_peft and best_score is not None and mlflow_run_id is not None:
            log_lora_to_mlflow(
                cfg=cfg,
                model_module=model_module,
                tokenizer=tokenizer,
                run_id=mlflow_run_id,
                best_score=best_score,
            )
        elif not is_peft:
            logger.info("Full Fine-Tuning — MLflow регистрация адаптеров пропущена.")
        else:
            logger.warning(
                "MLflow регистрация пропущена: best_score=%s, run_id=%s",
                best_score,
                mlflow_run_id,
            )


if __name__ == "__main__":
    train()
