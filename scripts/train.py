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
    if not trainer.logger:
        return None

    for attr in ("run_id", "_run_id", "runid"):
        val = getattr(trainer.logger, attr, None)
        if val:
            return val

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
    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден. Запускаем тест на текущих весах (last state)...")
        trainer.test(model=model_module, datamodule=datamodule)
        return None

    register_safe_globals()
    logger.info("Загрузка лучших весов из %s...", best_ckpt_path)

    checkpoint = torch.load(best_ckpt_path, map_location=model_module.device, weights_only=False)
    lora_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
    model_module.load_state_dict(lora_state_dict, strict=False)

    logger.info("Тестирование на отложенной выборке (best model)...")
    trainer.test(model=model_module, datamodule=datamodule)

    score = trainer.checkpoint_callback.best_model_score
    return float(score) if score is not None else None


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Старт обучения...")

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
    lora_resume_path = resolve_lora_resume_path(cfg.model.get("lora_resume", {}))

    logger.info("Сборка модели...")
    builder = hydra.utils.instantiate(cfg.model.builder)
    builder.lora_resume_path = lora_resume_path
    builder.modifiers_cfg = cfg.model.get("modifiers")
    base_model = builder.build(tokenizer=tokenizer)

    # ── 3. DataModule ─────────────────────────────────────────────────────────
    logger.info("Инициализация DataModule...")
    datamodule = NLPDataModule(data_cfg=cfg.data, tokenizer=tokenizer)

    # ── 4. LightningModule (с определением task_mode) ─────────────────────────
    data_cfg = cfg.data
    task_val = (
        data_cfg.get("task") if isinstance(data_cfg, dict) else getattr(data_cfg, "task", None)
    )

    if task_val in ["sft", "cpt"]:
        task_mode = task_val
    else:
        has_prompt = (
            bool(data_cfg.get("prompt_column"))
            if isinstance(data_cfg, dict)
            else bool(getattr(data_cfg, "prompt_column", None))
        )
        task_mode = "sft" if has_prompt else "cpt"

    model_module = CausalLMLightningModule(
        model=base_model,
        optimizer_cfg=hydra.utils.instantiate(cfg.optimizer),
        scheduler_cfg=hydra.utils.instantiate(cfg.scheduler) if "scheduler" in cfg else None,
        task_mode=task_mode,
    )

    if cfg.model.get("compile", False):
        logger.info("torch.compile включён — компиляция графа вычислений...")
        model_module.model = torch.compile(model_module.model)

    # ── 5. Trainer ────────────────────────────────────────────────────────────
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
        mlflow_run_id = _extract_mlflow_run_id(trainer)
        logger.info("MLflow run_id: %s", mlflow_run_id)

        best_score = _run_post_training_evaluation(trainer, model_module, datamodule)

        logger.info("Очистка памяти GPU...")
        del trainer
        del datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
