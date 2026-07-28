# scripts/eval.py
import json
import logging
import sys
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


load_dotenv()
from src.core.data.builder import NLPDataModule  # noqa
from src.training.module import CausalLMLightningModule  # noqa
from src.utils.checkpoint_utils import load_checkpoint  # noqa
from src.utils.hydra_utils import setup_config  # noqa
from src.utils.logger import setup_logging  # noqa
from src.utils.mlflow import resolve_lora_resume_path  # noqa
from src.utils.torch_utils import register_safe_globals  # noqa


setup_logging()
logger = logging.getLogger(__name__)


def _check_drift(
    metrics: dict[str, Any], drift_threshold: float, metric_key: str = "test_perplexity"
) -> None:
    # (Функция остается без изменений, она отлично написана)
    primary_metric = metrics.get(metric_key)

    if primary_metric is None:
        logger.warning("Ключ '%s' не найден в результатах.", metric_key)
        return

    logger.info("Метрика %s: %.4f, порог дрифта: %s", metric_key, primary_metric, drift_threshold)
    is_lower_better = metric_key in ["test_loss", "test_perplexity"]

    if (is_lower_better and primary_metric > drift_threshold) or (
        not is_lower_better and primary_metric < drift_threshold
    ):
        logger.error("ДРИФТ (деградация). Выход с кодом 1.")
        sys.exit(1)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    logger.info("Инициализация компонентов для оценки...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    resume_cfg = cfg.get("lora_resume", {})
    lora_resume_path = resolve_lora_resume_path(resume_cfg)
    if lora_resume_path:
        logger.info("LoRA адаптер будет загружен из: %s", lora_resume_path)
        OmegaConf.update(
            cfg, "model.modifiers.finetuning.lora_resume_path", lora_resume_path, force_add=True
        )

    # Единая точка сборки модели со всеми модификаторами
    builder = hydra.utils.instantiate(cfg.model.builder)
    builder.modifiers_cfg = cfg.model.get("modifiers")
    base_model = builder.build(tokenizer=tokenizer)

    model_module = CausalLMLightningModule(
        model=base_model,
        optimizer_cfg=hydra.utils.instantiate(cfg.optimizer),
        scheduler_cfg=hydra.utils.instantiate(cfg.scheduler) if "scheduler" in cfg else None,
    )
    datamodule = NLPDataModule(data_cfg=cfg.data, tokenizer=tokenizer)
    trainer = hydra.utils.instantiate(cfg.trainer)

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info("Загрузка кастомного Lightning-чекпоинта из: %s", ckpt_path)
        register_safe_globals()
        model_module.model = load_checkpoint(model_module.model, ckpt_path, device="cpu")
        ckpt_path = None

    logger.info("Старт процесса оценки...")
    results = trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)

    if not results:
        logger.warning("trainer.test() вернул пустые результаты.")
        return

    metrics = results[0]
    metrics_file = cfg.get("metrics_output_path", "metrics.json")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info("Метрики успешно экспортированы в %s", metrics_file)

    drift_threshold = cfg.get("drift_threshold")
    if drift_threshold is not None:
        _check_drift(
            metrics,
            drift_threshold=drift_threshold,
            metric_key=cfg.get("drift_metric_key", "test_perplexity"),
        )


if __name__ == "__main__":
    evaluate()
