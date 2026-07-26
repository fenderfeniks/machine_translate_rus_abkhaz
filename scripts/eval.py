# scripts/eval.py
import json
import logging
import sys
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()

from src.utils.hydra_utils import setup_config  # noqa E402
from src.utils.logger import setup_logging  # noqa E402
from src.utils.mlflow_helpers import resolve_lora_resume_path  # noqa E402
from src.utils.torch_utils import register_safe_globals  # noqa E402


setup_logging()
logger = logging.getLogger(__name__)


def _check_drift(
    metrics: dict[str, Any], drift_threshold: float, metric_key: str = "test_perplexity"
) -> None:
    """Анализирует метрики на предмет деградации (дрифта) модели.

    Если целевая метрика хуже заданного порога, скрипт завершается
    с кодом ошибки (1), сигнализируя оркестратору об остановке пайплайна.

    Args:
        metrics: Словарь с результатами оценки модели.
        drift_threshold: Пороговое значение для фиксации дрифта.
        metric_key: Ключ целевой метрики в словаре.
    """
    primary_metric = metrics.get(metric_key)

    if primary_metric is None:
        logger.warning(
            "Ключ '%s' не найден в результатах. Доступные: %s.",
            metric_key,
            list(metrics.keys()),
        )
        return

    logger.info("Метрика %s: %.4f, порог дрифта: %s", metric_key, primary_metric, drift_threshold)

    is_lower_better = metric_key in ["test_loss", "test_perplexity"]

    if is_lower_better and primary_metric > drift_threshold:
        logger.error(
            "ДРИФТ (деградация): %.4f > %s. Выход с кодом 1.",
            primary_metric,
            drift_threshold,
        )
        sys.exit(1)
    elif not is_lower_better and primary_metric < drift_threshold:
        logger.error(
            "ДРИФТ (деградация): %.4f < %s. Выход с кодом 1.",
            primary_metric,
            drift_threshold,
        )
        sys.exit(1)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    """Точка входа для скрипта оценки модели.

    Подготавливает окружение, загружает веса (базовые или с LoRA),
    проводит тестирование на отложенной выборке, сохраняет метрики
    и вызывает проверку на дрифт.

    Args:
        cfg: Конфигурация Hydra (DictConfig).
    """
    cfg = setup_config(cfg)

    logger.info("Инициализация компонентов для оценки...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # Загрузка LoRA адаптера из MLflow Registry (если включено в конфиге)
    resume_cfg = cfg.model.get("lora_resume", {})
    lora_resume_path = resolve_lora_resume_path(resume_cfg)
    if lora_resume_path:
        logger.info("LoRA адаптер будет загружен из: %s", lora_resume_path)
    else:
        logger.warning("lora_resume.enabled=false — оценка на базовой архитектуре без адаптера.")

    base_model = hydra.utils.instantiate(
        cfg.model.builder, tokenizer=tokenizer, lora_resume_path=lora_resume_path
    ).build()

    model_module = hydra.utils.instantiate(cfg.model_module, model=base_model)
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Опциональная загрузка Lightning-чекпоинта поверх (для отладки/экспериментов)
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info("Загрузка кастомного Lightning-чекпоинта из: %s", ckpt_path)
        register_safe_globals()
        from src.utils.checkpoint_utils import load_checkpoint

        model_module.model = load_checkpoint(model_module.model, ckpt_path, device="cpu")
        ckpt_path = None

    logger.info("Старт процесса оценки...")
    results = trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Оценка завершена.")

    if not results:
        logger.warning("trainer.test() вернул пустые результаты.")
        return

    metrics = results[0]

    # === ЭКСПОРТ ДЛЯ ОРКЕСТРАТОРА (Apache Airflow) ===
    metrics_file = cfg.get("metrics_output_path", "metrics.json")
    try:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        logger.info("Метрики успешно экспортированы в %s для оркестратора.", metrics_file)
    except Exception as e:
        logger.error("Не удалось экспортировать метрики: %s", e)
        raise

    # === ПРОВЕРКА ДРИФТА ===
    drift_threshold = cfg.get("drift_threshold")
    metric_key = cfg.get("drift_metric_key", "test_perplexity")

    if drift_threshold is not None:
        _check_drift(metrics, drift_threshold=drift_threshold, metric_key=metric_key)


if __name__ == "__main__":
    try:
        evaluate()
    except SystemExit as e:
        raise e
