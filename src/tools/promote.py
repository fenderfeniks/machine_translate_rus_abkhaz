# src/tools/promote.py
import logging
import os
import sys
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Оценивает Staging-модель и выполняет Promotion в Production.

    Сравнивает метрику val_loss версии Staging с текущей версией
    Production. Если Staging превосходит Production, обновляет
    алиасы в MLflow Model Registry.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Безопасное определение пути через pathlib
    default_db_path = (Path(__file__).resolve().parents[2] / "logs" / "mlflow.db").resolve()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{default_db_path}")

    logger.info("Используется MLFLOW_TRACKING_URI: %s", tracking_uri)
    model_name = os.getenv("MLFLOW_MODEL_NAME", "GenerativeLLM")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1. Берём текущий Staging
    try:
        staging_mv = client.get_model_version_by_alias(model_name, "Staging")
    except MlflowException:
        logger.error("Алиас 'Staging' не найден для модели '%s'.", model_name)
        sys.exit(1)

    staging_version = staging_mv.version
    staging_score_str = staging_mv.tags.get("val_loss")

    if staging_score_str is None:
        logger.error("У Staging модели нет тега 'val_loss'. Невозможно оценить качество.")
        sys.exit(1)

    staging_score = float(staging_score_str)

    # 2. Проверяем текущий Production
    try:
        current_prod = client.get_model_version_by_alias(model_name, "Production")
        if current_prod.version == staging_version:
            logger.warning("Версия %s уже является Production. Промоут пропущен.", staging_version)
            return

        prod_score_str = current_prod.tags.get("val_loss")
        prod_score = float(prod_score_str) if prod_score_str else float("inf")

        logger.info(
            "Текущий Production: версия %s (val_loss=%.4f)", current_prod.version, prod_score
        )
    except MlflowException:
        logger.info("Production алиаса ещё нет — первый промоут.")
        prod_score = float("inf")

    logger.info("Сравнение: Staging (%.4f) vs Production (%.4f)", staging_score, prod_score)

    if staging_score < prod_score:
        client.set_registered_model_alias(model_name, "Production", staging_version)
        logger.info(
            "УСПЕХ! Версия %s (val_loss=%.4f) стала новой Production моделью.",
            staging_version,
            staging_score,
        )
    else:
        logger.warning(
            "ОТКАЗ: Модель в Staging (%.4f) хуже или равна текущей Production (%.4f). "
            "Промоут отменен.",
            staging_score,
            prod_score,
        )


if __name__ == "__main__":
    main()
