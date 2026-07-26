# src/jobs/batch_analytics.py
import logging
import os
import sys
from typing import Any

import pandas as pd
import torch
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_mlflow(tracking_uri: str, model_name: str, alias: str = "Production") -> Any:
    """Загружает пайплайн модели из MLflow Model Registry.

    Args:
        tracking_uri: URI сервера MLflow (например, 'http://localhost:5000').
        model_name: Имя зарегистрированной модели в Registry.
        alias: Алиас версии модели для загрузки. По умолчанию 'Production'.

    Returns:
        Пайплайн Hugging Face transformers.
    """
    import mlflow.transformers

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{alias}"
    logger.info("Загрузка модели из MLflow: %s", model_uri)

    pipeline = mlflow.transformers.load_model(
        model_uri,
        device=0 if torch.cuda.is_available() else -1,
        # Для LLM лучше явно указать возврат только новых токенов
        return_full_text=False,
    )
    return pipeline


def main() -> None:
    """Основная точка входа для batch-аналитики."""
    db_url = os.getenv("DB_CONN")
    if not db_url:
        raise ValueError("DB_CONN is not set! Check your K8s secrets.")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set! Check your configmap.")

    model_name = os.getenv("MLFLOW_MODEL_NAME", "GenerativeLLM")

    try:
        generator = load_model_from_mlflow(mlflow_uri, model_name)
    except Exception as e:
        logger.exception("Не удалось загрузить модель из MLflow: %s", e)
        sys.exit(1)

    # Имитация данных
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "prompt": [
                "Напиши краткое саммари для новости о снижении ключевой ставки.",
                "Объясни, что такое градиентный спуск.",
            ],
        }
    )

    logger.info("Running batch text generation...")
    # Передаем параметры генерации
    results = generator(df["prompt"].tolist(), max_new_tokens=150, temperature=0.3)

    # Парсинг ответа генеративной модели
    df["generated_text"] = [res[0]["generated_text"] for res in results]

    logger.info("Sample results:\n%s", df.head())
    logger.info("Batch analytics completed successfully.")


if __name__ == "__main__":
    main()
