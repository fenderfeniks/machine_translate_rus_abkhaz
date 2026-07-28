# src/tools/batch_analytics.py
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from src.sdk.inference import LLMGenerationPipeline


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Основная точка входа для batch-аналитики."""
    # Проверяем наличие доступов (например, к базе данных)
    db_url = os.getenv("DB_CONN")
    if not db_url:
        raise ValueError("DB_CONN is not set! Check your K8s secrets.")

    logger.info("Инициализация LLMGenerationPipeline...")
    try:
        pipeline = LLMGenerationPipeline(config_name="train")
    except Exception as e:
        logger.exception("Не удалось инициализировать пайплайн: %s", e)
        sys.exit(1)

    # Имитация данных (в реальности здесь будет выгрузка по DB_CONN)
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "prompt": [
                "Напиши краткое саммари для новости о снижении ключевой ставки.",
                "Объясни, что такое градиентный спуск.",
            ],
        }
    )

    logger.info("Запуск батч-генерации текстов...")

    results = pipeline(df["prompt"].tolist())

    df["generated_text"] = [res["generated_text"] for res in results]

    logger.info("Пример результатов:\n%s", df.head())
    logger.info("Батч-аналитика успешно завершена.")


if __name__ == "__main__":
    main()
