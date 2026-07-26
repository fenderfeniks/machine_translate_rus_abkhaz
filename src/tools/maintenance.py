# src/tools/maintenance.py
import argparse
import logging
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv


# Загружаем локальный .env (если запуск вне K8s)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_mlruns(days: int) -> None:
    """Удаляет папки и файлы артефактов, старше указанного количества дней.

    Очищает директорию, переданную в переменной окружения MLRUNS_DIR
    (по умолчанию /app/logs), от старых чекпоинтов и артефактов MLflow.

    Args:
        days: Возраст файлов в днях для удаления.
    """
    # Путь по умолчанию соответствует нашему mount_path в Airflow и Docker
    target_dir = Path(os.getenv("MLRUNS_DIR", "/app/logs"))
    logger.info("Запуск очистки логов в %s старше %d дней...", target_dir, days)

    if not target_dir.exists():
        logger.warning("Директория %s не существует. Очистка пропущена.", target_dir)
        return

    # Вычисляем timestamp отсечения (текущее время минус days в секундах)
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0

    for item_path in target_dir.iterdir():
        try:
            # Получаем время последней модификации файла/папки
            mtime = item_path.stat().st_mtime

            # Если объект старше точки отсечения — удаляем
            if mtime < cutoff_time:
                if item_path.is_dir():
                    shutil.rmtree(item_path)  # Рекурсивное удаление папки
                else:
                    item_path.unlink()  # Удаление файла
                deleted_count += 1
                logger.debug("Удалено: %s", item_path)
        except Exception as e:
            logger.error("Ошибка при удалении %s: %s", item_path, e)

    logger.info("Очистка завершена. Удалено старых объектов: %d.", deleted_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт обслуживания инфраструктуры (Очистка старых логов)"
    )
    parser.add_argument(
        "--action", choices=["cleanup"], required=True, help="Какое действие выполнить"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Возраст файлов в днях для удаления",
    )
    args = parser.parse_args()

    if args.action == "cleanup":
        cleanup_mlruns(args.days)
