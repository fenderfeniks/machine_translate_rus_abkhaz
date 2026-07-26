# src/utils/logger.py
import logging
import os
import sys


def setup_logging(default_level: int = logging.INFO) -> None:
    """Настраивает централизованное логирование для проекта.

    Устанавливает базовый обработчик вывода в stdout, очищает
    существующие обработчики и применяет форматирование в
    зависимости от переменной окружения ``ENVIRONMENT``.
    Для 'prod' используется JSON-формат (при наличии
    python-json-logger), для остальных окружений — человекочитаемый.
    Вызывается один раз в точке входа (например, в scripts/*.py).

    Args:
        default_level: Базовый уровень логирования.
            По умолчанию ``logging.INFO``.

    Returns:
        None
    """
    env = os.getenv("ENVIRONMENT", "dev")

    # Сбрасываем все текущие хэндлеры (если кто-то успел их инициализировать)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Базовый хэндлер для вывода в stdout
    console_handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter

    if env == "prod":
        try:
            # Формат JSON для продакшена (если установлен python-json-logger)
            from pythonjsonlogger import jsonlogger

            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        except ImportError:
            # Fallback, если забыли добавить в зависимости
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
    else:
        # Человекочитаемый формат для локальной разработки
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(default_level)

    # Глушим избыточный шум от сторонних библиотек
    noisy_loggers = ["httpx", "huggingface_hub", "urllib3"]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
