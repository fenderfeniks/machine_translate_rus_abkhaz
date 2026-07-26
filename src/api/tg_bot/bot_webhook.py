# src/api/tg_bot/bot_webhook.py
"""Инициализация диспетчера Telegram-бота для работы через Webhooks.

Интегрируется напрямую в веб-сервер FastAPI. Поддерживает RedisStorage
для безопасного масштабирования стейт-машины бота.
"""

import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.redis import RedisStorage

from src.api.tg_bot.handlers.chat import router as chat_router


logger = logging.getLogger(__name__)

# Подключаем Redis для масштабирования FSM (Fallback: in-memory)
redis_url = os.getenv("REDIS_URL")
if redis_url:
    logger.info("Инициализация RedisStorage для стейт-машины бота.")
    storage = RedisStorage.from_url(redis_url)
else:
    logger.warning(
        "REDIS_URL не задан, используется In-Memory хранилище. В кластере могут теряться сессии!"
    )
    storage = MemoryStorage()

dp = Dispatcher(storage=storage)
dp.include_router(chat_router)


def get_webhook_bot(token: str) -> Bot:
    """Инициализирует экземпляр бота для вебхуков.

    Args:
        token: Строка с токеном Telegram-бота.

    Returns:
        Инициализированный объект Bot.
    """
    logger.info("Инициализация экземпляра Bot для Webhooks...")
    return Bot(token=token)
