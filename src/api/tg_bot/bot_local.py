# src/api/tg_bot/bot_local.py
"""Локальный Telegram-бот (Long Polling).

Запускается как независимый процесс и общается с FastAPI по HTTP.
Идеально подходит для локальной разработки и тестирования.
"""

import asyncio
import logging
import os

import hydra
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.api.tg_bot.handlers.chat import router as chat_router


load_dotenv()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Точка входа для запуска локального бота."""
    OmegaConf.resolve(cfg)

    bot_token = os.getenv("TG_BOT_TOKEN") or cfg.api.telegram.bot_token
    if not bot_token:
        raise ValueError("Критическая ошибка: TG_BOT_TOKEN не найден ни в .env, ни в конфигурации!")

    logger.info("Используется токен: %s...", bot_token[:10])

    bot = Bot(token=bot_token)
    dp = Dispatcher()

    dp.include_router(chat_router)

    api_url = f"{cfg.api.domain}/api/v1/generate"

    async def start_polling() -> None:
        logger.info("Удаление старых вебхуков...")
        await bot.delete_webhook(drop_pending_updates=True)

        logger.info("Запуск локального бота в режиме Polling...")
        # Передаем данные через workflow_data диспетчера
        dp["cfg"] = cfg
        dp["api_url"] = api_url
        await dp.start_polling(bot)

    asyncio.run(start_polling())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
