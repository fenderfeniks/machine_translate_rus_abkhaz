# src/api/rest/server.py
import asyncio
import gc
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from aiogram import types
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.rest.endpoints import generate, health
from src.api.rest.limiter import limiter
from src.api.rest.middlewares import setup_middlewares
from src.api.tg_bot.bot_webhook import dp, get_webhook_bot
from src.core.prompts.manager import PromptManager
from src.sdk.inference import LLMGenerationPipeline


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Проверяет наличие и валидность API ключа в заголовке запроса.

    Args:
        api_key: Ключ из заголовка X-API-Key.

    Returns:
        Провалидированный API ключ.

    Raises:
        HTTPException: Если ключ отсутствует или неверен.
    """
    expected_key = os.getenv("API_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key


def create_app(load_ml: bool = True) -> FastAPI:
    """Фабрика для создания инстанса FastAPI.

    Args:
        load_ml: Флаг загрузки тяжелых ML-моделей в память.
            Полезно отключать для тестов или легковесных воркеров.

    Returns:
        Сконфигурированное приложение FastAPI.
    """
    load_dotenv()
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ml_models = {}
        concurrency_limit = cfg.api.get("concurrency_limit", 1)
        app.state.gpu_semaphore = asyncio.Semaphore(concurrency_limit)

        # Передаем промпты напрямую из конфига Hydra (cfg.prompts)
        app.state.prompt_manager = PromptManager(templates=cfg.get("prompts", {}))

        if load_ml:
            logger.info("Загрузка LLM в видеопамять...")
            try:
                # Инициализация генератора
                generator = LLMGenerationPipeline(config_name="main")
                app.state.ml_models["generator"] = generator
                logger.info("LLM успешно загружена.")
            except Exception as e:
                logger.warning("Не удалось загрузить LLM: %s. API запустится без неё.", e)
                app.state.ml_models["generator"] = None

            bot_token = os.getenv("TG_BOT_TOKEN") or cfg.api.telegram.bot_token
            if bot_token:
                bot = get_webhook_bot(bot_token)
                app.state.tg_bot = bot
                try:
                    webhook_url = cfg.api.telegram.webhook_url
                    await bot.set_webhook(url=webhook_url, drop_pending_updates=True)
                except Exception as e:
                    logger.warning("Не удалось установить вебхук: %s", e)

        yield

        if load_ml:
            if "tg_bot" in app.state:
                await app.state.tg_bot.delete_webhook()
                await app.state.tg_bot.session.close()

            app.state.ml_models.clear()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    app = FastAPI(
        title=cfg.api.title,
        description=cfg.api.description,
        version=cfg.api.version,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Сохраняем полный конфиг Hydra в стейт приложения
    app.state.config = cfg
    setup_middlewares(app, cors_origins=list(cfg.api.cors_origins))

    app.include_router(health.router)
    # Регистрируем новый роутер генерации
    app.include_router(generate.router, dependencies=[Depends(verify_api_key)])

    Instrumentator(should_group_status_codes=False, should_ignore_untemplated=True).instrument(
        app
    ).expose(app, include_in_schema=False, endpoint="/metrics")

    @app.post(cfg.api.telegram_webhook.path, include_in_schema=False)
    async def telegram_webhook_endpoint(update: dict) -> dict[str, str]:
        bot = app.state.tg_bot
        if not bot:
            raise HTTPException(status_code=503, detail="Telegram bot service is unavailable")
        await dp.feed_update(
            bot,
            update=types.Update(**update),
            cfg=cfg,
            generator=app.state.ml_models.get("generator"),
        )
        return {"status": "ok"}

    return app


app = create_app(load_ml=True)
