# src/api/tg_bot/handlers/chat.py
import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import Any

import aiohttp
from aiogram import F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from omegaconf import DictConfig

# Импортируем метрики
from src.api.metrics import (
    LLM_GENERATED_TOKENS_TOTAL,
    LLM_GENERATION_REQUESTS_TOTAL,
    LLM_GENERATION_TIME,
)
from src.api.tg_bot.keyboards.reply import get_main_keyboard
from src.api.tg_bot.states import ChatProcess
from src.core.prompts.manager import PromptManager


logger = logging.getLogger(__name__)
router = Router()


@router.message(CommandStart())
async def cmd_start(
    message: types.Message, state: FSMContext, cfg: DictConfig, prompt_manager: PromptManager
) -> None:
    """Обрабатывает команду /start, приветствуя пользователя."""
    await state.set_state(ChatProcess.chatting)
    logger.info("Получена команда /start от пользователя.")

    user_name = message.from_user.full_name if message.from_user else "Пользователь"

    welcome_text = prompt_manager.render("telegram_welcome", username=user_name)
    await message.answer(text=welcome_text, reply_markup=get_main_keyboard())


@router.message(ChatProcess.chatting, F.text)
async def process_chat_message(
    message: types.Message,
    cfg: DictConfig,
    generator: Any | None = None,
    api_url: str | None = None,
    prompt_manager: PromptManager | None = None,
) -> None:
    """Обрабатывает текстовые запросы пользователя и возвращает генерацию."""
    logger.info("Получено текстовое сообщение: %s", message.text)

    if not prompt_manager:
        logger.error("prompt_manager не передан в хэндлер.")
        await message.answer("Ошибка конфигурации: prompt_manager недоступен.")
        return

    # 1. Мгновенный ответ пользователю
    processing_msg = await message.answer("✨ Генерирую ответ, подождите немного...")

    # 2. Рендерим промпт через наш менеджер (подставляем текст пользователя)
    final_prompt = prompt_manager.render(template_name="rag_qa", question=message.text, context="")

    # 3. Фоновая задача для генерации
    async def _background_generate() -> None:
        try:
            start_time = time.perf_counter()
            answer = ""

            # --- СЦЕНАРИЙ А: ПРОДАКШЕН (Webhooks со стримингом) ---
            if generator:
                LLM_GENERATION_REQUESTS_TOTAL.labels(source="tg").inc()

                with LLM_GENERATION_TIME.labels(source="tg").time():
                    # Запускаем стриминг
                    stream_iter = generator.generate_stream(final_prompt)
                    last_edit_time = time.time()
                    edit_interval = 1.5  # Редактируем сообщение не чаще раза в 1.5 сек

                    while True:
                        try:
                            # Достаем следующий кусок текста из фонового потока
                            chunk = await asyncio.to_thread(next, stream_iter)
                            answer += chunk
                            current_time = time.time()

                            # Обновляем UI с анимацией загрузки
                            if current_time - last_edit_time > edit_interval:
                                with suppress(TelegramBadRequest):
                                    # Парс-мод отключен, чтобы незакрытые теги (например, **текст) не ломали API
                                    await processing_msg.edit_text(f"{answer} ⏳")
                                last_edit_time = current_time

                        except StopIteration:
                            break  # Токены закончились

                # Подсчет токенов для телеметрии
                approx_tokens = len(answer.split()) * 1.3
                LLM_GENERATED_TOKENS_TOTAL.labels(source="tg").inc(approx_tokens)

            # --- СЦЕНАРИЙ Б: ЛОКАЛЬНАЯ РАЗРАБОТКА (Polling через HTTP) ---
            elif api_url:
                payload = {"prompt": final_prompt}
                headers = {"X-API-Key": os.getenv("API_KEY", "")}
                timeout = aiohttp.ClientTimeout(total=300)

                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.post(api_url, json=payload, headers=headers) as resp,
                ):
                    if resp.status == 200:
                        data = await resp.json()
                        answer = data.get("generated_text", "Пустой ответ от модели.")
                    else:
                        answer = f"Ошибка API: HTTP {resp.status}"
            else:
                raise ValueError("Не передан ни generator, ни api_url!")

            # 4. Финальное обновление сообщения с метрикой времени
            elapsed = time.perf_counter() - start_time
            answer_with_telemetry = f"{answer}\n\n_⏱ Сгенерировано за {elapsed:.1f} сек._"

            with suppress(TelegramBadRequest):
                # В конце включаем Markdown, когда текст уже сформирован целиком
                await processing_msg.edit_text(answer_with_telemetry, parse_mode="Markdown")

        except Exception as e:
            logger.error("Ошибка в фоновой генерации ТГ: %s", e)
            with suppress(TelegramBadRequest):
                # Фолбек сообщение об ошибке
                error_msg = cfg.api.telegram.messages.get(
                    "error", "Произошла ошибка при генерации."
                )
                await processing_msg.edit_text(error_msg)

    # Запускаем генерацию в фоне и сразу завершаем хэндлер
    asyncio.create_task(_background_generate())
