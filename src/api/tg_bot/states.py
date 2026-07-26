# src/api/tg_bot/states.py
"""Машина состояний (FSM) для Telegram-бота."""

from aiogram.fsm.state import State, StatesGroup


class ChatProcess(StatesGroup):
    """Базовые состояния пользователя в боте."""

    chatting = State()  # Обычный режим общения (отправка запросов)
