# src/api/tg_bot/keyboards/reply.py
"""Reply-клавиатуры (нижние кнопки)."""

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Создает главную клавиатуру бота с кнопкой /start.

    Returns:
        Инициализированный объект ReplyKeyboardMarkup.
    """
    keyboard = [
        [KeyboardButton(text="/start")],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        input_field_placeholder="Отправьте промпт для генерации...",
    )
