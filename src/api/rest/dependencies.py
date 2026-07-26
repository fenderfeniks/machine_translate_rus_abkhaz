# src/api/rest/dependencies.py
from fastapi import HTTPException, Request

from src.core.prompts.manager import PromptManager
from src.sdk.inference import LLMGenerationPipeline


def get_prompt_manager(request: Request) -> PromptManager:
    """Извлекает менеджер промптов из глобального состояния приложения.

    Args:
        request: Входящий запрос FastAPI.

    Returns:
        Инициализированный объект PromptManager.
    """
    return request.app.state.prompt_manager


def get_generator(request: Request) -> LLMGenerationPipeline:
    """Извлекает пайплайн генерации (LLM) из глобального состояния.

    Args:
        request: Входящий запрос FastAPI.

    Returns:
        Пайплайн генерации текста.

    Raises:
        HTTPException: Если модель не была загружена в память при старте сервера.
    """
    generator = request.app.state.ml_models.get("generator")
    if not generator:
        raise HTTPException(status_code=503, detail="Модель генерации еще не загружена в память.")
    return generator
