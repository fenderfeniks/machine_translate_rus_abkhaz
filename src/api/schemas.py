# src/api/schemas.py
"""Схемы данных Pydantic для REST API.

Используются для строгой типизации и валидации входящих
и исходящих JSON-сообщений в FastAPI.
"""

from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    """Модель входящего запроса на генерацию текста."""

    prompt: str = Field(
        ..., description="Входящий текст (промпт) для LLM.", min_length=1, max_length=2000
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"prompt": "Объясни, что такое Retrieval-Augmented Generation (RAG)."}
        }
    )


class GenerationResponse(BaseModel):
    """Модель исходящего ответа с результатами генерации."""

    generated_text: str = Field(
        ..., description="Сгенерированный моделью текст (без исходного промпта)."
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"generated_text": "RAG — это архитектура, которая..."}}
    )
