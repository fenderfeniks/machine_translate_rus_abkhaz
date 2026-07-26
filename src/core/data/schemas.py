# src/core/data/schemas.py
from pydantic import BaseModel, Field, field_validator


class RawDatasetRecord(BaseModel):
    """Контракт для сырой записи датасета перед токенизацией.

    Используется при подготовке обучающей выборки для строгой 
    типизации и валидации входящих данных.
    """

    prompt: str = Field(..., description="Входной промпт для модели")
    target: str | None = Field(
        default=None, description="Ожидаемый ответ (для Fine-Tuning)"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_length(cls, v: str) -> str:
        """Проверяет промпт на пустоту и минимальную длину.

        Args:
            v: Входная строка промпта.

        Returns:
            Очищенная от пробелов по краям строка.

        Raises:
            ValueError: Если промпт пустой или короче 3 символов.
        """
        v = v.strip()
        if not v:
            raise ValueError("Промпт не может быть пустым")
        if len(v) < 3:
            raise ValueError("Промпт слишком короткий (минимум 3 символа)")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str | None) -> str | None:
        """Проверяет целевой ответ на пустоту, если он передан.

        Args:
            v: Входная строка таргета.

        Returns:
            Очищенная от пробелов по краям строка или None.

        Raises:
            ValueError: Если таргет передан, но является пустой строкой.
        """
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Target передан, но является пустой строкой")
        return v