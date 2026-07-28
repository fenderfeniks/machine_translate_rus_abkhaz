# src/core/data/schemas.py
from pydantic import BaseModel, Field, field_validator, model_validator


class RawDatasetRecord(BaseModel):
    """Контракт для сырой записи датасета перед токенизацией.

    Поддерживает как CPT (Continual Pre-Training) через поле text, 
    так и SFT (Supervised Fine-Tuning) через поля prompt и target.
    """

    text: str | None = Field(
        default=None, description="Сырой текст документа (для CPT)"
    )
    prompt: str | None = Field(
        default=None, description="Входной промпт для модели (для SFT)"
    )
    target: str | None = Field(
        default=None, description="Ожидаемый ответ (для SFT)"
    )

    @model_validator(mode="after")
    def validate_task_fields(self) -> "RawDatasetRecord":
        """Проверяет, что заполнены нужные поля для конкретной задачи."""
        if self.text is None and self.prompt is None:
            raise ValueError(
                "Должен быть заполнен либо 'text' (для CPT), либо 'prompt' (для SFT)"
            )
        return self

    @field_validator("text", "prompt")
    @classmethod
    def validate_content_length(cls, v: str | None) -> str | None:
        """Проверяет текстовые поля на пустоту и минимальную длину.

        Args:
            v: Входная строка.

        Returns:
            Очищенная от пробелов по краям строка или None.

        Raises:
            ValueError: Если текст состоит только из пробелов или короче 3 символов.
        """
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Поле ввода не может быть пустым")
            if len(v) < 3:
                raise ValueError("Текст слишком короткий (минимум 3 символа)")
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