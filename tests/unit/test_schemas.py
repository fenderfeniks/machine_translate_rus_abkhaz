# tests/unit/test_schemas.py
import pytest
from pydantic import ValidationError

from src.api.schemas import GenerationRequest, GenerationResponse


class TestGenerationRequest:
    def test_valid_prompt_accepted(self) -> None:
        """Проверка валидного промпта в пределах допустимой длины."""
        req = GenerationRequest(prompt="Объясни принцип работы RAG.")
        assert req.prompt == "Объясни принцип работы RAG."

    def test_missing_prompt_rejects(self) -> None:
        """Схема падает, если не передать обязательное поле prompt."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest()
        assert "Field required" in str(exc_info.value)

    def test_empty_prompt_rejects(self) -> None:
        """Схема падает, если промпт пустой (срабатывает min_length=1)."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="")

    def test_oversized_prompt_rejects(self) -> None:
        """Схема падает, если промпт превышает 2000 символов (max_length=2000)."""
        long_prompt = "A" * 2001
        with pytest.raises(ValidationError):
            GenerationRequest(prompt=long_prompt)

    def test_wrong_types_rejects(self) -> None:
        """Схема не принимает числа или списки вместо строки."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt=["Это", "список", "а", "не", "строка"])

        with pytest.raises(ValidationError):
            GenerationRequest(prompt=12345)


class TestGenerationResponse:
    def test_valid_response(self) -> None:
        """Проверка валидного ответа модели."""
        resp = GenerationResponse(generated_text="Сгенерированный текст.")
        assert resp.generated_text == "Сгенерированный текст."
