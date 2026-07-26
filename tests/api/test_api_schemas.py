# tests/api/test_api_schemas.py
"""Тесты Pydantic-схем API."""

import pytest
from pydantic import ValidationError

from src.api.schemas import GenerationRequest, GenerationResponse


class TestGenerationRequest:
    def test_valid_prompt(self):
        req = GenerationRequest(prompt="Hello, what is AI?")
        assert req.prompt == "Hello, what is AI?"

    def test_empty_prompt_raises(self):
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="")

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            GenerationRequest()

    def test_numeric_prompt_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            GenerationRequest(prompt=123)

    def test_unicode_prompt_valid(self):
        req = GenerationRequest(prompt="Привет, как дела?")
        assert "Привет" in req.prompt


class TestGenerationResponse:
    def test_valid_response(self):
        resp = GenerationResponse(generated_text="This is the answer.")
        assert resp.generated_text == "This is the answer."

    def test_empty_generated_text_valid(self):
        """Пустой ответ модели технически допустим."""
        resp = GenerationResponse(generated_text="")
        assert resp.generated_text == ""
