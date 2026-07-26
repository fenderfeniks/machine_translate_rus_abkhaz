# tests/core/test_response_cleaner.py
"""
Тесты ResponseCleaner — постпроцессинг вывода LLM.
Индустриальный стандарт: декодеры генерируют мусор который нужно вырезать.
"""

import pytest

from src.core.models.response_cleaner import ResponseCleaner


@pytest.fixture
def cleaner():
    return ResponseCleaner()


class TestEchoPromptStripping:
    def test_strips_exact_prompt_prefix(self, cleaner):
        prompt = "What is Python?"
        response = "What is Python? Python is a programming language."
        result = cleaner.clean(raw_text=response, prompt=prompt)
        assert not result.startswith(prompt)
        assert "Python is a programming language" in result

    def test_no_prompt_passed_does_not_strip(self, cleaner):
        text = "Some generated text."
        assert cleaner.clean(raw_text=text) == text

    def test_prompt_not_prefix_is_kept(self, cleaner):
        result = cleaner.clean(raw_text="Answer: yes.", prompt="Question: ok?")
        assert "Answer: yes" in result


class TestSpecialTokenRemoval:
    def test_removes_llama_header_tags(self, cleaner):
        text = "<|start_header_id|>assistant<|end_header_id|>\\nHello!"
        result = cleaner.clean(raw_text=text)
        assert "<|start_header_id|>" not in result
        assert "<|end_header_id|>" not in result
        assert "Hello!" in result

    def test_removes_eos_token(self, cleaner):
        result = cleaner.clean(raw_text="Answer here</s>")
        assert "</s>" not in result
        assert "Answer here" in result

    def test_removes_bos_token(self, cleaner):
        result = cleaner.clean(raw_text="<s>Hello world")
        assert "<s>" not in result
        assert "Hello world" in result

    def test_removes_generic_special_tokens(self, cleaner):
        result = cleaner.clean(raw_text="Text<|eot_id|>more text")
        assert "<|eot_id|>" not in result


class TestMarkdownCleaning:
    def test_removes_code_block_markers(self):
        cleaner = ResponseCleaner(remove_markdown_blocks=True)
        text = "```python\nprint('hello')\n```"
        result = cleaner.clean(raw_text=text)
        assert "```" not in result
        assert "print" in result

    def test_markdown_disabled_keeps_backticks(self):
        cleaner = ResponseCleaner(remove_markdown_blocks=False)
        text = "```python\\ncode\\n```"
        result = cleaner.clean(raw_text=text)
        assert "```" in result


class TestIncompleteSentenceTrimming:
    def test_trims_incomplete_last_sentence(self, cleaner):
        text = "First sentence. Second sentence. Incomplete without punct"
        result = cleaner.clean(raw_text=text)
        assert result.endswith(".")
        assert "First sentence" in result

    def test_complete_sentence_not_trimmed(self, cleaner):
        text = "This is complete."
        result = cleaner.clean(raw_text=text)
        assert result == "This is complete."

    def test_trim_disabled_keeps_incomplete(self):
        cleaner = ResponseCleaner(trim_incomplete_sentence=False)
        text = "Complete. Incomplete without"
        result = cleaner.clean(raw_text=text)
        assert "Incomplete without" in result


class TestEdgeCases:
    def test_empty_string_returns_empty(self, cleaner):
        assert cleaner.clean(raw_text="") == ""

    def test_only_special_tokens_returns_empty_or_whitespace(self, cleaner):
        result = cleaner.clean(raw_text="</s><s><|eot_id|>")
        assert result.strip() == ""

    def test_unicode_preserved(self, cleaner):
        text = "Ответ модели на русском языке."
        result = cleaner.clean(raw_text=text)
        assert "Ответ модели" in result

    @pytest.mark.parametrize(
        "text",
        [
            "Normal answer.",
            "",
            "   ",
            "<s>token soup</s>",
            "Multi.\\nLine\\nText.",
        ],
    )
    def test_does_not_raise_on_various_inputs(self, cleaner, text):
        result = cleaner.clean(raw_text=text)
        assert isinstance(result, str)
