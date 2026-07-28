# tests/core/test_transforms.py
"""
Тесты трансформаций датасета.
Используем синтетические HF-датасеты — без скачивания данных.
"""

import pytest
from datasets import Dataset


def _make_hf_dataset(records: list[dict]) -> Dataset:
    keys = records[0].keys()
    return Dataset.from_dict({k: [r[k] for r in records] for k in keys})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def null_pipeline():
    """Пустой pipeline-заглушка для ValidationTransform (валидация через Pydantic,
    очистка не нужна)."""
    from src.core.data.cleaners import TextCleaningPipeline

    return TextCleaningPipeline(cleaners=[])


@pytest.fixture()
def html_strip_pipeline():
    """Pipeline, удаляющий HTML-теги — для тестов CleaningTransform."""
    from src.core.data.cleaners import RegexCleaner, TextCleaningPipeline

    return TextCleaningPipeline(cleaners=[RegexCleaner(pattern="<.*?>", replacement="")])


@pytest.fixture()
def newline_escape_pipeline():
    """Pipeline, заменяющий '\\n' на пробел — для тестов CleaningTransform."""
    from src.core.data.cleaners import RegexCleaner, TextCleaningPipeline

    return TextCleaningPipeline(cleaners=[RegexCleaner(pattern="\\\\n", replacement=" ")])


# ---------------------------------------------------------------------------
# TestValidationTransform
# ---------------------------------------------------------------------------


class TestValidationTransform:
    def test_filters_empty_prompts(self, null_pipeline):
        from src.core.data.transforms.validation import ValidationTransform

        ds = _make_hf_dataset(
            [
                {"prompt": "Valid prompt", "target": "Valid target"},
                {"prompt": "", "target": "Target"},
                {"prompt": "Another valid", "target": "ok"},
            ]
        )
        transform = ValidationTransform(pipeline=null_pipeline, num_proc=1, batch_size=10)
        result = transform(ds)
        assert len(result) == 2
        assert all(r["prompt"] for r in result)

    def test_filters_short_prompts(self, null_pipeline):
        from src.core.data.transforms.validation import ValidationTransform

        ds = _make_hf_dataset(
            [
                {"prompt": "ab", "target": "ok"},  # слишком короткий
                {"prompt": "Valid prompt text", "target": "ok"},
            ]
        )
        transform = ValidationTransform(pipeline=null_pipeline, num_proc=1, batch_size=10)
        result = transform(ds)
        assert len(result) == 1

    def test_cpt_mode_validates_text_column(self, null_pipeline):
        from src.core.data.transforms.validation import ValidationTransform

        ds = _make_hf_dataset(
            [
                {"text": "Valid text for CPT training"},
                {"text": ""},
                {"text": "Another valid text"},
            ]
        )
        transform = ValidationTransform(pipeline=null_pipeline, num_proc=1, batch_size=10)
        result = transform(ds)
        assert len(result) == 2

    def test_raises_without_required_columns(self, null_pipeline):
        from src.core.data.transforms.validation import ValidationTransform

        ds = _make_hf_dataset([{"unknown_col": "data"}])
        transform = ValidationTransform(pipeline=null_pipeline, num_proc=1, batch_size=10)
        with pytest.raises(ValueError, match="ValidationTransform"):
            transform(ds)


# ---------------------------------------------------------------------------
# TestLengthFilterTransform
# ---------------------------------------------------------------------------


class TestLengthFilterTransform:
    def test_removes_long_sequences(self, tiny_tokenizer):
        from src.core.data.transforms.filtering import LengthFilterTransform
        from src.core.data.transforms.tokenization import TokenizationTransform

        ds = _make_hf_dataset(
            [
                {"text": "short"},
                {"text": "word " * 1000},
            ]
        )
        tok = TokenizationTransform(tokenizer=tiny_tokenizer, text_column="text", num_proc=1)
        ds_tokenized = tok(ds)
        length_filter = LengthFilterTransform(max_length=50, num_proc=1)
        result = length_filter(ds_tokenized)
        assert len(result) == 1
        assert all(len(r["input_ids"]) <= 50 for r in result)


# ---------------------------------------------------------------------------
# TestTokenizationTransform
# ---------------------------------------------------------------------------


class TestTokenizationTransform:
    def test_text_column_mode(self, tiny_tokenizer):
        from src.core.data.transforms.tokenization import TokenizationTransform

        ds = _make_hf_dataset([{"text": "Hello world"}, {"text": "Test input"}])
        transform = TokenizationTransform(
            tokenizer=tiny_tokenizer,
            text_column="text",
            num_proc=1,
        )
        result = transform(ds)
        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names
        assert len(result) == 2

    def test_prompt_response_mode_adds_prompt_len(self, tiny_tokenizer):
        from src.core.data.transforms.tokenization import TokenizationTransform

        ds = _make_hf_dataset(
            [
                {"prompt": "Question:", "response": "Answer."},
            ]
        )
        transform = TokenizationTransform(
            tokenizer=tiny_tokenizer,
            prompt_column="prompt",
            target_column="response",
            num_proc=1,
        )
        result = transform(ds)
        assert "prompt_len" in result.column_names
        assert result[0]["prompt_len"] > 0

    def test_removes_original_columns(self, tiny_tokenizer):
        from src.core.data.transforms.tokenization import TokenizationTransform

        ds = _make_hf_dataset([{"text": "Hello"}, {"text": "World"}])
        transform = TokenizationTransform(tokenizer=tiny_tokenizer, text_column="text", num_proc=1)
        result = transform(ds)
        assert "text" not in result.column_names


# ---------------------------------------------------------------------------
# TestSequencePackingTransform
# ---------------------------------------------------------------------------


class TestSequencePackingTransform:
    def test_packing_creates_equal_length_chunks(self, tiny_tokenizer):
        from src.core.data.transforms.packing import SequencePackingTransform
        from src.core.data.transforms.tokenization import TokenizationTransform

        ds = _make_hf_dataset([{"text": "word " * 20}] * 10)
        tok = TokenizationTransform(tokenizer=tiny_tokenizer, text_column="text", num_proc=1)
        ds_tok = tok(ds)
        packing = SequencePackingTransform(packing_chunk_size=32, drop_remainder=True, num_proc=1)
        result = packing(ds_tok)
        assert all(len(r["input_ids"]) == 32 for r in result)

    def test_drop_remainder_true_drops_incomplete_chunk(self, tiny_tokenizer):
        from src.core.data.transforms.packing import SequencePackingTransform
        from src.core.data.transforms.tokenization import TokenizationTransform

        ds = _make_hf_dataset([{"text": "word " * 5}] * 3)
        tok = TokenizationTransform(tokenizer=tiny_tokenizer, text_column="text", num_proc=1)
        ds_tok = tok(ds)
        total_tokens = sum(len(r["input_ids"]) for r in ds_tok)
        chunk_size = 32
        expected_chunks = total_tokens // chunk_size
        packing = SequencePackingTransform(
            packing_chunk_size=chunk_size, drop_remainder=True, num_proc=1
        )
        result = packing(ds_tok)
        assert len(result) == expected_chunks


# ---------------------------------------------------------------------------
# TestCleaningTransform
# ---------------------------------------------------------------------------


class TestCleaningTransform:
    def test_cleans_prompt_target_columns(self, html_strip_pipeline):
        # CleaningTransform живёт в validation.py
        from src.core.data.transforms.validation import CleaningTransform

        ds = _make_hf_dataset(
            [
                {"prompt": "<b>Вопрос</b>", "target": "<i>Ответ</i>"},
            ]
        )
        transform = CleaningTransform(
            pipeline=html_strip_pipeline,
            prompt_column="prompt",
            target_column="target",
            num_proc=1,
        )
        result = transform(ds)
        assert result[0]["prompt"] == "Вопрос"
        assert result[0]["target"] == "Ответ"

    def test_cleans_text_column(self, newline_escape_pipeline):
        # CleaningTransform живёт в validation.py (не в filtering.py)
        from src.core.data.transforms.validation import CleaningTransform

        ds = _make_hf_dataset([{"text": "Текст\\nс\\nпереносами"}])
        transform = CleaningTransform(
            pipeline=newline_escape_pipeline,
            text_column="text",
            num_proc=1,
        )
        result = transform(ds)
        assert result[0]["text"] == "Текст с переносами"
