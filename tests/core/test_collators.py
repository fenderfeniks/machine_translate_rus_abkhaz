# tests/core/test_collators.py
"""
Тесты InstructionDataCollator.
Используем реальный gpt2-токенизатор из conftest (tiny_tokenizer).
"""

import pytest
import torch

from src.core.data.collators import InstructionDataCollator


@pytest.fixture
def collator(tiny_tokenizer):
    return InstructionDataCollator(
        tokenizer=tiny_tokenizer,
        max_sequence_length=64,
        mask_prompt=True,
    )


class TestInstructionDataCollator:
    def test_returns_input_ids_and_attention_mask(self, collator, tiny_tokenizer):
        ids = tiny_tokenizer("Hello world")["input_ids"]
        batch = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
        result = collator(batch)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_returns_labels(self, collator, tiny_tokenizer):
        ids = tiny_tokenizer("Hello world")["input_ids"]
        batch = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
        result = collator(batch)
        assert "labels" in result

    def test_padding_tokens_masked_in_labels(self, collator, tiny_tokenizer):
        """Паддинг-токены должны быть замаскированы (-100) в labels."""
        short_ids = tiny_tokenizer("Hi")["input_ids"]
        long_ids = tiny_tokenizer("This is a much longer sentence")["input_ids"]
        batch = [
            {"input_ids": short_ids, "attention_mask": [1] * len(short_ids)},
            {"input_ids": long_ids, "attention_mask": [1] * len(long_ids)},
        ]
        result = collator(batch)
        labels_row = result["labels"][0]
        pad_positions = (result["attention_mask"][0] == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            assert all(labels_row[i] == -100 for i in pad_positions)

    def test_prompt_len_masking(self, tiny_tokenizer):
        """Если передан prompt_len — токены промпта маскируются в labels."""
        collator = InstructionDataCollator(
            tokenizer=tiny_tokenizer,
            max_sequence_length=64,
            mask_prompt=True,
        )
        ids = tiny_tokenizer("prompt response text")["input_ids"]
        prompt_len = 2
        batch = [
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "prompt_len": prompt_len,
            }
        ]
        result = collator(batch)
        assert all(result["labels"][0, :prompt_len] == -100)
        assert result["labels"][0, prompt_len] != -100

    def test_batch_size_preserved(self, collator, tiny_tokenizer):
        ids = tiny_tokenizer("test")["input_ids"]
        batch = [{"input_ids": ids, "attention_mask": [1] * len(ids)}] * 3
        result = collator(batch)
        assert result["input_ids"].shape[0] == 3

    def test_output_dtype_is_long(self, collator, tiny_tokenizer):
        ids = tiny_tokenizer("test")["input_ids"]
        batch = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
        result = collator(batch)
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long

    def test_mask_prompt_false_does_not_mask_labels(self, tiny_tokenizer):
        """При mask_prompt=False labels совпадают с input_ids (кроме паддинга)."""
        collator = InstructionDataCollator(
            tokenizer=tiny_tokenizer,
            max_sequence_length=64,
            mask_prompt=False,
        )
        ids = tiny_tokenizer("Hello world response")["input_ids"]
        batch = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
        result = collator(batch)
        mask = result["attention_mask"][0].bool()
        assert torch.equal(result["labels"][0][mask], result["input_ids"][0][mask])
