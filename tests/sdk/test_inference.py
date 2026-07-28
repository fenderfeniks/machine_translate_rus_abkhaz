# tests/sdk/test_inference.py
"""
Тесты LLMGenerationPipeline и HFTextGenerator.
Все тяжёлые зависимости мокируются — тесты быстрые, без GPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_generated_ids(batch_size: int = 1, total_len: int = 10):
    """Имитирует вывод model.generate() — [input_ids + generated_ids]."""
    return torch.randint(0, 100, (batch_size, total_len))


def _make_text_generator(generated_texts: list[str]):
    """Создаёт HFTextGenerator с замоканной моделью и токенизатором."""
    # ИСПРАВЛЕНИЕ: Добавлен префикс src.
    from src.core.inference.generator import HFTextGenerator

    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "left"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    encoded = MagicMock()
    encoded.to.return_value = encoded
    encoded.__getitem__ = lambda self, k: torch.ones(1, 4, dtype=torch.long)
    encoded.__iter__ = lambda self: iter({})
    encoded.__getitem__ = MagicMock(return_value=torch.ones(1, 4, dtype=torch.long))
    input_ids_mock = torch.ones(1, 4, dtype=torch.long)
    encoded.__class__.__getitem__ = lambda self, k: input_ids_mock
    mock_tokenizer.return_value = encoded

    mock_model = MagicMock()
    mock_model.generate.return_value = _make_mock_generated_ids(1, 8)
    mock_model.parameters.return_value = iter([torch.zeros(1)])

    mock_tokenizer.batch_decode.return_value = generated_texts

    generator = object.__new__(HFTextGenerator)
    generator.model = mock_model
    generator.tokenizer = mock_tokenizer
    generator.generation_kwargs = {"max_new_tokens": 50}

    # ИСПРАВЛЕНИЕ: Добавлен префикс src.
    from src.core.inference.response_cleaner import ResponseCleaner

    generator.cleaner = ResponseCleaner(
        trim_incomplete_sentence=False,
        remove_markdown_blocks=False,
    )
    return generator


# ---------------------------------------------------------------------------
# HFTextGenerator
# ---------------------------------------------------------------------------
class TestHFTextGenerator:
    def test_single_string_wrapped_in_list(self):
        generator = _make_text_generator(["Generated response."])
        with patch.object(generator.model, "generate", return_value=_make_mock_generated_ids(1, 8)):
            with patch.object(
                generator.tokenizer, "batch_decode", return_value=["Generated response."]
            ):
                result = generator.generate("Single prompt")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_batch_returns_correct_count(self):
        generator = _make_text_generator(["R1", "R2", "R3"])
        with patch.object(generator.model, "generate", return_value=_make_mock_generated_ids(3, 8)):
            with patch.object(generator.tokenizer, "batch_decode", return_value=["R1", "R2", "R3"]):
                result = generator.generate(["P1", "P2", "P3"])
        assert len(result) == 3

    def test_kwargs_override_generation_params(self):
        generator = _make_text_generator(["response"])
        with patch.object(
            generator.model, "generate", return_value=_make_mock_generated_ids(1, 8)
        ) as mock_gen:
            with patch.object(generator.tokenizer, "batch_decode", return_value=["response"]):
                generator.generate("prompt", max_new_tokens=100)
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("max_new_tokens") == 100

    def test_result_is_list_of_strings(self):
        generator = _make_text_generator(["Clean output."])
        with patch.object(generator.model, "generate", return_value=_make_mock_generated_ids(1, 8)):
            with patch.object(generator.tokenizer, "batch_decode", return_value=["Clean output."]):
                result = generator.generate("prompt")
        assert all(isinstance(r, str) for r in result)


# ---------------------------------------------------------------------------
# Генеративные метрики
# ---------------------------------------------------------------------------
class TestGenerativeMetrics:
    def test_perplexity_from_loss(self):
        loss = torch.tensor(2.0)
        perplexity = torch.exp(loss)
        assert abs(perplexity.item() - 7.389) < 0.01

    def test_perplexity_zero_loss_is_one(self):
        assert abs(torch.exp(torch.tensor(0.0)).item() - 1.0) < 1e-6

    def test_perplexity_overflow_handled(self):
        loss = torch.tensor(1000.0)
        try:
            ppl = torch.exp(loss)
            assert ppl.item() == float("inf") or ppl.item() > 1e10
        except OverflowError:
            pass

    @pytest.mark.parametrize(
        "text,expected_len",
        [
            ("Hello world", 2),
            ("One two three four five", 5),
            ("", 0),
        ],
    )
    def test_avg_generation_length_calculation(self, text, expected_len):
        texts = [text]
        avg_len = sum(len(t.split()) for t in texts) / len(texts)
        assert avg_len == expected_len

    def test_rouge_scores_are_between_0_and_1(self):
        pytest.importorskip("evaluate", reason="evaluate not installed")
        import evaluate

        rouge = evaluate.load("rouge")
        result = rouge.compute(
            predictions=["The cat sat on the mat"],
            references=["The cat sat on the mat"],
            use_stemmer=True,
        )
        for key in ["rouge1", "rouge2", "rougeL"]:
            score = float(result[key])
            assert 0.0 <= score <= 1.0

    def test_rouge_identical_texts_gives_1(self):
        pytest.importorskip("evaluate", reason="evaluate not installed")
        import evaluate

        rouge = evaluate.load("rouge")
        result = rouge.compute(
            predictions=["exact match text"],
            references=["exact match text"],
            use_stemmer=True,
        )
        assert float(result["rouge1"]) == pytest.approx(1.0, abs=1e-3)

    def test_rouge_unrelated_texts_gives_low_score(self):
        pytest.importorskip("evaluate", reason="evaluate not installed")
        import evaluate

        rouge = evaluate.load("rouge")
        result = rouge.compute(
            predictions=["quantum physics equations"],
            references=["banana apple orange fruit"],
            use_stemmer=True,
        )
        assert float(result["rouge1"]) < 0.3
