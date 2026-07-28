# tests/training/test_callbacks.py
"""Тесты пользовательских коллбэков PyTorch Lightning."""

from src.training.callbacks import _MODE_CPT, _MODE_SFT, GenerationEvaluationCallback


class TestGenerationEvaluationCallback:
    def test_resolve_mode_auto_resolves_to_sft_when_prompt_column_exists(self) -> None:
        """Режим 'auto' должен определять SFT, если в конфиге есть prompt_column."""
        cb = GenerationEvaluationCallback(model_name="test_model", mode="auto")
        data_cfg = {"prompt_column": "prompt", "target_column": "target"}

        resolved = cb._resolve_mode(data_cfg)
        assert resolved == _MODE_SFT

    def test_resolve_mode_auto_resolves_to_cpt_when_no_prompt_column(self) -> None:
        """Режим 'auto' должен определять CPT, если prompt_column отсутствует."""
        cb = GenerationEvaluationCallback(model_name="test_model", mode="auto")
        data_cfg = {"text_column": "text"}

        resolved = cb._resolve_mode(data_cfg)
        assert resolved == _MODE_CPT

    def test_resolve_mode_respects_explicit_mode(self) -> None:
        """Явно заданный режим должен игнорировать структуру конфига данных."""
        cb = GenerationEvaluationCallback(model_name="test_model", mode=_MODE_CPT)
        data_cfg = {"prompt_column": "prompt"}  # Несмотря на наличие prompt_column

        resolved = cb._resolve_mode(data_cfg)
        assert resolved == _MODE_CPT

    def test_extract_rouge_score_handles_float(self) -> None:
        cb = GenerationEvaluationCallback(model_name="test_model")
        assert cb._extract_rouge_score(0.85) == 0.85

    def test_extract_rouge_score_handles_list(self) -> None:
        cb = GenerationEvaluationCallback(model_name="test_model")
        assert cb._extract_rouge_score([0.90]) == 0.90
