# tests/training/test_callbacks.py
"""Тесты пользовательских коллбэков PyTorch Lightning."""

from unittest.mock import MagicMock

import pytest
from torch import nn

from src.training.callbacks import ModelFreezingCallback


class DummyModel(nn.Module):
    """Фиктивная модель для тестирования заморозки."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 10)
        self.encoder_layer = nn.Linear(10, 10)
        self.lm_head = nn.Linear(10, 10)


@pytest.fixture
def mock_trainer_and_module() -> tuple[MagicMock, MagicMock]:
    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.model = DummyModel()
    return trainer, pl_module


class TestModelFreezingCallback:
    def test_lm_head_only_freezes_everything_else(
        self, mock_trainer_and_module: tuple[MagicMock, MagicMock]
    ) -> None:
        trainer, pl_module = mock_trainer_and_module
        cb = ModelFreezingCallback(finetuning_type="lm_head_only")
        cb.setup(trainer, pl_module, stage="fit")

        model = pl_module.model
        assert not model.embed_tokens.weight.requires_grad
        assert not model.encoder_layer.weight.requires_grad
        assert model.lm_head.weight.requires_grad

    def test_frozen_embeddings_freezes_embed_and_head(
        self, mock_trainer_and_module: tuple[MagicMock, MagicMock]
    ) -> None:
        trainer, pl_module = mock_trainer_and_module
        cb = ModelFreezingCallback(finetuning_type="frozen_embeddings")
        cb.setup(trainer, pl_module, stage="fit")

        model = pl_module.model
        assert not model.embed_tokens.weight.requires_grad
        assert model.encoder_layer.weight.requires_grad
        assert not model.lm_head.weight.requires_grad

    def test_full_finetuning_keeps_all_trainable(
        self, mock_trainer_and_module: tuple[MagicMock, MagicMock]
    ) -> None:
        trainer, pl_module = mock_trainer_and_module
        cb = ModelFreezingCallback(finetuning_type="full")
        cb.setup(trainer, pl_module, stage="fit")

        model = pl_module.model
        assert model.embed_tokens.weight.requires_grad
        assert model.encoder_layer.weight.requires_grad
        assert model.lm_head.weight.requires_grad

    def test_invalid_finetuning_type_raises_error(
        self, mock_trainer_and_module: tuple[MagicMock, MagicMock]
    ) -> None:
        trainer, pl_module = mock_trainer_and_module
        cb = ModelFreezingCallback(finetuning_type="unknown_mode")
        with pytest.raises(ValueError, match="Неизвестный режим"):
            cb.setup(trainer, pl_module, stage="fit")
