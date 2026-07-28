# tests/training/test_module.py
"""
Тесты CausalLMLightningModule.
Используем tiny fake-модель — без скачивания весов.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fake causal LM модель
# ---------------------------------------------------------------------------
class FakeCausalOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class FakeCausalLM(nn.Module):
    """Минимальная замена GPT/Llama — два линейных слоя."""

    def __init__(self, vocab_size: int = 100, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.config = MagicMock()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embed(input_ids).mean(dim=1, keepdim=True)
        x = x.expand(-1, input_ids.shape[1], -1)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return FakeCausalOutput(loss=loss, logits=logits)


@pytest.fixture
def fake_causal_model():
    return FakeCausalLM()


@pytest.fixture
def causal_module(fake_causal_model):
    from src.training.module import CausalLMLightningModule

    optimizer_cfg = MagicMock()
    optimizer_cfg._target_ = "torch.optim.AdamW"
    module = CausalLMLightningModule(
        model=fake_causal_model,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=None,
    )
    return module


def _make_causal_batch(batch_size: int = 2, seq_len: int = 16, vocab_size: int = 100):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :4] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
class TestTrainingStep:
    def test_returns_scalar_loss(self, causal_module):
        causal_module.log = MagicMock()
        batch = _make_causal_batch()
        loss = causal_module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_is_positive(self, causal_module):
        causal_module.log = MagicMock()
        loss = causal_module.training_step(_make_causal_batch(), batch_idx=0)
        assert loss.item() > 0

    def test_loss_requires_grad(self, causal_module):
        causal_module.log = MagicMock()
        loss = causal_module.training_step(_make_causal_batch(), batch_idx=0)
        assert loss.requires_grad

    def test_calls_log(self, causal_module):
        causal_module.log = MagicMock()
        causal_module.training_step(_make_causal_batch(), batch_idx=0)
        causal_module.log.assert_called()

    def test_raises_without_labels(self, causal_module):
        causal_module.log = MagicMock()
        batch = _make_causal_batch()
        del batch["labels"]
        with pytest.raises((ValueError, Exception)):
            causal_module.training_step(batch, batch_idx=0)


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------
class TestValidationStep:
    def test_logs_val_loss(self, causal_module):
        causal_module.log = MagicMock()
        causal_module.validation_step(_make_causal_batch(), batch_idx=0)
        logged_keys = [call.args[0] for call in causal_module.log.call_args_list]
        assert "val_loss" in logged_keys

    def test_logs_val_perplexity(self, causal_module):
        causal_module.log = MagicMock()
        causal_module.validation_step(_make_causal_batch(), batch_idx=0)
        logged_keys = [call.args[0] for call in causal_module.log.call_args_list]
        assert "val_perplexity" in logged_keys

    def test_perplexity_is_exp_of_loss(self, causal_module):
        causal_module.log = MagicMock()
        causal_module.validation_step(_make_causal_batch(), batch_idx=0)
        logged = {call.args[0]: call.args[1] for call in causal_module.log.call_args_list}
        if "val_loss" in logged and "val_perplexity" in logged:
            expected_ppl = torch.exp(torch.as_tensor(logged["val_loss"])).item()
            assert abs(logged["val_perplexity"] - expected_ppl) < 1e-3


# ---------------------------------------------------------------------------
# Test step
# ---------------------------------------------------------------------------
class TestTestStep:
    def test_logs_test_loss_and_perplexity(self, causal_module):
        causal_module.log = MagicMock()
        causal_module.test_step(_make_causal_batch(), batch_idx=0)
        logged_keys = [call.args[0] for call in causal_module.log.call_args_list]
        assert "test_loss" in logged_keys
        assert "test_perplexity" in logged_keys


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------
class TestConfigureOptimizers:
    def test_returns_optimizer_without_scheduler(self, fake_causal_model):
        from unittest.mock import MagicMock, NonCallableMagicMock, patch

        import torch

        from src.training.module import CausalLMLightningModule

        opt_cfg = NonCallableMagicMock()  # ← не callable → пойдёт в ветку instantiate
        with patch("src.training.module.instantiate") as mock_inst:
            mock_inst.return_value = torch.optim.AdamW(fake_causal_model.parameters(), lr=1e-3)
            module = CausalLMLightningModule(
                model=fake_causal_model,
                optimizer_cfg=opt_cfg,
                scheduler_cfg=None,
            )
            module.trainer = MagicMock()
            result = module.configure_optimizers()

        assert isinstance(result, torch.optim.Optimizer)

    def test_returns_dict_with_scheduler(self, fake_causal_model):
        from unittest.mock import MagicMock, patch

        import torch

        from src.training.module import CausalLMLightningModule

        opt_cfg = MagicMock()
        sched_cfg = MagicMock()
        with patch("src.training.module.instantiate") as mock_inst:
            optimizer = torch.optim.AdamW(fake_causal_model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            mock_inst.side_effect = [optimizer, scheduler]
            module = CausalLMLightningModule(
                model=fake_causal_model,
                optimizer_cfg=opt_cfg,
                scheduler_cfg=sched_cfg,
            )

            module.trainer = MagicMock()
            result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_only_trainable_params_in_optimizer(self, fake_causal_model):
        from unittest.mock import MagicMock, patch

        import torch

        from src.training.module import CausalLMLightningModule

        for p in fake_causal_model.parameters():
            p.requires_grad = False
        captured = []
        with patch("src.training.module.instantiate") as mock_inst:

            def fake_instantiate(cfg, params):
                captured.extend(list(params))
                return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)

            mock_inst.side_effect = fake_instantiate
            module = CausalLMLightningModule(
                model=fake_causal_model,
                optimizer_cfg=MagicMock(),
                scheduler_cfg=None,
            )

            module.trainer = MagicMock()
            module.configure_optimizers()

        assert len(captured) == 0

    def test_returns_optimizer_with_callable_cfg(self, fake_causal_model):
        """Тестирует новую архитектуру с _partial_: true (callable config)."""
        import torch

        from src.training.module import CausalLMLightningModule

        # Имитируем поведение Hydra с _partial_: true
        def fake_partial_optimizer(params):
            return torch.optim.AdamW(params, lr=1e-5)

        module = CausalLMLightningModule(
            model=fake_causal_model,
            optimizer_cfg=fake_partial_optimizer,  # Передаем как функцию, а не конфиг
            scheduler_cfg=None,
        )

        module.trainer = MagicMock()
        result = module.configure_optimizers()

        assert isinstance(result, torch.optim.Optimizer)
        # Проверяем, что оптимизатор применил правильный lr
        assert result.param_groups[0]["lr"] == 1e-5
