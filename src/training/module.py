# src/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class CausalLMLightningModule(pl.LightningModule):
    """Чистый LightningModule для обучения Causal LM.

    Поддерживает динамическое сохранение: для LoRA сохраняет только адаптеры,
    для Full Fine-Tuning сохраняет все обучаемые веса.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
        task_mode: str = "cpt",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.task_mode = task_mode

        # Маршрутизация метрик: избегаем if/else на каждом шаге
        if self.task_mode == "cpt":
            self._compute_extra_metrics = self._log_perplexity
        else:
            self._compute_extra_metrics = lambda loss, phase: None

        self.save_hyperparameters(ignore=["model", "_compute_extra_metrics"])

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Модифицирует чекпоинт перед сохранением на диск.

        Если используется PEFT, оставляет в чекпоинте только веса адаптера
        для экономии дискового пространства.
        """
        from peft import PeftModel
        from peft.utils import get_peft_model_state_dict

        if isinstance(self.model, PeftModel):
            checkpoint["state_dict"] = get_peft_model_state_dict(
                self.model, state_dict=checkpoint["state_dict"]
            )
        else:
            pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs.loss

        if loss is None:
            raise ValueError("Модель не вернула loss. Проверь передачу labels из коллатора.")

        if not torch.isfinite(loss):
            logger.warning(
                "batch_idx=%d: loss=%s — пропускаем батч (возврат None).", batch_idx, loss.item()
            )
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _log_perplexity(self, loss: torch.Tensor, phase: str) -> None:
        try:
            perplexity = torch.exp(loss)
            self.log(f"{phase}_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True)
        except OverflowError:
            self.log(f"{phase}_perplexity", float("inf"), on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(**batch)
        loss = outputs.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self._compute_extra_metrics(loss, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(**batch)
        loss = outputs.loss

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self._compute_extra_metrics(loss, "test")

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("Нет параметров для обучения! Проверь конфигурацию модификаторов.")

        if callable(self.optimizer_cfg):
            optimizer = self.optimizer_cfg(trainable_params)
        else:
            optimizer = instantiate(self.optimizer_cfg, params=trainable_params)

        if self.scheduler_cfg is None:
            return optimizer

        if callable(self.scheduler_cfg):
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps == float("inf"):
                raise ValueError(
                    "estimated_stepping_batches=inf: задайте max_steps в конфиге тренера."
                )
            scheduler = self.scheduler_cfg(optimizer=optimizer, num_training_steps=total_steps)
        else:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
