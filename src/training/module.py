# src/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class CausalLMLightningModule(pl.LightningModule):
    """Чистый LightningModule для обучения Causal LM.

    Освобожден от логики заморозки и генерации (делегировано Callbacks).
    Поддерживает сохранение и загрузку только LoRA весов для экономии места.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
    ) -> None:
        """Инициализирует модуль Lightning.

        Args:
            model: Базовая архитектура модели PyTorch.
            optimizer_cfg: Конфигурация оптимизатора (Hydra).
            scheduler_cfg: Конфигурация планировщика (Hydra).
        """
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.save_hyperparameters(ignore=["model"])

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Модифицирует чекпоинт перед сохранением.

        Оставляет только веса, относящиеся к LoRA, чтобы не сохранять
        тяжелую квантованную базу.
        """
        lora_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if "lora_" in k or "modules_to_save" in k
        }
        checkpoint["state_dict"] = lora_state_dict

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Модифицирует состояние при загрузке из чекпоинта."""
        # При загрузке игнорируем несовпадение ключей
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        checkpoint["state_dict"] = self.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Переопределяет метод загрузки состояния для фильтрации весов."""
        # Фильтруем только LoRA веса при загрузке
        lora_state_dict = {
            k: v for k, v in state_dict.items() if "lora_" in k or "modules_to_save" in k
        }
        return super().load_state_dict(lora_state_dict, strict=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Прямой проход модели."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Один шаг обучения."""
        outputs = self(**batch)
        loss = outputs.loss

        if loss is None:
            raise ValueError("Модель не вернула loss. Проверь передачу labels из коллатора.")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Один шаг валидации с подсчетом Perplexity."""
        outputs = self(**batch)
        loss = outputs.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Расчет Perplexity
        try:
            perplexity = torch.exp(loss)
            self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True)
        except OverflowError:
            self.log(
                "val_perplexity",
                float("inf"),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Один шаг тестирования с подсчетом Perplexity."""
        outputs = self(**batch)
        loss = outputs.loss

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        try:
            perplexity = torch.exp(loss)
            self.log("test_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True)
        except OverflowError:
            self.log(
                "test_perplexity",
                float("inf"),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """Настраивает оптимизаторы и планировщики шагов обучения."""
        # КРИТИЧНО: Заморозка происходит ДО вызова этой функции через Callback.
        # Поэтому здесь мы собираем только те параметры, которые остались размороженными.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("Нет параметров для обучения! Проверь настройки заморозки в Callback.")

        if callable(self.optimizer_cfg):
            optimizer = self.optimizer_cfg(trainable_params)
        else:
            optimizer = instantiate(self.optimizer_cfg, params=trainable_params)

        if self.scheduler_cfg is None:
            return optimizer

        if callable(self.scheduler_cfg):
            # Вычисляем num_training_steps динамически
            total_steps = self.trainer.estimated_stepping_batches
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
