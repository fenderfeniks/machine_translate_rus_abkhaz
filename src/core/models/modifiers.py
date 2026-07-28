# src/core/models/modifiers.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class BaseModelModifier(ABC):
    """Базовый интерфейс для модификации архитектуры модели после загрузки."""

    @abstractmethod
    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        pass


class EmbeddingResizeModifier(BaseModelModifier):
    """Синхронизирует размер матрицы эмбеддингов с размером словаря токенизатора."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        vocab_size = len(self.tokenizer)
        old_vocab_size = model.config.vocab_size

        if old_vocab_size != vocab_size:
            logger.warning(
                "Изменение размера матрицы эмбеддингов (%d -> %d). "
                "Это увеличит потребление VRAM.", 
                old_vocab_size, vocab_size
            )
            model.resize_token_embeddings(vocab_size)

            if vocab_size > old_vocab_size:
                input_embeddings = model.get_input_embeddings().weight.data
                input_mean = input_embeddings[:old_vocab_size].mean(dim=0, keepdim=True)
                input_embeddings[old_vocab_size:] = input_mean

                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None:
                    output_weight = output_embeddings.weight.data
                    output_mean = output_weight[:old_vocab_size].mean(dim=0, keepdim=True)
                    output_weight[old_vocab_size:] = output_mean
        else:
            logger.info("Размер словаря совпадает (%d). Ресайз не требуется.", vocab_size)

        return model


class PEFTModifier(BaseModelModifier):
    """Подготавливает модель и применяет LoRA-адаптеры."""

    def __init__(
        self, 
        peft_config: Any, 
        lora_resume_path: Optional[str] = None,
        gradient_checkpointing: bool = True,
        is_quantized: bool = True,
    ) -> None:
        self.peft_config = peft_config
        self.lora_resume_path = lora_resume_path
        self.gradient_checkpointing = gradient_checkpointing
        self.is_quantized = is_quantized

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        # Подготовка к квантованному обучению (заморозка базовых весов, касты fp32)
        if self.is_quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )
        elif self.gradient_checkpointing:
            # use_reentrant=False обязателен с PyTorch ≥ 2.1 + PEFT:
            # reentrant-режим несовместим с LoRA hooks и даёт некорректные градиенты
            # для modules_to_save. Без флага PyTorch бросает UserWarning и использует
            # устаревший путь.
            model.gradient_checkpointing_enable({"use_reentrant": False})

        if self.lora_resume_path is not None:
            logger.info("PEFT: загрузка существующего адаптера из %s", self.lora_resume_path)
            model = PeftModel.from_pretrained(model, self.lora_resume_path, is_trainable=True)
        else:
            logger.info("PEFT: Инициализация нового LoRA адаптера.")
            if isinstance(self.peft_config, LoraConfig):
                lora_config = self.peft_config
            else:
                peft_dict = (
                    OmegaConf.to_container(self.peft_config, resolve=True)
                    if isinstance(self.peft_config, DictConfig)
                    else dict(self.peft_config)
                )
                lora_config = LoraConfig(**peft_dict)
                
            model = get_peft_model(model, lora_config)

        trainable, all_param = model.get_nb_trainable_parameters()
        logger.info("LoRA: %d обучаемых из %d (%.4f%%)", trainable, all_param, 100 * trainable / all_param)

        return model

class FullFinetuningModifier(BaseModelModifier):
    """Подготавливает модель для полного дообучения (Full Fine-Tuning)."""

    def __init__(self, gradient_checkpointing: bool = True) -> None:
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.gradient_checkpointing:
            logger.info("Активация Gradient Checkpointing для Full Fine-Tuning.")
            model.gradient_checkpointing_enable({"use_reentrant": False})
        
        # Принудительно размораживаем все веса (на случай, если базовый класс их заморозил)
        for param in model.parameters():
            param.requires_grad = True
            
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Full Fine-Tuning: %d обучаемых параметров (100%%)", trainable)
        
        return model