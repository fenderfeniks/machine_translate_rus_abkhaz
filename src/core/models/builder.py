# src/core/models/builder.py
import importlib
import logging
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class HFModelBuilder:
    """Индустриальная фабрика для моделей Hugging Face.

    Отвечает строго за сборку архитектуры: веса, квантование, память и PEFT.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        auto_model_class: str = "transformers.AutoModelForCausalLM",
        cache_dir: Optional[str] = None,
        quantization_config: Optional[Any] = None,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        peft_config: Optional[Any] = None,
        attn_implementation: Optional[str] = "flash_attention_2",
        rope_scaling: Optional[dict[str, Any]] = None,
        gradient_checkpointing: bool = False,
        lora_resume_path: Optional[str] = None,
    ) -> None:
        """Инициализирует фабрику сборки модели.

        Args:
            model_name_or_path: Путь к модели на HF Hub или локально.
            tokenizer: Токенизатор (необходим для ресайза эмбеддингов).
            auto_model_class: Класс для загрузки (например, AutoModelForCausalLM).
            cache_dir: Директория кэша для весов.
            quantization_config: Конфигурация BitsAndBytes (объект или словарь).
            trust_remote_code: Разрешить выполнение удаленного кода.
            torch_dtype: Тип данных тензоров (например, 'float16', 'bfloat16' или 'auto').
            peft_config: Конфигурация LoraConfig (объект или словарь).
            attn_implementation: Реализация Attention (по умолчанию 'flash_attention_2').
            rope_scaling: Конфигурация масштабирования RoPE.
            gradient_checkpointing: Флаг включения чекпоинтинга градиентов.
            lora_resume_path: Путь к обученному адаптеру для дообучения (Transfer Learning).
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.auto_model_class = auto_model_class
        self.cache_dir = cache_dir
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.peft_config = peft_config
        self.attn_implementation = attn_implementation
        self.rope_scaling = rope_scaling
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_resume_path = lora_resume_path

    def build(self) -> PreTrainedModel:
        """Собирает и возвращает готовую к работе модель.

        Выполняет загрузку базовой архитектуры, применяет квантование,
        настраивает эмбеддинги (если словарь токенизатора изменился),
        включает градиентный чекпоинтинг и навешивает PEFT адаптеры.

        Returns:
            Экземпляр PreTrainedModel (или PeftModel).
        """
        logger.info("Загрузка базовой архитектуры: %s", self.model_name_or_path)

        module_name, class_name = self.auto_model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        bnb_config = None
        if self.quantization_config is not None:
            logger.info("Применение квантизации BitsAndBytes.")
            
            # 1. Если Hydra уже инстанцировала объект
            if isinstance(self.quantization_config, BitsAndBytesConfig):
                bnb_config = self.quantization_config
                # Аккуратно конвертируем строковый dtype (если он есть) в torch.dtype
                if isinstance(bnb_config.bnb_4bit_compute_dtype, str):
                    bnb_config.bnb_4bit_compute_dtype = getattr(
                        torch, bnb_config.bnb_4bit_compute_dtype
                    )
            # 2. Фолбэк, если передан словарь или DictConfig
            else:
                quant_dict = (
                    OmegaConf.to_container(self.quantization_config, resolve=True)
                    if isinstance(self.quantization_config, DictConfig)
                    else dict(self.quantization_config)
                )
                compute_dtype_str = quant_dict.get("bnb_4bit_compute_dtype")
                if isinstance(compute_dtype_str, str):
                    quant_dict["bnb_4bit_compute_dtype"] = getattr(torch, compute_dtype_str)
                bnb_config = BitsAndBytesConfig(**quant_dict)

        parsed_dtype = (
            getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"
        )

        # Защита для DDP: device_map="auto" ломает Multi-GPU обучение, используем только при квантовании
        device_map = (
            {"": torch.cuda.current_device()}
            if bnb_config is not None and torch.cuda.is_available()
            else None
        )

        parsed_rope_scaling = None
        if self.rope_scaling is not None:
            parsed_rope_scaling = (
                OmegaConf.to_container(self.rope_scaling, resolve=True)
                if isinstance(self.rope_scaling, DictConfig)
                else dict(self.rope_scaling)
            )

        # Собираем аргументы динамически, чтобы не передавать None туда, где библиотека этого не ждет
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "quantization_config": bnb_config,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": parsed_dtype,
            "device_map": device_map,
            "attn_implementation": self.attn_implementation,
        }
        
        if parsed_rope_scaling is not None:
            model_kwargs["rope_scaling"] = parsed_rope_scaling

        model = model_class.from_pretrained(
            self.model_name_or_path,
            **model_kwargs
        )

        # Синхронизация словаря и умная инициализация эмбеддингов
        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)
            old_vocab_size = model.config.vocab_size

            if old_vocab_size != vocab_size:
                logger.info(
                    "Изменение размера матрицы эмбеддингов: %d -> %d", 
                    old_vocab_size, 
                    vocab_size
                )
                model.resize_token_embeddings(vocab_size)

                if vocab_size > old_vocab_size:
                    input_embeddings = model.get_input_embeddings().weight.data
                    input_mean = input_embeddings[:old_vocab_size].mean(
                        dim=0, keepdim=True
                    )
                    input_embeddings[old_vocab_size:] = input_mean

                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        output_weight = output_embeddings.weight.data
                        output_mean = output_weight[:old_vocab_size].mean(
                            dim=0, keepdim=True
                        )
                        output_weight[old_vocab_size:] = output_mean

        if self.gradient_checkpointing:
            logger.info("Активация Gradient Checkpointing (Экономия VRAM).")
            model.gradient_checkpointing_enable()

        # PEFT меняет архитектуру графа, поэтому остается в Билдере
        if self.peft_config is not None or self.lora_resume_path is not None:
            from peft import (
                LoraConfig,
                PeftModel,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            if bnb_config is not None:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=self.gradient_checkpointing
                )

            if self.lora_resume_path is not None:
                # --- ВАРИАНТ 2: Transfer Learning — грузим готовый адаптер, продолжаем учить ---
                logger.info(
                    "Режим PEFT Resume: загрузка существующего адаптера из %s", 
                    self.lora_resume_path
                )
                model = PeftModel.from_pretrained(
                    model,
                    self.lora_resume_path,
                    is_trainable=True,  # Градиенты включены — продолжаем учить
                )
            else:
                # --- СТАНДАРТНЫЙ ПУТЬ: создаём новый пустой адаптер с нуля ---
                logger.info("Режим PEFT: Инициализация нового LoRA адаптера.")
                
                # Фикс для PEFT: проверяем, инициализировал ли Hydra LoraConfig
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
            logger.info(
                "LoRA: %d обучаемых из %d (%.4f%%)", 
                trainable, 
                all_param, 
                100 * trainable / all_param
            )

        return model