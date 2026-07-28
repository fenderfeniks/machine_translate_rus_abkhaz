# src/core/models/builder.py
import importlib
import logging
from typing import Any, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class HFModelBuilder:
    """Индустриальная фабрика для загрузки базовых моделей Hugging Face."""

    def __init__(
        self,
        model_name_or_path: str,
        auto_model_class: str = "transformers.AutoModelForCausalLM",
        cache_dir: Optional[str] = None,
        quantization_config: Optional[Any] = None,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        rope_scaling: Optional[dict[str, Any]] = None,
        # modifiers — DictConfig из model.modifiers (embedding_resize, finetuning, ...).
        # Не список: Hydra мержит подгруппы как dict с ключами по имени группы.
        # build() итерирует .values() в порядке defaults из model/default.yaml.
        modifiers: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.auto_model_class = auto_model_class
        self.cache_dir = cache_dir
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.rope_scaling = rope_scaling
        self.modifiers_cfg = modifiers  # DictConfig | None
        self.lora_resume_path: Optional[str] = None  # устанавливается снаружи через train.py

    @staticmethod
    def _resolve_attn_implementation(requested: Optional[str]) -> Optional[str]:
        """Выбирает реализацию внимания с автоматическим fallback.

        Flash Attention 2 требует:
          - пакет flash-attn (pip install flash-attn)
          - CUDA compute capability >= 8.0 (A100, H100, RTX 30xx+)

        Если условия не выполнены — откатывается на "sdpa" (встроен в PyTorch >= 2.0).
        """
        if requested != "flash_attention_2":
            return requested

        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash-attn не установлен → откат на attn_implementation='sdpa'. "
                "Установите: pip install flash-attn --no-build-isolation"
            )
            return "sdpa"

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.warning(
                    "GPU compute capability %d.x < 8.0 — Flash Attention 2 не поддерживается "
                    "→ откат на 'sdpa'.", major
                )
                return "sdpa"
        else:
            logger.warning("CUDA недоступна → attn_implementation сбрасывается в None.")
            return None

        logger.info("Flash Attention 2: железо и пакет совместимы, используем fa2.")
        return "flash_attention_2"

    def _build_modifiers(self, tokenizer: Any, lora_resume_path: Optional[str] = None) -> list:
        """Инстанциирует модификаторы из DictConfig в порядке defaults.

        Порядок ключей в model.modifiers соответствует порядку в defaults
        model/default.yaml: embedding_resize → finetuning. Hydra его гарантирует.

        Runtime-аргументы которые нельзя хранить в конфиге:
          - tokenizer → EmbeddingResizeModifier (размер словаря известен только после загрузки)
          - lora_resume_path → PEFTModifier (путь резолвится из MLflow в train.py)
        """
        if not self.modifiers_cfg:
            return []

        modifiers = []
        for name, modifier_cfg in self.modifiers_cfg.items():
            target = modifier_cfg.get("_target_", "")
            if "EmbeddingResizeModifier" in target:
                modifier = instantiate(modifier_cfg, tokenizer=tokenizer)
            elif "PEFTModifier" in target and lora_resume_path is not None:
                modifier = instantiate(modifier_cfg, lora_resume_path=lora_resume_path)
            else:
                modifier = instantiate(modifier_cfg)
            logger.info("Modifier инициализирован: %s", name)
            modifiers.append(modifier)

        return modifiers

    def build(self, tokenizer: Any = None) -> PreTrainedModel:
        logger.info("Загрузка базовой архитектуры: %s", self.model_name_or_path)

        module_name, class_name = self.auto_model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        bnb_config = None
        if self.quantization_config is not None:
            if isinstance(self.quantization_config, BitsAndBytesConfig):
                bnb_config = self.quantization_config
                
                # Универсальная проверка готового объекта
                for attr_name in dir(bnb_config):
                    if attr_name.endswith("_dtype"):
                        val = getattr(bnb_config, attr_name)
                        if isinstance(val, str):
                            setattr(bnb_config, attr_name, getattr(torch, val))
            else:
                quant_dict = (
                    OmegaConf.to_container(self.quantization_config, resolve=True)
                    if isinstance(self.quantization_config, DictConfig)
                    else dict(self.quantization_config)
                )
                
                # Если словарь не пустой (не none.yaml)
                if quant_dict:
                    # Универсальный парсер: ищем любые ключи, оканчивающиеся на _dtype
                    for k, v in quant_dict.items():
                        if k.endswith("_dtype") and isinstance(v, str):
                            quant_dict[k] = getattr(torch, v)
                            
                    bnb_config = BitsAndBytesConfig(**quant_dict)

        parsed_dtype = getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"
        device_map = {"": torch.cuda.current_device()} if bnb_config is not None else None
        attn_impl = self._resolve_attn_implementation(self.attn_implementation)

        parsed_rope_scaling = None
        if self.rope_scaling is not None:
            parsed_rope_scaling = (
                OmegaConf.to_container(self.rope_scaling, resolve=True)
                if isinstance(self.rope_scaling, DictConfig)
                else dict(self.rope_scaling)
            )

        model_kwargs = {
            "cache_dir": self.cache_dir,
            "quantization_config": bnb_config,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": parsed_dtype,
            "device_map": device_map,
            "attn_implementation": attn_impl,
        }

        if parsed_rope_scaling is not None:
            model_kwargs["rope_scaling"] = parsed_rope_scaling

        model = model_class.from_pretrained(self.model_name_or_path, **model_kwargs)

        # Инстанциируем и прогоняем модификаторы
        for modifier in self._build_modifiers(tokenizer, self.lora_resume_path):
            model = modifier(model)

        return model
    