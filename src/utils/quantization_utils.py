# src/utils/quantization_utils.py

"""Утилиты для применения BitsAndBytes-квантования во время инференса.

Модуль содержит вспомогательные функции для создания конфигурации
квантования и её интеграции в Hydra-конфиг модели.

Используется сценариями инференса для динамического включения
4-битного или 8-битного режима загрузки модели без изменения
основной конфигурации проекта.
"""

import logging

import torch
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)

DEFAULT_QUANTIZATION_BITS = 4
DEFAULT_COMPUTE_DTYPE = torch.float16


def build_quantization_config(
    bits: int,
    compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
):
    """Создаёт объект BitsAndBytesConfig.

    Формирует конфигурацию квантования Hugging Face Transformers
    в зависимости от требуемой разрядности.

    Args:
        bits: Количество бит для загрузки модели.
            Поддерживаются значения 4 и 8.
        compute_dtype: Тип данных для вычислений в 4-битном режиме.
            Игнорируется при ``bits=8``.
            По умолчанию ``torch.float16``.

    Returns:
        Экземпляр ``BitsAndBytesConfig``.

    Raises:
        ValueError:
            Если передана неподдерживаемая разрядность.
        ImportError:
            Если библиотека ``transformers`` не установлена.
    """
    if bits not in (4, 8):
        raise ValueError(
            f"Unsupported quantization mode: {bits}. "
            "Only 4-bit and 8-bit quantization are supported."
        )

    from transformers import BitsAndBytesConfig

    kwargs: dict = {
        "load_in_4bit": (bits == 4),
        "load_in_8bit": (bits == 8),
    }

    if bits == 4:
        kwargs["bnb_4bit_compute_dtype"] = compute_dtype

    return BitsAndBytesConfig(**kwargs)


def apply_inference_quantization(cfg: DictConfig) -> DictConfig:
    """Добавляет конфигурацию квантования в Hydra-конфиг.

    Если инференс-квантование включено, создаёт объект
    ``BitsAndBytesConfig`` и сохраняет его в
    ``cfg.model.builder.quantization_config``.

    Если квантование отключено, конфигурация возвращается без
    изменений.

    Args:
        cfg: Hydra-конфиг проекта.

    Returns:
        Обновлённый объект ``DictConfig``.

    Raises:
        ValueError:
            Если в конфиге указана неподдерживаемая разрядность.
        ImportError:
            Если библиотека ``transformers`` не установлена.
    """
    infer_quant = OmegaConf.select(cfg, "inference.quantization", default=None)

    if not infer_quant or not infer_quant.get("enabled", False):
        return cfg

    bits = infer_quant.get("bits", DEFAULT_QUANTIZATION_BITS)

    logger.info("Applying %d-bit BitsAndBytes quantization.", bits)

    if OmegaConf.select(cfg, "model.builder") is None:
        logger.warning("cfg.model.builder does not exist; it will be created automatically.")

    OmegaConf.update(
        cfg,
        "model.builder.quantization_config",
        build_quantization_config(bits),
        force_add=True,
    )

    return cfg
