# src/utils/torch_utils.py
import functools
import logging

import torch


logger = logging.getLogger(__name__)


BASE_SAFE_GLOBALS = [
    functools.partial,
    torch.optim.AdamW,
]


def _collect_omegaconf_globals() -> list:
    """Возвращает список безопасных классов OmegaConf.

    Returns:
        Список классов OmegaConf для регистрации.
        Пустой список, если OmegaConf не установлен.
    """
    try:
        from omegaconf import base, dictconfig, listconfig, nodes

        return [
            listconfig.ListConfig,
            dictconfig.DictConfig,
            base.ContainerMetadata,
            base.Metadata,
            nodes.AnyNode,
            nodes.IntegerNode,
            nodes.FloatNode,
            nodes.BooleanNode,
            nodes.StringNode,
            nodes.EnumNode,
        ]
    except ImportError:
        logger.debug("OmegaConf is not installed, skipping registration.")
        return []


def _collect_lr_scheduler_globals() -> list:
    """Возвращает список безопасных классов PyTorch LR-scheduler.

    Returns:
        Список классов lr_scheduler для регистрации.
        Пустой список, если модуль недоступен.
    """
    try:
        from torch.optim import lr_scheduler

        return [
            lr_scheduler.StepLR,
            lr_scheduler.LambdaLR,
            lr_scheduler.OneCycleLR,
            lr_scheduler.ReduceLROnPlateau,
            lr_scheduler.CosineAnnealingLR,
            lr_scheduler.CosineAnnealingWarmRestarts,
        ]
    except ImportError:
        logger.debug("PyTorch schedulers are unavailable.")
        return []


def _collect_transformers_globals() -> list:
    """Возвращает список безопасных объектов из Transformers.

    Returns:
        Список объектов Transformers для регистрации.
        Пустой список, если библиотека не установлена.
    """
    try:
        from transformers import get_cosine_schedule_with_warmup

        return [get_cosine_schedule_with_warmup]
    except ImportError:
        logger.debug("Transformers is not installed, skipping registration.")
        return []


def register_safe_globals() -> None:
    """Регистрирует безопасные объекты для загрузки чекпоинтов PyTorch.

    Добавляет в список доверенных объектов PyTorch классы и функции,
    которые могут присутствовать внутри сохранённых чекпоинтов.
    Это позволяет корректно восстанавливать модели, оптимизаторы,
    scheduler'ы и объекты конфигурации без ошибок десериализации.

    Функция поддерживает необязательные зависимости. Если OmegaConf
    или Transformers отсутствуют в окружении, соответствующие объекты
    будут пропущены без прерывания выполнения.

    Рекомендуется вызывать один раз при старте приложения перед
    загрузкой любых чекпоинтов.

    Raises:
        Exception: Проброс исключений из
            ``torch.serialization.add_safe_globals``, если
            переданы некорректные объекты.
    """
    collectors = [
        _collect_omegaconf_globals,
        _collect_lr_scheduler_globals,
        _collect_transformers_globals,
    ]

    safe_globals = list(BASE_SAFE_GLOBALS)
    for collect in collectors:
        safe_globals.extend(collect())

    torch.serialization.add_safe_globals(safe_globals)

    logger.debug(
        "Registered %d safe globals for checkpoint loading.",
        len(safe_globals),
    )
