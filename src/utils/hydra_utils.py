# src/utils/hydra_utils.py
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.config_schema import ConfigSchema


logger = logging.getLogger(__name__)


def _force_utf8_console_encoding() -> None:
    """Принудительно устанавливает кодировку UTF-8 для вывода в консоль.

    Полезно для корректного отображения спецсимволов и кириллицы
    в логах, особенно в средах Windows или нестандартных терминалах.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="backslashreplace")
                except (ValueError, OSError):
                    pass


def _resolve_trainer_callbacks(trainer_cfg: DictConfig) -> DictConfig:
    """Конвертирует словарь коллбэков в список для pytorch_lightning.Trainer.

    В Hydra-конфигах коллбэки часто задаются как словарь для удобства
    точечного переопределения параметров, но Trainer ожидает список.
    Функция выполняет эту трансформацию.

    Args:
        trainer_cfg: Конфигурация тренера (извлеченная секция ``trainer``).

    Returns:
        Обновленная конфигурация тренера со списком коллбэков.
    """
    OmegaConf.set_struct(trainer_cfg, False)
    callbacks_dict = trainer_cfg.get("callbacks", {})
    if callbacks_dict:
        trainer_cfg.callbacks = list(callbacks_dict.values())
    return trainer_cfg


def setup_config(cfg: DictConfig) -> DictConfig:
    """Валидирует конфиг, разрешает ссылки и выставляет пути проекта.

    Выполняет следующие шаги:
    - принудительно включает UTF-8 для консоли;
    - временно извлекает секцию ``trainer`` для обхода строгой валидации;
    - определяет корень проекта и прописывает его в ``cfg.paths.root_dir``;
    - разрешает интерполяции OmegaConf;
    - валидирует структуру конфигурации через ``ConfigSchema``;
    - трансформирует словари коллбэков в списки;
    - возвращает структуру к строгой форме (set_struct=True).

    Args:
        cfg: Исходный Hydra-конфиг (DictConfig).

    Returns:
        Валидированный и подготовленный к инжекции в пайплайн DictConfig.
    """
    _force_utf8_console_encoding()

    OmegaConf.set_struct(cfg, False)

    # Вынимаем trainer до валидации схемой — он слишком динамичный
    trainer_cfg = cfg.pop("trainer")

    project_root = str(Path(__file__).resolve().parents[2])
    cfg.paths.root_dir = project_root

    OmegaConf.resolve(cfg)

    schema = OmegaConf.structured(ConfigSchema)
    validated_cfg = OmegaConf.merge(schema, cfg)

    OmegaConf.set_struct(validated_cfg, False)

    # Конвертируем callbacks dict -> list и возвращаем trainer
    trainer_cfg = _resolve_trainer_callbacks(trainer_cfg)
    validated_cfg.trainer = trainer_cfg

    OmegaConf.set_struct(validated_cfg, True)

    logger.debug("Финальная конфигурация эксперимента:\n%s", OmegaConf.to_yaml(validated_cfg))

    return validated_cfg
