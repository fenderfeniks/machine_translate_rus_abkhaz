# src/utils/notebook_setup.py
import logging
import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Ищет корень проекта по наличию ``pyproject.toml``.

    Поднимается вверх по дереву директорий от текущей рабочей
    директории до тех пор, пока не найдёт ``pyproject.toml``.

    Returns:
        Путь к корню проекта.

    Raises:
        FileNotFoundError:
            Если ``pyproject.toml`` не найден ни в одной
            из родительских директорий.
    """
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / "pyproject.toml").exists():
            return directory

    raise FileNotFoundError(
        "Could not locate project root: 'pyproject.toml' not found "
        f"in '{Path.cwd()}' or any of its parents."
    )


def _ensure_on_path(directory: Path) -> None:
    """Добавляет директорию в начало ``sys.path``, если её там нет.

    Использует ``insert(0, ...)`` для обеспечения приоритета
    над системными путями.

    Args:
        directory: Путь к директории для добавления в ``sys.path``.
    """
    path_str = str(directory)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.debug("Added '%s' to sys.path.", path_str)


def _init_hydra(config_path: str, config_name: str) -> DictConfig:
    """Инициализирует Hydra и возвращает собранный конфиг.

    Перед инициализацией сбрасывает глобальное состояние Hydra,
    что позволяет безопасно повторно вызывать функцию в рамках
    одной сессии ноутбука.

    Args:
        config_path: Путь к директории конфигов относительно
            данного файла.
        config_name: Имя корневого конфига без расширения.

    Returns:
        Собранный ``DictConfig``.
    """
    GlobalHydra.instance().clear()
    initialize(config_path=config_path, version_base="1.3")
    return compose(config_name=config_name)


def setup_notebook(config_name: str = "main") -> DictConfig:
    """Инициализирует окружение для запуска ноутбука.

    Выполняет следующие шаги:
    - находит корень проекта по ``pyproject.toml``;
    - добавляет корень в ``sys.path``;
    - инициализирует Hydra и собирает конфиг;
    - настраивает базовое логирование.

    Args:
        config_name: Имя корневого Hydra-конфига без расширения.
            По умолчанию ``"main"``.

    Returns:
        Собранный ``DictConfig``.

    Raises:
        FileNotFoundError:
            Если ``pyproject.toml`` не найден в дереве директорий.
    """
    project_root = _find_project_root()
    _ensure_on_path(project_root)

    cfg = _init_hydra(
        config_path="../../configs",
        config_name=config_name,
    )

    logging.basicConfig(level=logging.INFO, force=True)
    logger.info("NLP Environment ready. Root: %s", project_root)

    return cfg
