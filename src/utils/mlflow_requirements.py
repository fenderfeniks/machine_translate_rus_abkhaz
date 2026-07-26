# src/utils/mlflow_requirements.py
"""
Формирование pip_requirements для артефактов MLflow из pyproject.toml.

MLflow при mlflow.transformers.log_model(...) по умолчанию пытается сам
определить нужные зависимости (get_default_pip_requirements) — и в этот
список может попасть пакет, которого нет в окружении (например,
torchvision), из-за чего log_model падает с ModuleNotFoundError.

Вместо того чтобы хардкодить список пакетов в коде, читаем его из
единственного источника правды — pyproject.toml, группа
[project.optional-dependencies.inference-core]. Так requirements
артефакта модели всегда соответствуют тому, что реально объявлено
как нужное для инференса, без дублирования и рассинхронизации.
"""

import logging
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


logger = logging.getLogger(__name__)

_INFERENCE_GROUP: str = "inference-core"


def _strip_version_specifier(requirement: str) -> str:
    """Удаляет спецификаторы версии и extras из строки зависимости.

    Преобразует строки вида 'torch>=2.0.0' или 'uvicorn[standard]'
    в чистое имя пакета ('torch', 'uvicorn').

    Args:
        requirement: Строка с описанием зависимости.

    Returns:
        Чистое имя пакета без версий и дополнительных опций.
    """
    name = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
    return name


def get_inference_pip_requirements(pyproject_path: str | Path) -> list[str]:
    """Читает зависимости для инференса и фиксирует их текущие версии.

    Извлекает группу [project.optional-dependencies.inference-core]
    из pyproject.toml и возвращает список requirements с версиями,
    реально установленными в текущем окружении. Это гарантирует,
    что MLflow-артефакт задекларирует именно ту версию, с которой
    модель была обучена.

    Args:
        pyproject_path: Путь к файлу pyproject.toml.

    Returns:
        Список строк с зафиксированными версиями пакетов
        (например, ['torch==2.0.1']). Если группа не найдена,
        возвращает пустой список.

    Raises:
        FileNotFoundError:
            Если файл pyproject.toml не найден по указанному пути.
        tomllib.TOMLDecodeError:
            Если файл pyproject.toml содержит невалидный TOML.
    """
    pyproject_path = Path(pyproject_path)
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        declared = data["project"]["optional-dependencies"][_INFERENCE_GROUP]
    except KeyError:
        logger.warning(
            "Группа [project.optional-dependencies.%s] не найдена в %s. "
            "Возвращаю пустой список pip_requirements — MLflow будет "
            "пытаться определить зависимости автоматически.",
            _INFERENCE_GROUP,
            pyproject_path,
        )
        return []

    pinned: list[str] = []
    for requirement in declared:
        pkg_name = _strip_version_specifier(requirement)
        try:
            installed_version = version(pkg_name)
            pinned.append(f"{pkg_name}=={installed_version}")
        except PackageNotFoundError:
            logger.warning(
                "Пакет '%s' объявлен в группе '%s', но не установлен "
                "в текущем окружении — пропускаю.",
                pkg_name,
                _INFERENCE_GROUP,
            )

    return pinned
