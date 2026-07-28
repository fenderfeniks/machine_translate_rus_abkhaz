# src/core/data/fetcher.py
import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)

# Маппинг расширений -> тип загрузчика HF datasets
_EXT_TO_LOADER: dict[str, str] = {
    "csv": "csv",
    "tsv": "csv",  # load_dataset("csv", sep="\t")
    "json": "json",
    "jsonl": "json",
    "txt": "text",
    "parquet": "parquet",
    "pq": "parquet",
    "arrow": "arrow",
}


def _detect_loader(file_name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Определяет тип загрузчика HF datasets по расширению файла.

    Для файлов формата TSV автоматически добавляет параметр sep='\\t' 
    в аргументы загрузчика.

    Args:
        file_name: Имя файла с данными.
        kwargs: Словарь дополнительных аргументов для загрузчика.

    Returns:
        Кортеж (тип_загрузчика, обновленные_аргументы).

    Raises:
        ValueError: Если расширение файла не поддерживается.
    """
    ext = file_name.rsplit(".", 1)[-1].lower()
    loader = _EXT_TO_LOADER.get(ext)

    if loader is None:
        raise ValueError(
            f"Неподдерживаемое расширение файла: '.{ext}'. "
            f"Поддерживаются: {list(_EXT_TO_LOADER.keys())}"
        )

    if ext == "tsv" and "sep" not in kwargs:
        kwargs = {**kwargs, "sep": "\t"}

    return loader, kwargs


class RawDataFetcher:
    """Универсальный класс для получения сырых данных.

    Проверяет локальное наличие, при необходимости скачивает с Kaggle 
    или HuggingFace. Автоматически определяет формат файла по расширению.
    """

    def __init__(
        self,
        source_type: str,
        raw_dir: str | Path,
        dataset_name: str | None = None,
        file_name: str | None = None,
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Инициализирует загрузчик данных.

        Args:
            source_type: Тип источника данных ('local', 'kaggle', 'hf').
            raw_dir: Директория для сохранения/чтения сырых данных.
            dataset_name: Имя датасета (для Kaggle или HF).
            file_name: Имя конкретного файла для загрузки.
            token: Токен доступа (используется только для HuggingFace).
            **kwargs: Дополнительные параметры для `load_dataset`.
        """
        self.source_type = source_type
        self.raw_dir = Path(raw_dir)
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.token = token
        self.kwargs = kwargs

    def load(self) -> Dataset | DatasetDict:
        """Единая точка входа для получения датасета.

        Returns:
            Загруженный датасет (Dataset или DatasetDict).

        Raises:
            ValueError: Если передан неизвестный `source_type`.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.source_type == "local":
            return self._load_local()
        elif self.source_type == "kaggle":
            return self._load_kaggle()
        elif self.source_type == "hf":
            return self._load_hf()
        else:
            raise ValueError(
                f"Неизвестный тип источника данных: '{self.source_type}'. "
                "Поддерживаются: 'local', 'kaggle', 'hf'."
            )

    def _load_local(self) -> Dataset | DatasetDict:
        if not self.file_name:
            raise ValueError("Для local источника необходимо указать file_name.")
            
        matched_files = list(self.raw_dir.glob(self.file_name))
        
        if not matched_files:
            raise FileNotFoundError(
                f"Локальные файлы не найдены по пути или шаблону: {self.raw_dir / self.file_name}"
            )

        data_files = [str(p) for p in matched_files]
        loader, kwargs = _detect_loader(self.file_name, self.kwargs)
        
        if len(data_files) == 1:
            logger.info("Загрузка локального файла: %s (loader=%s)", data_files[0], loader)
        else:
            logger.info(
                "Загрузка %d локальных файлов по шаблону %s (loader=%s)", 
                len(data_files), self.file_name, loader
            )
            
        return load_dataset(loader, data_files=data_files, **kwargs)

    def _load_kaggle(self) -> Dataset | DatasetDict:
        if not self.file_name or not self.dataset_name:
            raise ValueError("Для kaggle источника необходимы dataset_name и file_name.")

        file_path = self.raw_dir / self.file_name

        if file_path.exists():
            logger.info(
                "Kaggle датасет найден локально: %s. Скачивание пропущено.", file_path
            )
        else:
            username = os.getenv("KAGGLE_USERNAME")
            key = os.getenv("KAGGLE_KEY")

            if not username or not key:
                raise EnvironmentError(
                    "KAGGLE_USERNAME и KAGGLE_KEY не найдены в env. "
                    "Проверь K8s Secret и прокидывание через KubernetesPodOperator."
                )

            logger.info("Скачиваем %s с Kaggle (user: %s)...", self.dataset_name, username)
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(self.dataset_name, path=str(self.raw_dir), unzip=True)
            logger.info("Скачивание с Kaggle завершено.")

        loader, kwargs = _detect_loader(self.file_name, self.kwargs)
        logger.info("Загрузка Kaggle файла: %s (loader=%s)", file_path, loader)
        return load_dataset(loader, data_files=str(file_path), **kwargs)

    def _load_hf(self) -> Dataset | DatasetDict:
        if not self.dataset_name:
            raise ValueError("Для hf источника необходимо указать dataset_name.")

        hf_local_path = self.raw_dir / self.dataset_name.replace("/", "_")

        if not hf_local_path.exists():
            logger.info("Скачиваем %s из HuggingFace...", self.dataset_name)
            dataset = load_dataset(self.dataset_name, token=self.token, **self.kwargs)
            dataset.save_to_disk(str(hf_local_path))
            logger.info("HF датасет сохранен в %s", hf_local_path)
            return dataset

        logger.info("HF датасет найден локально: %s", hf_local_path)
        return load_from_disk(str(hf_local_path))