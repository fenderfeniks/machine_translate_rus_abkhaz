# src/core/data/mixers.py
import logging
from typing import Any

from datasets import Dataset, DatasetDict, interleave_datasets

logger = logging.getLogger(__name__)


class InterleavedDataFetcher:
    """Класс для смешивания нескольких источников данных.

    Выступает прозрачной заменой для RawDataFetcher. Загружает список
    фетчеров и смешивает их данные с заданными вероятностями.
    """

    def __init__(
        self,
        fetchers: list[Any],
        probabilities: list[float],
        seed: int = 42,
        stopping_strategy: str = "first_exhausted",
    ) -> None:
        """Инициализирует миксер датасетов.

        Args:
            fetchers: Список инстанцированных объектов-загрузчиков.
            probabilities: Список вероятностей выборки из каждого датасета 
                (сумма должна быть равна 1.0).
            seed: Seed для воспроизводимого смешивания.
            stopping_strategy: 'first_exhausted' (остановка, когда закончится 
                самый маленький датасет) или 'all_exhausted' (когда закончатся все).
        """
        if len(fetchers) != len(probabilities):
            raise ValueError(
                "Количество fetchers должно совпадать с количеством probabilities."
            )
            
        self.fetchers = fetchers
        self.probabilities = probabilities
        self.seed = seed
        self.stopping_strategy = stopping_strategy

    def load(self) -> DatasetDict:
        """Загружает и смешивает датасеты.

        Returns:
            DatasetDict, содержащий смешанный сплит 'train'.
        """
        logger.info(
            "Смешивание %d датасетов с вероятностями %s...", 
            len(self.fetchers), self.probabilities
        )
        
        loaded_train_splits = []
        
        for i, fetcher in enumerate(self.fetchers):
            ds = fetcher.load()
            # Унифицируем формат: извлекаем train, если это DatasetDict
            if isinstance(ds, DatasetDict):
                if "train" not in ds:
                    raise ValueError(f"Датасет по индексу {i} не содержит сплита 'train'")
                loaded_train_splits.append(ds["train"])
            else:
                loaded_train_splits.append(ds)

        mixed_train = interleave_datasets(
            loaded_train_splits,
            probabilities=self.probabilities,
            seed=self.seed,
            stopping_strategy=self.stopping_strategy,
        )
        
        logger.info("Смешивание завершено. Итоговый размер train: %d записей", len(mixed_train))
        
        # Оборачиваем в DatasetDict, чтобы сплиттер (RandomDatasetSplitter)
        # позже смог нарезать из этого валидацию и тест
        return DatasetDict({"train": mixed_train})