# src/core/data/transforms/base.py
from abc import ABC, abstractmethod

from datasets import Dataset as HFDataset


class BaseDatasetTransform(ABC):
    """Базовый интерфейс для всех шагов обработки данных."""

    @abstractmethod
    def __call__(self, dataset: HFDataset) -> HFDataset:
        """Применяет трансформацию к датасету.

        Args:
            dataset: Исходный датасет.

        Returns:
            Преобразованный датасет.
        """
        pass