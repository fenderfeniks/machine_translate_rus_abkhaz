# src/core/data/splitters.py
import logging

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class RandomDatasetSplitter:
    """Разбивает датасет на train, validation и test выборки."""

    def __init__(
        self, 
        val_size: float = 0.1, 
        test_size: float = 0.1, 
        seed: int = 42
    ) -> None:
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def __call__(self, raw_datasets: Dataset | DatasetDict) -> DatasetDict:
        """Применяет разбиение к датасету.

        Args:
            raw_datasets: Исходный датасет или DatasetDict.

        Returns:
            DatasetDict с ключами 'train', 'validation', 'test'.
        """
        # Если сплиты уже есть (например, скачаны с HF), просто возвращаем их
        if isinstance(raw_datasets, DatasetDict) and "validation" in raw_datasets and "test" in raw_datasets:
            logger.info("Датасет уже содержит сплиты validation и test. Разбиение пропущено.")
            return raw_datasets

        # Определяем базовый датасет для разбиения
        base_ds = raw_datasets["train"] if isinstance(raw_datasets, DatasetDict) else raw_datasets

        total_holdout_size = self.val_size + self.test_size
        
        if total_holdout_size <= 0:
            logger.warning("Размер val и test равен 0. Возвращаем только train.")
            return DatasetDict({"train": base_ds})

        logger.info(
            "Разбиение датасета: val_size=%.2f, test_size=%.2f, seed=%d", 
            self.val_size, self.test_size, self.seed
        )

        # Шаг 1: Отделяем train от общей отложенной выборки (val + test)
        split_1 = base_ds.train_test_split(test_size=total_holdout_size, seed=self.seed)
        
        # Шаг 2: Разделяем отложенную выборку на val и test
        test_proportion = self.test_size / total_holdout_size
        split_2 = split_1["test"].train_test_split(test_size=test_proportion, seed=self.seed)

        return DatasetDict({
            "train": split_1["train"],
            "validation": split_2["train"], 
            "test": split_2["test"],
        })