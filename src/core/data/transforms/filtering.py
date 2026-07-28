import logging

from datasets import Dataset as HFDataset

from src.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class LengthFilterTransform(BaseDatasetTransform):
    """Трансформация для отсечения слишком длинных последовательностей."""

    def __init__(self, max_length: int = 2048, num_proc: int = 4) -> None:
        self.max_length = max_length
        self.num_proc = num_proc

    def __call__(self, dataset: HFDataset) -> HFDataset:
        initial_count = len(dataset)
        filtered_ds = dataset.filter(
            lambda x: len(x["input_ids"]) <= self.max_length,
            num_proc=self.num_proc,
            desc=f"Filtering > {self.max_length} tokens",
        )
        logger.info("Отфильтровано по длине: %d -> %d", initial_count, len(filtered_ds))
        return filtered_ds