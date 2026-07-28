import functools
import logging
import operator
from typing import Any

from datasets import Dataset as HFDataset

from src.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class SequencePackingTransform(BaseDatasetTransform):
    """Трансформация для упаковки коротких текстов в длинные блоки."""

    def __init__(
        self,
        packing_chunk_size: int = 2048,
        drop_remainder: bool = True,
        num_proc: int = 4,
        batch_size: int = 1000,
        writer_batch_size: int = 200,
    ) -> None:
        self.packing_chunk_size = packing_chunk_size
        self.drop_remainder = drop_remainder
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Упаковка последовательностей (Sequence Packing)...")

        def _pack_sequences(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            concatenated = {
                k: functools.reduce(operator.iconcat, examples[k], [])
                for k in examples
                if k in ["input_ids", "attention_mask"]
            }
            total_length = len(concatenated["input_ids"])
            if self.drop_remainder:
                total_length = (
                    total_length // self.packing_chunk_size
                ) * self.packing_chunk_size
            return {
                k: [
                    t[i : i + self.packing_chunk_size]
                    for i in range(0, total_length, self.packing_chunk_size)
                ]
                for k, t in concatenated.items()
            }

        return dataset.map(
            _pack_sequences,
            batched=True,
            batch_size=self.batch_size,
            writer_batch_size=self.writer_batch_size,
            num_proc=self.num_proc,
            desc=f"Packing to {self.packing_chunk_size}",
        )