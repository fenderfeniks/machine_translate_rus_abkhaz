# src/core/data/transforms/validation.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from pydantic import ValidationError

from src.core.data.cleaners import TextCleaningPipeline
from src.core.data.schemas import RawDatasetRecord
from src.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class ValidationTransform(BaseDatasetTransform):
    """Фильтрует датасет через Pydantic, отбрасывая битые записи.

    Поддерживает два режима в зависимости от наличия колонок:
    - prompt + target -> валидация пары (SFT-сценарий)
    - text            -> валидация одиночного текста (CPT-сценарий)
    """

    def __init__(
        self,
        pipeline,
        text_column: str | None = "text",
        prompt_column: str | None = "prompt",
        target_column: str | None = "target",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.pipeline = pipeline
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Применение валидации записей (Pydantic)...")
        initial_count = len(dataset)

        has_prompt_target = (
            self.prompt_column in dataset.column_names and 
            self.target_column in dataset.column_names
        )
        has_text = self.text_column in dataset.column_names

        if not has_prompt_target and not has_text:
            raise ValueError(
                "ValidationTransform: датасет должен содержать колонки "
                f"'{self.prompt_column}'+'{self.target_column}' или '{self.text_column}'."
            )

        if has_prompt_target:
            dataset = dataset.map(
                self._validate_prompt_target_batch,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                desc=f"Validating {self.prompt_column}+{self.target_column} records",
            )
            dataset = dataset.filter(
                lambda x: bool(x[self.prompt_column]),
                num_proc=self.num_proc,
            )
        else:
            dataset = dataset.map(
                self._validate_text_batch,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                desc=f"Validating {self.text_column} records",
            )
            dataset = dataset.filter(
                lambda x: bool(x[self.text_column]),
                num_proc=self.num_proc,
            )

        logger.info("Валидация завершена: %d -> %d записей", initial_count, len(dataset))
        return dataset

    def _validate_prompt_target_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        valid_prompts, valid_targets = [], []
        for p, t in zip(batch.get(self.prompt_column, []), batch.get(self.target_column, [])):
            try:
                record = RawDatasetRecord(prompt=p, target=t)
                valid_prompts.append(record.prompt)
                valid_targets.append(record.target)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (prompt+target). Ошибка: %s", e)
                valid_prompts.append("")
                valid_targets.append("")
                
        return {self.prompt_column: valid_prompts, self.target_column: valid_targets}

    def _validate_text_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        valid_texts = []
        for text in batch.get(self.text_column, []):
            try:
                # ИСПРАВЛЕНО: Теперь передаем аргумент text, а не prompt
                record = RawDatasetRecord(text=text)
                valid_texts.append(record.text)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (text). Ошибка: %s", e)
                valid_texts.append("")
        return {self.text_column: valid_texts}


class CleaningTransform(BaseDatasetTransform):
    """Трансформация для очистки текста через кастомные клинеры."""

    def __init__(
        self,
        pipeline,
        text_column: Optional[str] = "text",
        prompt_column: Optional[str] = "prompt",
        target_column: Optional[str] = "target",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.pipeline = pipeline
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Применение пайплайна очистки текста...")
        
        has_prompt_target = (
            self.prompt_column in dataset.column_names and 
            self.target_column in dataset.column_names
        )
        has_text = self.text_column in dataset.column_names

        def _clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            res = {}
            if has_prompt_target:
                res[self.prompt_column] = [self.pipeline(t) for t in batch[self.prompt_column]]
                res[self.target_column] = [self.pipeline(t) for t in batch[self.target_column]]
            elif has_text:
                res[self.text_column] = [self.pipeline(t) for t in batch[self.text_column]]
            return res

        return dataset.map(
            _clean_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Cleaning text",
        )