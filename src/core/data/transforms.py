# src/core/data/transforms.py
import functools
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Optional

from datasets import Dataset as HFDataset
from pydantic import ValidationError
from transformers import PreTrainedTokenizerBase

from src.core.data.cleaners import TextCleaningPipeline
from src.core.data.schemas import RawDatasetRecord

logger = logging.getLogger(__name__)


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


class TokenizationTransform(BaseDatasetTransform):
    """Трансформация для токенизации текстов и диалогов."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        use_chat_template: bool = False,
        text_column: Optional[str] = "text",
        prompt_column: Optional[str] = "prompt",
        target_column: Optional[str] = "response",
        messages_column: str = "messages",
        separator: str = " ",
        num_proc: int = 4,
        batch_size: int = 1000,
        writer_batch_size: int = 200,
    ) -> None:
        """Инициализирует трансформацию токенизации.

        Args:
            tokenizer: Токенизатор для обработки текстов.
            use_chat_template: Флаг использования chat_template токенизатора.
            text_column: Имя колонки с цельным текстом.
            prompt_column: Имя колонки с промптом.
            target_column: Имя колонки с ответом.
            messages_column: Имя колонки с сообщениями для chat_template.
            separator: Разделитель для склейки prompt и response.
            num_proc: Количество процессов для параллельной обработки.
            batch_size: Размер батча для маппинга.
            writer_batch_size: Размер батча при записи на диск.
        """
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.messages_column = messages_column
        self.separator = separator
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Применение токенизации...")

        def _process(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            # 1. Chat Template
            if self.use_chat_template and self.messages_column in examples:
                return self.tokenizer.apply_chat_template(
                    examples[self.messages_column], tokenize=True, return_dict=True
                )

            # 2. Готовый склеенный текст
            if self.text_column and self.text_column in examples:
                return self.tokenizer(
                    examples[self.text_column], add_special_tokens=True
                )

            # 3. Раздельные prompt и response
            if self.prompt_column in examples and self.target_column in examples:
                prompts = examples[self.prompt_column]
                responses = examples[self.target_column]
                full_texts = [
                    p + self.separator + r for p, r in zip(prompts, responses)
                ]
                encodings = self.tokenizer(full_texts, add_special_tokens=True)
                prompt_encodings = self.tokenizer(prompts, add_special_tokens=False)
                return {
                    "input_ids": encodings["input_ids"],
                    "attention_mask": encodings["attention_mask"],
                    "prompt_len": [len(p) for p in prompt_encodings["input_ids"]],
                }

            raise ValueError("Не найдены нужные колонки для токенизации.")

        return dataset.map(
            _process,
            batched=True,
            batch_size=self.batch_size,
            writer_batch_size=self.writer_batch_size,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )


class LengthFilterTransform(BaseDatasetTransform):
    """Трансформация для отсечения слишком длинных последовательностей."""

    def __init__(self, max_length: int = 2048, num_proc: int = 4) -> None:
        """Инициализирует фильтр по длине.

        Args:
            max_length: Максимально допустимая длина последовательности.
            num_proc: Количество процессов для параллельной фильтрации.
        """
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
        """Инициализирует упаковщик последовательностей.

        Args:
            packing_chunk_size: Целевой размер упакованного блока.
            drop_remainder: Отбрасывать ли остаток, не влезающий в блок.
            num_proc: Количество процессов.
            batch_size: Размер батча для маппинга.
            writer_batch_size: Размер батча при записи на диск.
        """
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


class ValidationTransform(BaseDatasetTransform):
    """Фильтрует датасет через Pydantic, отбрасывая битые записи.

    Поддерживает два режима в зависимости от наличия колонок:
    - prompt + target (динамические имена) -> валидация пары (SFT-сценарий)
    - text (динамическое имя)              -> валидация одиночного текста (CPT-сценарий)
    """

    def __init__(
        self, 
        text_column: Optional[str] = "text",
        prompt_column: Optional[str] = "prompt",
        target_column: Optional[str] = "target", 
        num_proc: int = 4, 
        batch_size: int = 1000
    ) -> None:
        """Инициализирует валидатор записей.

        Args:
            text_column: Имя колонки для CPT-сценария.
            prompt_column: Имя колонки промпта для SFT-сценария.
            target_column: Имя колонки таргета для SFT-сценария.
            num_proc: Количество процессов.
            batch_size: Размер батча.
        """
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
                record = RawDatasetRecord(prompt=text)
                valid_texts.append(record.prompt)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (text). Ошибка: %s", e)
                valid_texts.append("")
        return {self.text_column: valid_texts}


class CleaningTransform(BaseDatasetTransform):
    """Трансформация для очистки текста через кастомные клинеры."""

    def __init__(
        self,
        cleaners: list, 
        text_column: Optional[str] = "text",
        prompt_column: Optional[str] = "prompt",
        target_column: Optional[str] = "target",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        """Инициализирует пайплайн очистки.

        Args:
            cleaners: Список инициализированных объектов клинеров (из Hydra).
            text_column: Имя колонки для CPT-сценария.
            prompt_column: Имя колонки промпта для SFT-сценария.
            target_column: Имя колонки таргета для SFT-сценария.
            num_proc: Количество процессов.
            batch_size: Размер батча.
        """
        self.pipeline = TextCleaningPipeline(cleaners)
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