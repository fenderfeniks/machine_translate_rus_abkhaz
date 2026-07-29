# src/core/data/transforms/tokenization.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from src.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


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

            # 2. Готовый склеенный текст (для CPT)
            if self.text_column and self.text_column in examples:
                return self.tokenizer(
                    examples[self.text_column], add_special_tokens=True
                )

            # 3. Раздельные prompt и response (для SFT)
            if (
                self.prompt_column and self.prompt_column in examples and 
                self.target_column and self.target_column in examples
            ):
                prompts = examples[self.prompt_column]
                responses = examples[self.target_column]
                full_texts = [
                    p + self.separator + r for p, r in zip(prompts, responses)
                ]
                encodings = self.tokenizer(full_texts, add_special_tokens=True)
                # Включаем separator в prompt_len — Loss считается только по target,
                # separator является частью промпта и не должен обучаться
                prompts_with_sep = [p + self.separator for p in prompts]
                prompt_encodings = self.tokenizer(prompts_with_sep, add_special_tokens=False)
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