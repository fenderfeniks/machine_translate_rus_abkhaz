# src/core/data/collators.py
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase


class InstructionDataCollator:
    """Облегченный коллатор для подготовки батчей инструктивных данных.

    Принимает готовые input_ids, выполняет быструю сборку батча и
    маскирование промптов для корректного расчета Loss только по ответам.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int = 2048,
        response_template: Optional[str] = None,
        mask_prompt: bool = True,
    ) -> None:
        """Инициализирует коллатор.

        Args:
            tokenizer: Токенизатор модели.
            max_sequence_length: Максимальная длина последовательности.
                По умолчанию 2048.
            response_template: Строковый шаблон, предваряющий ответ модели
                (используется для поиска начала ответа, если текст склеен).
            mask_prompt: Флаг маскирования промпта (Loss не считается по вопросу).
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.response_template = response_template
        self.mask_prompt = mask_prompt

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Формирует батч из списка признаков.

        Args:
            features: Список словарей с ключами `input_ids`, `attention_mask`
                и опционально `prompt_len`.

        Returns:
            Словарь с тензорами `input_ids`, `attention_mask` и `labels`.
        """
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        # tokenizer.pad() не поддерживает truncation — обрезаем вручную до паддинга.
        # Это страховка: датасет должен обрезать на этапе подготовки, но если
        # последовательность всё же длиннее max_sequence_length — clip здесь.
        input_ids = [ids[: self.max_sequence_length] for ids in input_ids]
        attention_masks = [mask[: self.max_sequence_length] for mask in attention_masks]

        # Динамический паддинг до длины самой длинной последовательности в батче.
        # padding="longest" + max_length устраняет UserWarning, который возникал
        # при padding=True: тогда max_length молча игнорировался.
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="longest",
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Маскирование промпта (Loss не считается по вопросу)
        if self.mask_prompt:
            # Вариант А: Если промпт токенизировался отдельно (есть prompt_len)
            if "prompt_len" in features[0]:
                for i, f in enumerate(features):
                    p_len = f["prompt_len"]
                    labels[i, :p_len] = -100

            # Вариант Б: Если текст был единым, ищем место входа ответа
            elif self.response_template:
                response_token_ids = self.tokenizer.encode(
                    self.response_template, add_special_tokens=False
                )
                for i in range(len(features)):
                    labels[i] = self._mask_labels_before_response(
                        labels[i], response_token_ids
                    )

        # Маскируем токены паддинга
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch

    def _mask_labels_before_response(
        self, label_row: torch.Tensor, response_token_ids: list[int]
    ) -> torch.Tensor:
        """Ищет шаблон ответа в токенах и маскирует все, что до него.

        Args:
            label_row: Тензор лейблов для одной последовательности.
            response_token_ids: Список токенов, обозначающих начало ответа.

        Returns:
            Обновленный тензор лейблов.
        """
        response_len = len(response_token_ids)
        for idx in range(len(label_row) - response_len + 1):
            if label_row[idx : idx + response_len].tolist() == response_token_ids:
                label_row[: idx + response_len] = -100
                return label_row
        return label_row