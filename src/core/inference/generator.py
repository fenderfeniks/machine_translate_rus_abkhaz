# src/core/models/generator.py
"""Модуль для текстовой генерации (Inference).

Оборачивает логику декодирования, сэмплирования и очистки токенов.
"""

import logging
from collections.abc import Iterator
from threading import Thread
from typing import Any

import torch
from hydra.utils import instantiate
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from src.core.inference.response_cleaner import ResponseCleaner


logger = logging.getLogger(__name__)


class HFTextGenerator:
    """Обертка над моделями Hugging Face для удобной генерации текста."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generation_kwargs: dict[str, Any],
        cleaner_cfg: Any = None,
    ) -> None:
        """Инициализирует генератор текстов.

        Args:
            model: Загруженная модель HF.
            tokenizer: Токенизатор для декодирования и паддинга.
            generation_kwargs: Базовые аргументы генерации (temperature, top_p и т.д.).
            cleaner_cfg: Конфигурация для инстанцирования ResponseCleaner.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs

        # Защита от OOM при батч-генерации (которую мы обсуждали)
        if self.tokenizer.padding_side != "left":
            logger.warning(
                "Принудительно устанавливаем padding_side='left' для корректной генерации!"
            )
            self.tokenizer.padding_side = "left"

        self.cleaner = instantiate(cleaner_cfg) if cleaner_cfg else ResponseCleaner()
        self.model.eval()

    @torch.inference_mode()  # Отключаем градиенты для ускорения и экономии памяти
    # ИСПРАВЛЕНИЕ 2: Добавляем **kwargs, чтобы эндпоинты могли безопасно переопределять параметры (Race Condition fix)
    def generate(self, texts: str | list[str], **kwargs: Any) -> list[str]:
        """Генерирует ответы для одного текста или батча текстов.

        Args:
            texts: Одиночный промпт или список промптов.
            **kwargs: Переопределения параметров генерации для конкретного вызова.

        Returns:
            Список сгенерированных и очищенных строк.
        """
        if isinstance(texts, str):
            texts = [texts]

        # ИСПРАВЛЕНИЕ 1: Надежно получаем device для inputs, даже если модель распределена по GPU
        device = next(self.model.parameters()).device

        # 1. Токенизация (обязательно на устройство модели)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,  # явно, без сюрпризов
        ).to(device)

        # Мержим глобальные параметры (из конфига) с локальными (из API)
        # Это позволяет динамически менять max_new_tokens на один запрос, не мутируя self.generation_kwargs
        current_gen_kwargs = self.generation_kwargs.copy()
        current_gen_kwargs.update(kwargs)

        # 2. Вызов встроенного метода генерации
        # Распаковываем обновленные **kwargs
        generated_ids = self.model.generate(
            **inputs,
            **current_gen_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. Очистка ответа от входного промпта на уровне токенов
        # model.generate возвращает [input_ids + generated_ids]
        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]

        # 4. Декодирование обратно в текст
        # Оставляем спецтокены, чтобы наш ResponseCleaner вырезал их наверняка
        decoded_texts = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        # 5. Финальный постпроцессинг (строковая очистка в один проход)
        final_responses = [
            self.cleaner.clean(raw_text=raw_response, prompt=original_prompt)
            for original_prompt, raw_response in zip(texts, decoded_texts)  # noqa
        ]

        return final_responses

    @torch.inference_mode()
    def generate_stream(self, text: str, **kwargs: Any) -> Iterator[str]:
        """Потоковая генерация текста.

        Возвращает итератор, который отдаёт куски текста по мере генерации.
        Поддерживает только одиночные запросы (batch_size=1).

        Args:
            text: Одиночный промпт для генерации.
            **kwargs: Переопределения параметров генерации.

        Returns:
            Итератор строковых фрагментов.

        Raises:
            ValueError: Если на вход передан список текстов.
        """
        if not isinstance(text, str):
            raise ValueError("Стриминг поддерживает только одиночные строки (batch_size=1).")

        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(device)

        current_gen_kwargs = self.generation_kwargs.copy()
        current_gen_kwargs.update(kwargs)

        # 1. Инициализируем стример с магическими флагами очистки
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # Модель "проглотит" промпт и не вернет его
            skip_special_tokens=True,  # Теги <|eot_id|> и [/INST] будут вырезаны "на лету"
        )

        # 2. Подготавливаем аргументы для потока
        generation_args = {
            **inputs,
            **current_gen_kwargs,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # 3. Запускаем генерацию в фоновом системном потоке.
        # Это необходимо, иначе model.generate заблокирует код, и мы не сможем читать streamer.
        thread = Thread(target=self.model.generate, kwargs=generation_args)
        thread.start()

        # 4. Читаем из стримера и отдаем наружу (yield)
        for new_text in streamer:
            # Важный нюанс: здесь нельзя делать strip(), иначе мы "склеим"
            # слова, потеряв пробелы, которые токенизатор отдает перед новыми словами.
            # Текст уже очищен от мусора параметрами стримера.
            if new_text:
                yield new_text
