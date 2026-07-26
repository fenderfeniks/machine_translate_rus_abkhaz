# src/core/models/response_cleaner.py
"""Постпроцессинг сырого вывода LLM.

Объединяет очистку спецтокенов, эхо-промпта, Markdown-артефактов
и обрезку незавершённых предложений в один настраиваемый класс.
"""

import logging
import re

logger = logging.getLogger(__name__)


class ResponseCleaner:
    """Универсальный постпроцессор ответов генеративной модели.

    Порядок шагов:
        1. Срезаем эхо-промпт (если модель вернула его в начале ответа).
        2. Удаляем Llama-заголовки <|start_header_id|>...<|end_header_id|>.
        3. Удаляем прочие системные токены (<|...|>, </s>, <s>).
        4. Удаляем Markdown-блоки кода (```lang ... ```).
        5. Нормализуем пробелы и обрезаем края.
        6. Обрезаем незавершённое последнее предложение.
    """

    def __init__(
        self,
        strip_prompt: bool = True,
        remove_special_tokens: bool = True,
        remove_markdown_blocks: bool = True,
        remove_extra_spaces: bool = True,
        trim_incomplete_sentence: bool = True,
    ) -> None:
        """Инициализирует клинер с заданными флагами очистки.

        Args:
            strip_prompt: Удалять ли исходный промпт из начала ответа.
            remove_special_tokens: Удалять ли системные токены вроде <s>, </s>.
            remove_markdown_blocks: Удалять ли маркеры блоков кода Markdown.
            remove_extra_spaces: Схлопывать ли множественные пробелы в один.
            trim_incomplete_sentence: Обрезать ли последнее предложение, если 
                оно не завершено знаком препинания.
        """
        self.strip_prompt = strip_prompt
        self.remove_special_tokens = remove_special_tokens
        self.remove_markdown_blocks = remove_markdown_blocks
        self.remove_extra_spaces = remove_extra_spaces
        self.trim_incomplete_sentence = trim_incomplete_sentence

    def clean(self, raw_text: str, prompt: str | None = None) -> str:
        """Очищает сгенерированный текст.

        Args:
            raw_text: Сырой текст, сгенерированный моделью.
            prompt: Исходный промпт (необходим, если strip_prompt=True).

        Returns:
            Очищенная строка.
        """
        if not raw_text:
            return ""

        text = raw_text

        # 1. Срезаем эхо-промпт
        if self.strip_prompt and prompt and text.startswith(prompt):
            text = text[len(prompt) :]

        # 2. Llama-заголовки: <|start_header_id|>assistant<|end_header_id|>
        text = re.sub(
            r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", text, flags=re.DOTALL
        )

        # 3. Прочие системные токены: <|eot_id|>, </s>, <s> и т.п.
        if self.remove_special_tokens:
            text = re.sub(r"<\|.*?\|>|</s>|<s>", "", text)

        # 4. Markdown-блоки кода: ```python\n...\n```
        if self.remove_markdown_blocks:
            text = re.sub(r"^```[a-zA-Z]*\n", "", text)
            text = re.sub(r"\n```$", "", text)

        # 5. Нормализация пробелов
        if self.remove_extra_spaces:
            text = re.sub(r" +", " ", text)
            text = text.strip()

        # 6. Обрезка незавершённого последнего предложения
        if self.trim_incomplete_sentence and text:
            last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_punct != -1 and last_punct < len(text) - 1:
                text = text[: last_punct + 1]

        return text