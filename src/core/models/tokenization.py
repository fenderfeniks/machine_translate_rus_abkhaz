# src/core/models/tokenization.py
import logging
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class HFTokenizerBuilder:
    """Фабрика для безопасной загрузки и настройки HuggingFace токенизаторов.

    Решает индустриальные проблемы с отсутствующим pad_token и 
    выравниванием padding_side.
    """

    def __init__(
        self,
        tokenizer_name: str,
        use_fast: bool = True,
        padding_side: str = "right",
        add_eos_token: bool = False,
        chat_template: Optional[str] = None,
    ) -> None:
        """Инициализирует фабрику токенизатора.

        Args:
            tokenizer_name: Путь к модели на HF Hub (напр., "meta-llama/Meta-Llama-3-8B").
            use_fast: Использовать ли Rust-версию токенизатора (быстрее, стандарт де-факто).
            padding_side: "right" для претрейна/файнтюна, "left" для батч-генерации.
            add_eos_token: Добавлять ли токен конца строки автоматически.
            chat_template: Строка с Jinja2 шаблоном для форматирования диалогов (опционально).
        """
        self.tokenizer_name = tokenizer_name
        self.use_fast = use_fast
        self.padding_side = padding_side
        self.add_eos_token = add_eos_token
        self.chat_template = chat_template

    def build(self) -> PreTrainedTokenizerBase:
        """Загружает токенизатор, применяет патчи и возвращает готовый объект.

        Returns:
            Инициализированный и пропатченный токенизатор.
        """
        logger.info("Загрузка токенизатора: %s", self.tokenizer_name)
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=self.use_fast,
            add_eos_token=self.add_eos_token,
        )

        # Принудительно устанавливаем сторону паддинга
        tokenizer.padding_side = self.padding_side

        # Индустриальный фикс для моделей семейства Llama/Mistral, у которых нет pad_token
        if tokenizer.pad_token is None:
            logger.warning(
                "У токенизатора %s нет pad_token. "
                "Устанавливаем pad_token_id = eos_token_id",
                self.tokenizer_name,
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Установка кастомного шаблона чата (если мы делаем Instruction Fine-Tuning)
        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template

        return tokenizer