# src/core/data/cleaners.py
import re
from abc import ABC, abstractmethod


class BaseCleaner(ABC):
    """Базовый класс для всех обработчиков текста.

    Задает единый интерфейс очистки.
    """

    @abstractmethod
    def clean(self, text: str) -> str:
        """Основной метод очистки текста.

        Args:
            text: Исходный сырой текст.

        Returns:
            Очищенный текст.
        """
        pass


class RegexCleaner(BaseCleaner):
    """Класс для очистки текста на основе регулярных выражений.

    Удобен для удаления ссылок, HTML-тегов или спецсимволов.
    """

    def __init__(self, pattern: str, replacement: str = "") -> None:
        """Инициализирует регулярное выражение.

        Args:
            pattern: Регулярное выражение для поиска.
            replacement: Строка, на которую заменяем найденные совпадения.
                По умолчанию пустая строка.
        """
        self.pattern = re.compile(pattern)
        self.replacement = replacement

    def clean(self, text: str) -> str:
        """Применяет регулярное выражение к тексту.

        Args:
            text: Исходный текст.

        Returns:
            Текст после применения регулярного выражения.
        """
        return self.pattern.sub(self.replacement, text)


class TextCleaningPipeline:
    """Пайплайн, объединяющий несколько шагов очистки в один вызов."""

    def __init__(self, cleaners: list[BaseCleaner]) -> None:
        """Инициализирует пайплайн списком клинеров.

        Args:
            cleaners: Список инстансов классов-наследников BaseCleaner.
        """
        self.cleaners = cleaners

    def __call__(self, text: str) -> str:
        """Прогоняет текст через все клинеры по очереди.

        Args:
            text: Исходный текст.

        Returns:
            Текст после прохождения всех этапов очистки.
        """
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text