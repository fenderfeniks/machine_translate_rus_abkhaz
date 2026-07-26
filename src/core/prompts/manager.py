# src/core/prompts/manager.py
import logging
from typing import Any

from jinja2 import Environment, meta
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


class PromptManager:
    """Индустриальный менеджер промптов.

    Обеспечивает безопасный рендеринг Jinja2-шаблонов со строгой валидацией
    аргументов. Автоматически обрабатывает конфигурации Hydra.
    """

    def __init__(self, templates: dict[str, str] | DictConfig | None = None) -> None:
        """Инициализирует менеджер промптов.

        Args:
            templates: Словарь с шаблонами или DictConfig от Hydra.
                Если передан DictConfig, он будет автоматически
                сконвертирован в нативный словарь.
        """
        if isinstance(templates, DictConfig):
            self.templates = OmegaConf.to_container(templates, resolve=True)
        else:
            self.templates = templates or {}

        self.env = Environment(autoescape=False)  # Для промптов эскейпинг HTML не нужен
        logger.info("Инициализирован PromptManager. Загружено шаблонов: %d", len(self.templates))

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Рендерит промпт по имени, подставляя переменные.

        Args:
            template_name: Имя шаблона из загруженной конфигурации.
            **kwargs: Значения для переменных шаблона.

        Returns:
            Собранная строка промпта.

        Raises:
            ValueError: Если шаблон не найден или переданы не все
                обязательные переменные.
        """
        if template_name not in self.templates:
            raise ValueError(
                f"Шаблон '{template_name}' не найден в реестре. "
                f"Доступные шаблоны: {list(self.templates.keys())}"
            )

        template_str = self.templates[template_name]

        # Строгая валидация: проверяем, все ли нужные переменные переданы
        ast = self.env.parse(template_str)
        required_vars = meta.find_undeclared_variables(ast)

        missing_vars = [var for var in required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(
                f"Ошибка сборки промпта '{template_name}'. "
                f"Пропущены обязательные переменные: {missing_vars}"
            )

        template = self.env.from_string(template_str)
        rendered_prompt = template.render(**kwargs)

        logger.debug(
            "Промпт '%s' успешно отрендерен (длина: %d симв.)",
            template_name,
            len(rendered_prompt),
        )
        return rendered_prompt
