# src/sdk/inference.py
import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.core.models.generator import HFTextGenerator
from src.utils.checkpoint_utils import load_checkpoint
from src.utils.logger import setup_logging
from src.utils.mlflow_helpers import resolve_lora_resume_path
from src.utils.quantization_utils import apply_inference_quantization


setup_logging()

logger = logging.getLogger(__name__)


class LLMGenerationPipeline:
    """SDK для инференса генеративных моделей.

    Скрывает инициализацию Hydra, квантование и логику генерации,
    предоставляя простой и готовый к использованию интерфейс
    для вызова модели (например, из FastAPI).
    """

    def __init__(
        self,
        config_name: str = "main",
        checkpoint_path: str | None = None,
    ) -> None:
        """Инициализирует пайплайн инференса.

        Args:
            config_name: Имя главного конфигурационного файла Hydra (без .yaml).
            checkpoint_path: Путь к локальным весам для ручной подгрузки (поверх базовой).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Инициализация LLMGenerationPipeline на устройстве: %s", self.device)

        config_dir = str(Path(__file__).resolve().parents[2] / "configs")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            self.cfg = compose(config_name=config_name)
            OmegaConf.resolve(self.cfg)

        self.tokenizer = instantiate(self.cfg.model.tokenizer).build()

        # Защита от OOM: Форсированное квантование для API
        self.cfg = apply_inference_quantization(self.cfg)

        # Загрузка LoRA адаптера из MLflow Registry (если включено в конфиге)
        resume_cfg = self.cfg.model.get("lora_resume", {})
        lora_resume_path = resolve_lora_resume_path(resume_cfg)
        if lora_resume_path:
            logger.info("LoRA адаптер будет загружен из: %s", lora_resume_path)
        else:
            logger.warning(
                "lora_resume.enabled=false — инференс на базовой архитектуре без адаптера."
            )

        self.model = instantiate(
            self.cfg.model.builder, tokenizer=self.tokenizer, lora_resume_path=lora_resume_path
        ).build()

        # Опциональная загрузка кастомного чекпоинта поверх (для отладки/экспериментов)
        if checkpoint_path:
            logger.info("Подгрузка кастомных весов из: %s", checkpoint_path)
            self.model = load_checkpoint(self.model, checkpoint_path, device=self.device)

        # Перенос на устройство не нужен, если модель уже квантована (device_map="auto")
        if not getattr(self.model, "is_quantized", False):
            self.model.to(self.device)

        self.model.eval()

        self.generator = HFTextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            generation_kwargs=self.cfg.get("inference", {}).get("generation_kwargs", {}),
        )

    @torch.no_grad()
    def __call__(self, texts: str | list[str]) -> list[dict[str, str]]:
        """Запускает генерацию для переданных текстов.

        Args:
            texts: Строка промпта или список строк.

        Returns:
            Список словарей, каждый из которых содержит исходный промпт
            и сгенерированный ответ.
        """
        if isinstance(texts, str):
            texts = [texts]

        generated_texts = self.generator.generate(texts)

        return [
            {"prompt": prompt, "generated_text": gen}
            for prompt, gen in zip(texts, generated_texts)  # noqa B905
        ]
