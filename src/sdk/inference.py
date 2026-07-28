# src/sdk/inference.py
import asyncio
import logging
import threading
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.core.inference.generator import HFTextGenerator
from src.utils.checkpoint_utils import load_checkpoint
from src.utils.logger import setup_logging
from src.utils.mlflow import resolve_lora_resume_path


setup_logging()
logger = logging.getLogger(__name__)


class LLMGenerationPipeline:
    def __init__(
        self,
        config_name: str = "main",
        checkpoint_path: str | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Инициализация LLMGenerationPipeline на устройстве: %s", self.device)

        config_dir = str(Path(__file__).resolve().parents[2] / "configs")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            self.cfg = compose(config_name=config_name)
            OmegaConf.resolve(self.cfg)

        self.tokenizer = instantiate(self.cfg.model.tokenizer).build()

        resume_cfg = self.cfg.get("lora_resume", {})
        lora_resume_path = resolve_lora_resume_path(resume_cfg)
        if lora_resume_path:
            logger.info("LoRA адаптер будет загружен из: %s", lora_resume_path)
            OmegaConf.update(
                self.cfg,
                "model.modifiers.finetuning.lora_resume_path",
                lora_resume_path,
                force_add=True,
            )

        builder = instantiate(self.cfg.model.builder)
        builder.modifiers_cfg = self.cfg.model.get("modifiers")
        self.model = builder.build(tokenizer=self.tokenizer)

        if checkpoint_path:
            logger.info("Подгрузка кастомных весов из: %s", checkpoint_path)
            self.model = load_checkpoint(self.model, checkpoint_path, device=self.device)

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
        if isinstance(texts, str):
            texts = [texts]

        generated_texts = self.generator.generate(texts)

        return [
            {"prompt": prompt, "generated_text": gen}
            for prompt, gen in zip(texts, generated_texts)  # noqa B905
        ]

    @torch.no_grad()
    async def generate_stream(self, text: str, **kwargs):
        """Асинхронно отдает токены через безопасную очередь."""
        sync_stream = self.generator.generate_stream(text, **kwargs)

        # Очередь для связи фонового потока и асинхронного цикла
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            """Эта функция будет работать в одном изолированном потоке."""
            try:
                for chunk in sync_stream:
                    # Безопасно закидываем токен в асинхронный цикл
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                # Если модель упала при генерации, передаем ошибку
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                # Отправляем маркер завершения (None)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # 1. Запускаем выделенный поток-производитель
        threading.Thread(target=producer, daemon=True).start()

        # 2. Асинхронно читаем токены из очереди и отдаем API
        while True:
            chunk = await queue.get()

            # Проверяем маркер завершения
            if chunk is None:
                break

            # Проверяем, не прилетела ли ошибка из потока
            if isinstance(chunk, Exception):
                raise chunk

            yield chunk
