# src/training/callbacks.py
import logging
import random
from typing import Any

import evaluate
import pandas as pd
import pytorch_lightning as pl
import torch


logger = logging.getLogger(__name__)


class ModelFreezingCallback(pl.Callback):
    """Callback для управления заморозкой градиентов до начала обучения."""

    def __init__(self, finetuning_type: str = "full") -> None:
        """Инициализирует коллбэк заморозки.

        Args:
            finetuning_type: Режим файнтюнинга
                ('lm_head_only', 'frozen_embeddings', 'peft' или 'full').
        """
        self.finetuning_type = finetuning_type

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Применяет логику заморозки параметров перед стартом обучения.

        Args:
            trainer: Объект тренера PyTorch Lightning.
            pl_module: Текущий LightningModule.
            stage: Текущая стадия (напр., 'fit', 'test').

        Raises:
            ValueError: Если передан неизвестный режим finetuning_type.
        """
        model = pl_module.model

        if self.finetuning_type == "lm_head_only":
            logger.info("Callback: Режим lm_head_only. Замораживаем всё, кроме lm_head.")
            for name, param in model.named_parameters():
                if "lm_head" not in name:
                    param.requires_grad = False

        elif self.finetuning_type == "frozen_embeddings":
            logger.info("Callback: Режим frozen_embeddings. Замораживаем матрицы токенов.")
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = False

        elif self.finetuning_type == "peft":
            logger.info("Callback: Режим PEFT. Модель уже заморожена Билдером.")

        elif self.finetuning_type == "full":
            logger.info("Callback: Режим full. Обучаются все параметры сети.")

        else:
            raise ValueError(f"Неизвестный режим finetuning_type: {self.finetuning_type}")


class GenerationEvaluationCallback(pl.Callback):
    """Callback для периодической генерации текста и подсчета ROUGE метрик."""

    def __init__(
        self,
        model_name: str,
        num_random: int = 5,
        generation_batch_size: int = 2,
        generation_kwargs: dict[str, Any] | None = None,
        fixed_samples: list[dict[str, Any]] | None = None,
    ) -> None:
        """Инициализирует коллбэк генерации.

        Args:
            model_name: Название архитектуры модели для тегирования в MLflow.
            num_random: Количество случайных примеров из валидации для генерации.
            generation_batch_size: Размер батча при генерации.
            generation_kwargs: Параметры генерации (temperature, top_p и т.д.).
            fixed_samples: Список фиксированных примеров для отслеживания прогресса.
        """
        self.model_name = model_name
        self.num_random = num_random
        self.generation_batch_size = generation_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.fixed_samples = fixed_samples or []
        self.rouge_metric = evaluate.load("rouge")

        # Будут заполнены в on_fit_start через trainer/datamodule
        self.generator = None
        self.val_raw_dataset = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Подготавливает генератор и выборку данных перед началом обучения."""
        from hydra.utils import instantiate

        from src.core.models.generator import HFTextGenerator

        self.generator = HFTextGenerator(
            model=pl_module.model,
            tokenizer=trainer.datamodule.tokenizer,
            generation_kwargs=self.generation_kwargs,
        )

        # Берём названия колонок из конфига данных
        data_cfg = trainer.datamodule.data_cfg
        prompt_col = data_cfg.get("prompt_column", "prompt")
        target_col = data_cfg.get("target_column", "completion")

        # Берём сырой датасет до трансформаций
        raw_datasets = instantiate(data_cfg.source).load()
        val_raw = raw_datasets.get("validation", raw_datasets["train"])
        n = min(self.num_random * 10, len(val_raw))
        self.val_raw_dataset = [
            {"prompt": val_raw[i][prompt_col], "response": val_raw[i][target_col]} for i in range(n)
        ]

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            mlflow_client.set_tag(run_id, "model_architecture", self.model_name)
            mlflow_client.set_tag(run_id, "task_type", "causal_lm_generation")

    def _extract_rouge_score(self, score: Any) -> float:
        """Индустриальный фикс для разных версий пакета evaluate.

        Args:
            score: Результат вычисления метрики ROUGE.

        Returns:
            Нормализованное значение метрики в виде float.
        """
        if hasattr(score, "mid"):  # Старые версии rouge_score
            return float(score.mid.fmeasure)
        elif isinstance(score, (list, tuple)) and len(score) > 0:
            return float(score[0])
        return float(score)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Выполняет генерацию текстов и расчет метрик в конце эпохи валидации."""
        logger.info("Сборка батча для генерации (Фиксированные + Случайные)...")

        # Защита от нехватки данных (если валидация меньше num_random)
        actual_num_random = min(self.num_random, len(self.val_raw_dataset))
        random_raw = random.sample(self.val_raw_dataset, actual_num_random)

        random_samples = [
            {"prompt": item["prompt"], "target": item["response"], "type": "Random"}
            for item in random_raw
        ]
        fixed_samples = [
            {"prompt": item["prompt"], "target": item["target"], "type": "Fixed"}
            for item in self.fixed_samples
        ]

        eval_batch = fixed_samples + random_samples
        prompts = [s["prompt"] for s in eval_batch]
        targets = [s["target"] for s in eval_batch]
        sample_types = [s["type"] for s in eval_batch]

        # Чанкирование генерации для защиты от OOM
        generated_texts = []
        for i in range(0, len(prompts), self.generation_batch_size):
            chunk_prompts = prompts[i : i + self.generation_batch_size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk_generated = self.generator.generate(chunk_prompts, **self.generation_kwargs)
            generated_texts.extend(chunk_generated)

        # Единственный расчёт ROUGE за эпох
        rouge_results = self.rouge_metric.compute(
            predictions=generated_texts, references=targets, use_stemmer=True
        )

        val_rouge1 = self._extract_rouge_score(rouge_results["rouge1"])
        val_rougeL = self._extract_rouge_score(rouge_results["rougeL"])  # noqa N816
        avg_gen_len = sum(len(t.split()) for t in generated_texts) / len(generated_texts)

        pl_module.log("val_rouge1", val_rouge1, sync_dist=True)
        pl_module.log("val_rougeL", val_rougeL, sync_dist=True)
        pl_module.log("val_avg_gen_length", avg_gen_len, sync_dist=True)

        # Таблица генераций в MLflow
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            df = pd.DataFrame(
                {
                    "Type": sample_types,
                    "Prompt": prompts,
                    "Target": targets,
                    "Generated": generated_texts,
                }
            )
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            epoch = trainer.current_epoch
            mlflow_client.log_table(
                run_id=run_id,
                data=df,
                artifact_file=f"generations/epoch_{epoch}_results.json",
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
