# src/training/callbacks.py
import logging
import random
from typing import Any

import evaluate
import pandas as pd
import pytorch_lightning as pl
import torch


logger = logging.getLogger(__name__)

_MODE_AUTO = "auto"
_MODE_CPT = "cpt"
_MODE_SFT = "sft"


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
    """Callback для периодической генерации текста и подсчета метрик.

    В режиме SFT считает ROUGE по парам prompt/completion и логирует таблицу генераций.
    В режиме CPT генерирует продолжения текстовых фрагментов для визуального контроля,
    ROUGE не считается (нет референсных таргетов).

    Режим определяется автоматически через data_cfg в on_fit_start (mode='auto'),
    либо задаётся явно через параметр mode ('cpt' | 'sft').
    """

    def __init__(
        self,
        model_name: str,
        num_random: int = 5,
        generation_batch_size: int = 2,
        generation_kwargs: dict[str, Any] | None = None,
        fixed_samples: list[dict[str, Any]] | None = None,
        mode: str = _MODE_AUTO,
    ) -> None:
        """Инициализирует коллбэк генерации.

        Args:
            model_name: Название архитектуры модели для тегирования в MLflow.
            num_random: Количество случайных примеров из валидации для генерации.
            generation_batch_size: Размер батча при генерации.
            generation_kwargs: Параметры генерации (temperature, top_p и т.д.).
            fixed_samples: Список фиксированных примеров для SFT-режима.
                Каждый элемент: {'prompt': str, 'target': str}.
                Игнорируется в CPT-режиме.
            mode: Режим работы коллбэка ('auto' | 'cpt' | 'sft').
                'auto' — определяется по наличию prompt_column в data_cfg.
        """
        self.model_name = model_name
        self.num_random = num_random
        self.generation_batch_size = generation_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.fixed_samples = fixed_samples or []
        self.mode = mode

        # Резолвится в on_fit_start
        self._resolved_mode: str | None = None
        self.rouge_metric: Any | None = None

        # Будут заполнены в on_fit_start
        self.generator = None
        self.val_raw_dataset: list[dict[str, str]] = []

    def _resolve_mode(self, data_cfg: Any) -> str:
        """Определяет режим работы по data_cfg если mode='auto'.

        Args:
            data_cfg: Конфиг данных из trainer.datamodule.data_cfg.

        Returns:
            Строка 'cpt' или 'sft'.
        """
        if self.mode != _MODE_AUTO:
            return self.mode

        has_prompt = hasattr(data_cfg, "prompt_column") or (
            isinstance(data_cfg, dict) and "prompt_column" in data_cfg
        )
        resolved = _MODE_SFT if has_prompt else _MODE_CPT
        logger.info(f"GenerationEvaluationCallback: mode=auto → resolved={resolved}")
        return resolved

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Подготавливает генератор и выборку данных перед началом обучения."""
        from hydra.utils import instantiate

        from src.core.models.generator import HFTextGenerator

        self.generator = HFTextGenerator(
            model=pl_module.model,
            tokenizer=trainer.datamodule.tokenizer,
            generation_kwargs=self.generation_kwargs,
        )

        data_cfg = trainer.datamodule.data_cfg
        self._resolved_mode = self._resolve_mode(data_cfg)

        # Загружаем сырой датасет до трансформаций
        raw_datasets = instantiate(data_cfg.source).load()
        val_raw = raw_datasets.get("validation", raw_datasets["train"])
        n = min(self.num_random * 10, len(val_raw))

        if self._resolved_mode == _MODE_CPT:
            text_col = (
                data_cfg.get("text_column", "text")
                if isinstance(data_cfg, dict)
                else getattr(data_cfg, "text_column", "text")
            )
            # Берём первые 200 символов фрагмента как промпт для продолжения
            self.val_raw_dataset = [
                {"prompt": val_raw[i][text_col][:200], "response": ""} for i in range(n)
            ]
            logger.info(
                f"CPT режим: собрано {len(self.val_raw_dataset)} текстовых фрагментов "
                f"для контроля генерации. ROUGE считаться не будет."
            )
        else:
            # SFT: нужны пары prompt/completion, грузим ROUGE
            prompt_col = (
                data_cfg.get("prompt_column", "prompt")
                if isinstance(data_cfg, dict)
                else getattr(data_cfg, "prompt_column", "prompt")
            )
            target_col = (
                data_cfg.get("target_column", "completion")
                if isinstance(data_cfg, dict)
                else getattr(data_cfg, "target_column", "completion")
            )
            self.val_raw_dataset = [
                {"prompt": val_raw[i][prompt_col], "response": val_raw[i][target_col]}
                for i in range(n)
            ]
            self.rouge_metric = evaluate.load("rouge")
            logger.info(
                f"SFT режим: собрано {len(self.val_raw_dataset)} пар prompt/completion. "
                f"ROUGE будет считаться."
            )

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            mlflow_client.set_tag(run_id, "model_architecture", self.model_name)
            mlflow_client.set_tag(run_id, "task_type", f"causal_lm_{self._resolved_mode}")

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

    def _generate_chunks(self, prompts: list[str]) -> list[str]:
        """Генерирует тексты чанками для защиты от OOM.

        Args:
            prompts: Список промптов для генерации.

        Returns:
            Список сгенерированных текстов.
        """
        generated_texts = []
        for i in range(0, len(prompts), self.generation_batch_size):
            chunk_prompts = prompts[i : i + self.generation_batch_size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk_generated = self.generator.generate(chunk_prompts, **self.generation_kwargs)
            generated_texts.extend(chunk_generated)
        return generated_texts

    def _log_mlflow_table(
        self,
        trainer: pl.Trainer,
        df: pd.DataFrame,
    ) -> None:
        """Логирует таблицу генераций в MLflow.

        Args:
            trainer: Объект тренера PyTorch Lightning.
            df: DataFrame с результатами генерации.
        """
        if not (trainer.logger and hasattr(trainer.logger, "experiment")):
            return

        mlflow_client = trainer.logger.experiment
        run_id = trainer.logger.run_id
        epoch = trainer.current_epoch
        mlflow_client.log_table(
            run_id=run_id,
            data=df,
            artifact_file=f"generations/epoch_{epoch}_results.json",
        )

    def _run_sft_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Логика оценки для SFT: генерация + ROUGE + таблица в MLflow."""
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

        generated_texts = self._generate_chunks(prompts)

        rouge_results = self.rouge_metric.compute(
            predictions=generated_texts, references=targets, use_stemmer=True
        )
        val_rouge1 = self._extract_rouge_score(rouge_results["rouge1"])
        val_rougeL = self._extract_rouge_score(rouge_results["rougeL"])  # noqa N816
        avg_gen_len = sum(len(t.split()) for t in generated_texts) / len(generated_texts)

        pl_module.log("val_rouge1", val_rouge1, sync_dist=True)
        pl_module.log("val_rougeL", val_rougeL, sync_dist=True)
        pl_module.log("val_avg_gen_length", avg_gen_len, sync_dist=True)

        df = pd.DataFrame(
            {
                "Type": sample_types,
                "Prompt": prompts,
                "Target": targets,
                "Generated": generated_texts,
            }
        )
        self._log_mlflow_table(trainer, df)

    def _run_cpt_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Логика оценки для CPT: генерация продолжений без ROUGE."""
        actual_num_random = min(self.num_random, len(self.val_raw_dataset))
        random_raw = random.sample(self.val_raw_dataset, actual_num_random)
        prompts = [item["prompt"] for item in random_raw]

        generated_texts = self._generate_chunks(prompts)

        avg_gen_len = sum(len(t.split()) for t in generated_texts) / len(generated_texts)
        pl_module.log("val_avg_gen_length", avg_gen_len, sync_dist=True)

        # Логируем только промпт и продолжение — таргетов нет
        df = pd.DataFrame(
            {
                "Prompt (first 200 chars)": prompts,
                "Generated continuation": generated_texts,
            }
        )
        self._log_mlflow_table(trainer, df)
        logger.info("CPT: таблица продолжений залогирована в MLflow.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Выполняет генерацию текстов и расчет метрик в конце эпохи валидации."""
        if self._resolved_mode is None:
            logger.warning("GenerationEvaluationCallback: режим не определён, пропуск.")
            return

        logger.info(f"GenerationEvaluationCallback: запуск в режиме {self._resolved_mode}...")

        if self._resolved_mode == _MODE_SFT:
            self._run_sft_eval(trainer, pl_module)
        else:
            self._run_cpt_eval(trainer, pl_module)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
