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


class GenerationEvaluationCallback(pl.Callback):
    """Callback для периодической генерации текста и подсчета метрик."""

    def __init__(
        self,
        model_name: str,
        num_random: int = 5,
        generation_batch_size: int = 2,
        generation_kwargs: dict[str, Any] | None = None,
        fixed_samples: list[dict[str, Any]] | None = None,
        mode: str = _MODE_AUTO,
    ) -> None:
        self.model_name = model_name
        self.num_random = num_random
        self.generation_batch_size = generation_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.fixed_samples = fixed_samples or []
        self.mode = mode
        self._env_ready: dict[str, bool] = {"val": False, "test": False}
        self._resolved_mode: str | None = None
        self.rouge_metric: Any | None = None
        self.bleu_metric: Any | None = None
        self.generator = None

        # Разделяем датасеты для валидации и теста
        self.eval_datasets: dict[str, list[dict[str, str]]] = {"val": [], "test": []}

    def _resolve_mode(self, data_cfg: Any) -> str:
        if self.mode != _MODE_AUTO:
            return self.mode

        task_val = None
        if isinstance(data_cfg, dict):
            task_val = data_cfg.get("task")
        else:
            task_val = getattr(data_cfg, "task", None)

        if task_val in [_MODE_SFT, _MODE_CPT]:
            resolved = task_val
        else:
            if isinstance(data_cfg, dict):
                has_prompt = bool(data_cfg.get("prompt_column"))
            else:
                has_prompt = bool(getattr(data_cfg, "prompt_column", None))
            resolved = _MODE_SFT if has_prompt else _MODE_CPT

        logger.info(f"GenerationEvaluationCallback: mode=auto → resolved={resolved}")
        return resolved

    def _setup_eval_env(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Инициализирует генератор и подготавливает нужный датасет (val или test)."""
        if self._env_ready[stage]:
            logger.info(f"_setup_eval_env: stage={stage} уже инициализирован, пропускаем.")
            return
        from src.core.inference.generator import HFTextGenerator

        if self.generator is None:
            self.generator = HFTextGenerator(
                model=pl_module.model,
                tokenizer=trainer.datamodule.tokenizer,
                generation_kwargs=self.generation_kwargs,
            )

        data_cfg = trainer.datamodule.data_cfg
        if self._resolved_mode is None:
            self._resolved_mode = self._resolve_mode(data_cfg)

        dataset_key = "validation" if stage == "val" else "test"

        if hasattr(trainer.datamodule, "datasets") and dataset_key in trainer.datamodule.datasets:
            raw_data = trainer.datamodule.datasets[dataset_key]
        else:
            from hydra.utils import instantiate

            logger.warning(
                f"Сырой датасет '{dataset_key}' не найден в памяти, загружаем с диска (source)..."
            )
            raw_datasets = instantiate(data_cfg.source).load()
            raw_data = raw_datasets.get(dataset_key, raw_datasets["train"])

        n = min(self.num_random * 10, len(raw_data))

        if self._resolved_mode == _MODE_CPT:
            text_col = (
                data_cfg.get("text_column", "text")
                if isinstance(data_cfg, dict)
                else getattr(data_cfg, "text_column", "text")
            )
            self.eval_datasets[stage] = [
                {"prompt": raw_data[i][text_col][:200], "response": ""} for i in range(n)
            ]
        else:
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
            self.eval_datasets[stage] = [
                {"prompt": raw_data[i][prompt_col], "response": raw_data[i][target_col]}
                for i in range(n)
            ]
            if self.rouge_metric is None:
                self.rouge_metric = evaluate.load("rouge")
            if self.bleu_metric is None:
                self.bleu_metric = evaluate.load("sacrebleu")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            mlflow_client.set_tag(run_id, "model_architecture", self.model_name)
            mlflow_client.set_tag(run_id, "task_type", f"causal_lm_{self._resolved_mode}")

        self._env_ready[stage] = True

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._setup_eval_env(trainer, pl_module, stage="val")

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._setup_eval_env(trainer, pl_module, stage="test")

    def _extract_rouge_score(self, score: Any) -> float:
        if hasattr(score, "mid"):
            return float(score.mid.fmeasure)
        elif isinstance(score, (list, tuple)) and len(score) > 0:
            return float(score[0])
        return float(score)

    def _generate_chunks(self, prompts: list[str]) -> list[str]:
        generated_texts = []
        for i in range(0, len(prompts), self.generation_batch_size):
            chunk_prompts = prompts[i : i + self.generation_batch_size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                chunk_generated = self.generator.generate(chunk_prompts, **self.generation_kwargs)
            generated_texts.extend(chunk_generated)
        return generated_texts

    def _log_mlflow_table(self, trainer: pl.Trainer, df: pd.DataFrame, stage: str) -> None:
        if not (trainer.logger and hasattr(trainer.logger, "experiment")):
            return

        mlflow_client = trainer.logger.experiment
        run_id = trainer.logger.run_id
        step = trainer.global_step
        mlflow_client.log_table(
            run_id=run_id,
            data=df,
            artifact_file=f"generations/{stage}_step_{step}_results.json",
        )

    def _run_sft_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        dataset = self.eval_datasets[stage]
        actual_num_random = min(self.num_random, len(dataset))
        random_raw = random.sample(dataset, actual_num_random)

        random_samples = [
            {"prompt": item["prompt"], "target": item["response"], "type": "Random"}
            for item in random_raw
        ]

        # Фиксированные примеры логируем только на валидации
        if stage == "val" and self.fixed_samples:
            fixed_samples = [
                {"prompt": item["prompt"], "target": item["target"], "type": "Fixed"}
                for item in self.fixed_samples
            ]
        else:
            fixed_samples = []

        eval_batch = fixed_samples + random_samples
        if not eval_batch:
            return

        prompts = [s["prompt"] for s in eval_batch]
        targets = [s["target"] for s in eval_batch]
        sample_types = [s["type"] for s in eval_batch]

        generated_texts = self._generate_chunks(prompts)

        # ROUGE
        rouge_results = self.rouge_metric.compute(
            predictions=generated_texts, references=targets, use_stemmer=True
        )
        val_rouge1 = self._extract_rouge_score(rouge_results["rouge1"])
        val_rougeL = self._extract_rouge_score(rouge_results["rougeL"])  # noqa

        # BLEU (оборачиваем таргеты в списки для sacrebleu)
        formatted_targets = [[t] for t in targets]
        bleu_results = self.bleu_metric.compute(
            predictions=generated_texts, references=formatted_targets
        )
        val_bleu = bleu_results["score"]

        avg_gen_len = sum(len(t.split()) for t in generated_texts) / len(generated_texts)

        # Вывод в консоль с динамическим префиксом (val_ или test_)
        pl_module.log(f"{stage}_rouge1", val_rouge1, sync_dist=True, prog_bar=True)
        pl_module.log(f"{stage}_rougeL", val_rougeL, sync_dist=True, prog_bar=True)
        pl_module.log(f"{stage}_bleu", val_bleu, sync_dist=True, prog_bar=True)
        pl_module.log(f"{stage}_avg_gen_length", avg_gen_len, sync_dist=True)

        df = pd.DataFrame(
            {
                "Type": sample_types,
                "Prompt": prompts,
                "Target": targets,
                "Generated": generated_texts,
            }
        )
        self._log_mlflow_table(trainer, df, stage)

    def _run_cpt_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        dataset = self.eval_datasets[stage]
        actual_num_random = min(self.num_random, len(dataset))
        random_raw = random.sample(dataset, actual_num_random)
        prompts = [item["prompt"] for item in random_raw]

        if not prompts:
            return

        generated_texts = self._generate_chunks(prompts)

        avg_gen_len = sum(len(t.split()) for t in generated_texts) / len(generated_texts)
        pl_module.log(f"{stage}_avg_gen_length", avg_gen_len, sync_dist=True, prog_bar=True)

        df = pd.DataFrame(
            {
                "Prompt (first 200 chars)": prompts,
                "Generated continuation": generated_texts,
            }
        )
        self._log_mlflow_table(trainer, df, stage)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._resolved_mode is None or trainer.sanity_checking:
            return

        logger.info(
            f"GenerationEvaluationCallback: запуск валидации в режиме {self._resolved_mode}..."
        )

        if self._resolved_mode == _MODE_SFT:
            self._run_sft_eval(trainer, pl_module, stage="val")
        else:
            self._run_cpt_eval(trainer, pl_module, stage="val")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._resolved_mode is None:
            return

        logger.info(f"GenerationEvaluationCallback: запуск теста в режиме {self._resolved_mode}...")

        if self._resolved_mode == _MODE_SFT:
            self._run_sft_eval(trainer, pl_module, stage="test")
        else:
            self._run_cpt_eval(trainer, pl_module, stage="test")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
