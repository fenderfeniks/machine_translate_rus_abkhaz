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

        self._resolved_mode: str | None = None
        self.rouge_metric: Any | None = None
        self.generator = None
        self.val_raw_dataset: list[dict[str, str]] = []

    def _resolve_mode(self, data_cfg: Any) -> str:
        if self.mode != _MODE_AUTO:
            return self.mode

        # Проверяем, что ключ есть И его значение не None
        if isinstance(data_cfg, dict):
            has_prompt = data_cfg.get("prompt_column") is not None
        else:
            has_prompt = getattr(data_cfg, "prompt_column", None) is not None

        resolved = _MODE_SFT if has_prompt else _MODE_CPT
        logger.info(f"GenerationEvaluationCallback: mode=auto → resolved={resolved}")
        return resolved

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from src.core.inference.generator import HFTextGenerator

        self.generator = HFTextGenerator(
            model=pl_module.model,
            tokenizer=trainer.datamodule.tokenizer,
            generation_kwargs=self.generation_kwargs,
        )

        data_cfg = trainer.datamodule.data_cfg
        self._resolved_mode = self._resolve_mode(data_cfg)

        # Оптимизация: берем датасет напрямую из DataModule, если он там есть
        if hasattr(trainer.datamodule, "datasets") and "validation" in trainer.datamodule.datasets:
            val_raw = trainer.datamodule.datasets["validation"]
        else:
            from hydra.utils import instantiate

            logger.warning("Сырой датасет не найден в памяти, загружаем с диска (source)...")
            raw_datasets = instantiate(data_cfg.source).load()
            val_raw = raw_datasets.get("validation", raw_datasets["train"])

        n = min(self.num_random * 10, len(val_raw))

        if self._resolved_mode == _MODE_CPT:
            text_col = (
                data_cfg.get("text_column", "text")
                if isinstance(data_cfg, dict)
                else getattr(data_cfg, "text_column", "text")
            )
            self.val_raw_dataset = [
                {"prompt": val_raw[i][text_col][:200], "response": ""} for i in range(n)
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
            self.val_raw_dataset = [
                {"prompt": val_raw[i][prompt_col], "response": val_raw[i][target_col]}
                for i in range(n)
            ]
            self.rouge_metric = evaluate.load("rouge")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            mlflow_client.set_tag(run_id, "model_architecture", self.model_name)
            mlflow_client.set_tag(run_id, "task_type", f"causal_lm_{self._resolved_mode}")

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
            # Явный no_grad обязателен: Lightning не гарантирует отключение
            # градиентного графа внутри кастомных callback-методов (в отличие от
            # validation_step где это делается автоматически). Без этого генерация
            # строит граф вычислений → OOM на длинных последовательностях.
            with torch.no_grad():
                chunk_generated = self.generator.generate(chunk_prompts, **self.generation_kwargs)
            generated_texts.extend(chunk_generated)
        return generated_texts

    def _log_mlflow_table(self, trainer: pl.Trainer, df: pd.DataFrame) -> None:
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
        actual_num_random = min(self.num_random, len(self.val_raw_dataset))
        random_raw = random.sample(self.val_raw_dataset, actual_num_random)
        prompts = [item["prompt"] for item in random_raw]

        generated_texts = self._generate_chunks(prompts)

        avg_gen_len = sum(len(t.split()) for t in generated_texts) / len(generated_texts)
        pl_module.log("val_avg_gen_length", avg_gen_len, sync_dist=True)

        df = pd.DataFrame(
            {
                "Prompt (first 200 chars)": prompts,
                "Generated continuation": generated_texts,
            }
        )
        self._log_mlflow_table(trainer, df)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # _resolved_mode=None означает что on_fit_start ещё не отработал.
        # Это происходит при num_sanity_val_steps > 0: Lightning прогоняет
        # валидацию ДО fit, чтобы поймать ошибки конфига — без инициализации генератора.
        if self._resolved_mode is None:
            return

        # Пропускаем sanity-check эпохи (trainer.sanity_checking — флаг Lightning)
        if trainer.sanity_checking:
            return

        logger.info(f"GenerationEvaluationCallback: запуск в режиме {self._resolved_mode}...")

        if self._resolved_mode == _MODE_SFT:
            self._run_sft_eval(trainer, pl_module)
        else:
            self._run_cpt_eval(trainer, pl_module)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
