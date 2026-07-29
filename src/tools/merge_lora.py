# src/tools/merge_lora.py
import gc
import logging
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logger import setup_logging
from src.utils.mlflow import resolve_lora_resume_path


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def merge_and_export(cfg: DictConfig) -> None:
    """Сливает LoRA адаптер с базовой моделью."""

    base_model_name = cfg.model.architecture.model_name_or_path
    cache_dir = cfg.paths.hf_cache_dir

    # Проверяем кеш перед загрузкой — HF хранит как models--<org>--<name>/
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path and cache_path.exists():
        hf_cache_name = "models--" + base_model_name.replace("/", "--")
        cached_model_dir = cache_path / hf_cache_name
        if cached_model_dir.exists():
            logger.info("Найден кеш базовой модели: %s — загружаем с диска.", cached_model_dir)
        else:
            logger.info(
                "Кеш не найден (%s) — базовая модель будет скачана в %s.",
                cached_model_dir,
                cache_dir,
            )
    else:
        logger.info("Загрузка базовой модели: %s (cache_dir: %s)", base_model_name, cache_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)

    # Берём tracking_uri из конфига Hydra — единственный источник правды
    tracking_uri = cfg.logger.pylightning.tracking_uri
    logger.info("MLflow tracking URI из конфига: %s", tracking_uri)

    mlflow_model_name = cfg.model.architecture.mlflow_model_name

    lora_cfg = OmegaConf.create(
        {
            "enabled": True,
            "model_name": f"{mlflow_model_name}_LoRA",
            "alias": "Staging",
            "artifact_path": cfg.logger.registry.artifact_path,
        }
    )

    logger.info("Поиск адаптера '%s' (алиас: %s)...", lora_cfg.model_name, lora_cfg.alias)
    lora_path = resolve_lora_resume_path(lora_cfg, tracking_uri=tracking_uri)

    logger.info("Навешивание LoRA адаптера на базовую модель...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Слияние весов (Merge and Unload)...")
    merged_model = model.merge_and_unload()

    if getattr(merged_model.generation_config, "pad_token_id", None) in (None, -1):
        merged_model.generation_config.pad_token_id = tokenizer.eos_token_id

    model_short_name = Path(base_model_name).name
    output_path = Path(cfg.paths.root_dir) / "models" / f"merged_{model_short_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Сохранение монолитной модели в: %s", output_path)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    del model, merged_model, base_model
    gc.collect()
    logger.info("Слияние успешно завершено!")


if __name__ == "__main__":
    merge_and_export()
