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
from src.utils.mlflow_helpers import resolve_lora_resume_path


load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def merge_and_export(cfg: DictConfig) -> None:
    """Сливает LoRA адаптер с базовой моделью.

    Загружает веса адаптера из MLflow Registry (алиас Staging),
    применяет их к базовой модели, выполняет слияние и сохраняет
    монолитную модель на диск.
    """
    base_model_name = cfg.model.model_name
    cache_dir = cfg.paths.hf_cache_dir

    logger.info("Загрузка базовой модели: %s (cache_dir: %s)", base_model_name, cache_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)

    mlflow_model_name = cfg.model.get("mlflow_model_name", "GenerativeLLM")

    # Оборачиваем в DictConfig, так как resolve_lora_resume_path ожидает этот тип
    lora_cfg = OmegaConf.create(
        {
            "enabled": True,
            "model_name": f"{mlflow_model_name}_LoRA",
            "alias": "Staging",
            "artifact_path": "lora_weights",
        }
    )

    logger.info(
        "Поиск адаптера '%s' (алиас: %s) в MLflow...",
        lora_cfg.model_name,
        lora_cfg.alias,
    )
    lora_path = resolve_lora_resume_path(lora_cfg)

    logger.info("Навешивание LoRA адаптера на базовую модель...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Слияние весов (Merge and Unload)...")
    merged_model = model.merge_and_unload()

    # Фикс невалидного pad_token_id перед сохранением
    if getattr(merged_model.generation_config, "pad_token_id", None) in (None, -1):
        merged_model.generation_config.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token_id исправлен на eos_token_id: %s", tokenizer.eos_token_id)

    model_short_name = base_model_name.split("/")[-1]
    output_path = Path(cfg.paths.root_dir) / "models" / f"merged_{model_short_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Сохранение новой монолитной модели в: %s", output_path)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    del model
    del merged_model
    del base_model
    gc.collect()

    logger.info("Слияние успешно завершено!")


if __name__ == "__main__":
    merge_and_export()
