# scripts/infer.py
import gc
import json
import logging
import time
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


load_dotenv()

from src.core.inference.generator import HFTextGenerator  # noqa
from src.utils.checkpoint_utils import load_checkpoint  # noqa
from src.utils.hydra_utils import setup_config  # noqa
from src.utils.logger import setup_logging  # noqa
from src.utils.mlflow import resolve_lora_resume_path  # noqa


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    logger.info("Загрузка токенизатора...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # === 1. ЗАГРУЗКА АДАПТЕРОВ (если нужно) ===
    # Квантование теперь применяется автоматически через cfg.model.quantization!
    resume_cfg = cfg.get("lora_resume", {})
    lora_resume_path = resolve_lora_resume_path(resume_cfg)

    if lora_resume_path:
        logger.info("LoRA адаптер будет загружен из: %s", lora_resume_path)
        # Динамически прокидываем путь в модификатор перед сборкой
        OmegaConf.update(
            cfg, "model.modifiers.finetuning.lora_resume_path", lora_resume_path, force_add=True
        )
    else:
        logger.warning("lora_resume не задан — инференс на базовой архитектуре.")

    logger.info("Загрузка модели...")
    builder = hydra.utils.instantiate(cfg.model.builder)
    builder.modifiers_cfg = cfg.model.get("modifiers")
    model = builder.build(tokenizer=tokenizer)

    # Опциональная загрузка кастомного чекпоинта поверх
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info("Подгрузка кастомных весов из: %s", ckpt_path)
        model = load_checkpoint(model, ckpt_path, device="cpu")

    generator = HFTextGenerator(
        model=model,
        tokenizer=tokenizer,
        generation_kwargs=cfg.get("inference", {}).get("generation_kwargs", {}),
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    input_file = cfg.get("inference", {}).get("input_file")
    output_file = cfg.get("inference", {}).get("output_file", "predictions.jsonl")

    # === 2. ПАКЕТНАЯ ИЛИ ОДИНОЧНАЯ ГЕНЕРАЦИЯ ===
    if input_file and Path(input_file).exists():
        logger.info("Запуск пакетного инференса (Batch) из файла: %s", input_file)
        with open(input_file, encoding="utf-8") as f:
            queries = [json.loads(line)["prompt"] for line in f if line.strip()]

        start_time = time.perf_counter()
        generated_texts = generator.generate(queries)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_tokens = sum(len(tokenizer.encode(t)) for t in generated_texts)
        tps = total_tokens / total_time if total_time > 0 else 0

        logger.info("Пакетная генерация завершена. Скорость: %.2f токенов/сек.", tps)

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps({"prompt": q, "generated": gen}, ensure_ascii=False) + "\n"
                for q, gen in zip(queries, generated_texts)  # noqa B905
            )
        logger.info("Результаты сохранены в %s", output_file)
    else:
        query = cfg.text or "Объясни, что такое Retrieval-Augmented Generation (RAG)."
        logger.info("Запуск одиночной генерации...")

        start_time = time.perf_counter()
        generated_texts = generator.generate([query])
        end_time = time.perf_counter()

        gen_text = generated_texts[0]
        gen_tokens = len(tokenizer.encode(gen_text))
        elapsed = end_time - start_time
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            "\n==================================================\n"
            "ПРОМПТ:\n%s\n"
            "--------------------------------------------------\n"
            "ОТВЕТ МОДЕЛИ:\n%s\n"
            "--------------------------------------------------\n"
            "ТЕЛЕМЕТРИЯ: %d токенов за %.2f сек | Скорость: %.2f t/s\n"
            "==================================================",
            query,
            gen_text,
            gen_tokens,
            elapsed,
            tps,
        )


if __name__ == "__main__":
    infer()
