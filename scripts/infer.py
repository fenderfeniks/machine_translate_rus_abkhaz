# scripts/infer.py
import gc
import json
import logging
import time
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()

from src.core.models.generator import HFTextGenerator  # noqa E402
from src.utils.hydra_utils import setup_config  # noqa E402
from src.utils.logger import setup_logging  # noqa E402
from src.utils.mlflow_helpers import resolve_lora_resume_path  # noqa E402
from src.utils.quantization_utils import apply_inference_quantization  # noqa E402


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    """Скрипт для тестирования генерации (одиночной или пакетной).

    Выполняет инициализацию пайплайна с квантованием, генерирует
    ответы на основе конфига и выводит подробную телеметрию производительности.

    Args:
        cfg: Конфигурация Hydra (DictConfig).
    """
    cfg = setup_config(cfg)

    logger.info("Загрузка токенизатора...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # === 1. ДИНАМИЧЕСКОЕ КВАНТОВАНИЕ (Защита от OOM) ===
    cfg = apply_inference_quantization(cfg)

    # Загрузка LoRA адаптера из MLflow Registry (если включено в конфиге)
    resume_cfg = cfg.model.get("lora_resume", {})
    lora_resume_path = resolve_lora_resume_path(resume_cfg)
    if lora_resume_path:
        logger.info("LoRA адаптер будет загружен из: %s", lora_resume_path)
    else:
        logger.warning("lora_resume.enabled=false — инференс на базовой архитектуре без адаптера.")

    logger.info("Загрузка модели...")
    model = hydra.utils.instantiate(
        cfg.model.builder, tokenizer=tokenizer, lora_resume_path=lora_resume_path
    ).build()

    # Опциональная загрузка кастомного чекпоинта поверх (для отладки/экспериментов)
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info("Подгрузка кастомных весов из: %s", ckpt_path)
        from src.utils.checkpoint_utils import load_checkpoint

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

    # === 2. ПАКЕТНАЯ ОБРАБОТКА ИЛИ ОДИНОЧНЫЙ ПРОГОН ===
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

        # === 3. ТЕЛЕМЕТРИЯ ГЕНЕРАЦИИ ===
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
