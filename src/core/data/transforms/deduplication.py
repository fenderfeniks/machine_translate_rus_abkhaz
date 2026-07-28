# src/core/data/transforms/deduplication.py
import hashlib
import logging
import re
from typing import Any, Optional

from datasets import Dataset as HFDataset

from src.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)

# Пытаемся импортировать datasketch, оставляем предупреждение если его нет
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash, MinHashLSH = None, None
    logger.warning("Библиотека datasketch не установлена. MinHashDeduplicationTransform будет недоступен.")


class ExactDeduplicationTransform(BaseDatasetTransform):
    """Отсеивает полные дубликаты текстов на основе точного MD5-хэширования."""

    def __init__(
        self,
        text_column: Optional[str] = "text",
        prompt_column: Optional[str] = "prompt",
        num_proc: int = 4,
    ) -> None:
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.num_proc = num_proc

    def __call__(self, dataset: HFDataset) -> HFDataset:
        initial_count = len(dataset)
        target_col = self.prompt_column if self.prompt_column in dataset.column_names else self.text_column
        
        if not target_col or target_col not in dataset.column_names:
            return dataset

        logger.info("Запуск точной дедупликации (MD5) по колонке '%s'...", target_col)

        def _compute_hash(example: dict[str, Any]) -> dict[str, str]:
            text_val = str(example[target_col])
            return {"_md5_hash": hashlib.md5(text_val.encode("utf-8")).hexdigest()}

        hashed_dataset = dataset.map(_compute_hash, num_proc=self.num_proc, desc="MD5 hashing")

        unique_hashes: set[str] = set()
        unique_indices: list[int] = []
        
        for idx, hash_val in enumerate(hashed_dataset["_md5_hash"]):
            if hash_val not in unique_hashes:
                unique_hashes.add(hash_val)
                unique_indices.append(idx)

        dedup_dataset = dataset.select(unique_indices)
        logger.info("Точная дедупликация: удалено %d дубликатов", initial_count - len(dedup_dataset))
        return dedup_dataset


class MinHashDeduplicationTransform(BaseDatasetTransform):
    """Отсеивает нечеткие дубликаты (Near-Duplicates) с помощью MinHash и LSH.
    
    Устойчив к изменению пунктуации, лишним пробелам и замене отдельных слов.
    """

    def __init__(
        self,
        text_column: Optional[str] = "text",
        prompt_column: Optional[str] = "prompt",
        num_perm: int = 128,
        threshold: float = 0.9,
        ngram_size: int = 5,
        num_proc: int = 4,
    ) -> None:
        """Инициализирует MinHash фильтр.

        Args:
            num_perm: Количество перестановок хэш-функции (точность vs скорость).
            threshold: Порог сходства Jaccard (0.0 до 1.0). 0.9 - очень похожие тексты.
            ngram_size: Размер n-грамм (шингов) для разбиения текста.
        """
        if MinHash is None:
            raise ImportError("Для использования MinHashDeduplicationTransform установите datasketch.")
            
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.num_proc = num_proc
        self.word_pattern = re.compile(r"(?u)\b\w+\b")

    def __call__(self, dataset: HFDataset) -> HFDataset:
        initial_count = len(dataset)
        target_col = self.prompt_column if self.prompt_column in dataset.column_names else self.text_column
        
        if not target_col or target_col not in dataset.column_names:
            return dataset

        logger.info(
            "Запуск MinHash (LSH) дедупликации по '%s' (threshold=%.2f, ngrams=%d)...", 
            target_col, self.threshold, self.ngram_size
        )

        def _compute_minhash(example: dict[str, Any]) -> dict[str, list[int]]:
            text = str(example[target_col]).lower()
            tokens = self.word_pattern.findall(text)
            
            m = MinHash(num_perm=self.num_perm)
            
            if len(tokens) < self.ngram_size:
                # Если текст короче n-граммы, хэшируем его целиком
                m.update(" ".join(tokens).encode('utf-8'))
            else:
                for i in range(len(tokens) - self.ngram_size + 1):
                    shingle = " ".join(tokens[i : i + self.ngram_size]).encode('utf-8')
                    m.update(shingle)
                    
            # Сохраняем значения хэша как обычный list[int], чтобы HF Dataset смог это сериализовать
            return {"_minhash_signature": m.hashvalues.tolist()}

        # 1. Параллельно считаем сигнатуры для всех текстов
        hashed_dataset = dataset.map(
            _compute_minhash, 
            num_proc=self.num_proc, 
            desc="Computing MinHash signatures"
        )

        # 2. Строим LSH индекс и ищем дубликаты
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        duplicates_to_remove: set[int] = set()

        for idx, signature_list in enumerate(hashed_dataset["_minhash_signature"]):
            if idx in duplicates_to_remove:
                continue
                
            # Восстанавливаем объект MinHash из списка интов
            m = MinHash(num_perm=self.num_perm, hashvalues=signature_list, scheme="legacy")
            
            # Ищем похожие документы в индексе
            result = lsh.query(m)
            
            if result:
                # Если нашли похожие, значит текущий документ — дубликат. Помечаем на удаление.
                duplicates_to_remove.add(idx)
            else:
                # Иначе это уникальный документ, добавляем его в индекс
                lsh.insert(str(idx), m)

        # 3. Фильтруем датасет
        unique_indices = [i for i in range(initial_count) if i not in duplicates_to_remove]
        dedup_dataset = dataset.select(unique_indices)
        
        logger.info("MinHash дедупликация: удалено %d дубликатов", initial_count - len(dedup_dataset))
        return dedup_dataset