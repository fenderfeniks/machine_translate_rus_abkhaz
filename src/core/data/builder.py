# src/core/data/builder.py
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
from datasets import DatasetDict, load_from_disk
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class NLPDataModule(pl.LightningDataModule):
    """Универсальный DataModule для работы с NLP датасетами.

    Делегирует получение сырых данных `fetcher'у`, а подготовку 
    данных — пайплайну трансформаций. Обработанные данные 
    кэшируются на диске по хэшу конфигурации.
    """

    def __init__(self, data_cfg: Any, tokenizer: PreTrainedTokenizerBase) -> None:
        """Инициализирует DataModule.

        Args:
            data_cfg: Конфигурация данных (DictConfig из Hydra).
            tokenizer: Токенизатор для применения в трансформациях и коллаторе.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer

        # Хэшируем конфигурацию данных для DVC/кэширования
        hash_dict = {
            "source": OmegaConf.to_container(self.data_cfg.source, resolve=True),
            "transforms": OmegaConf.to_container(
                self.data_cfg.transforms, resolve=True
            ),
            "seed": self.data_cfg.get("seed"),
            "tokenizer_name": getattr(tokenizer, "name_or_path", "custom_tokenizer"),
        }

        hash_str = json.dumps(hash_dict, sort_keys=True)
        config_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()[:8]

        dataset_name = self.data_cfg.get("dataset_name", "nlp_dataset")

        self.processed_dir = Path(self.data_cfg.paths.processed_data_dir) / f"{dataset_name}_processed_{config_hash}"

    def prepare_data(self) -> None:
        """Подготавливает данные: скачивает, трансформирует и кэширует на диск."""
        if self.processed_dir.exists() and not self.data_cfg.get("force_reprocess", False):
            logger.info(
                "Нашли кэш обработанных данных: %s. Подготовка пропущена.", 
                self.processed_dir
            )
            return

        logger.info("Начинаем загрузку и применение трансформаций...")

        # Инстанцируем класс загрузчика
        fetcher = instantiate(self.data_cfg.source)
        # Явно вызываем метод загрузки данных
        raw_datasets = fetcher.load()

        if "validation" in raw_datasets and "test" in raw_datasets:
            raw_train = raw_datasets["train"]
            raw_val = raw_datasets["validation"]
            raw_test = raw_datasets["test"]
        else:
            split_ds = raw_datasets["train"].train_test_split(
                test_size=self.data_cfg.val_split_size * 2,
                seed=self.data_cfg.seed,
            )
            raw_train = split_ds["train"]
            val_test_split = split_ds["test"].train_test_split(
                test_size=0.5, seed=self.data_cfg.seed
            )
            raw_val = val_test_split["train"]
            raw_test = val_test_split["test"]

        # Инициализация трансформаций. Токенизатору прокидываем объект tokenizer
        transforms = []
        for transform_cfg in self.data_cfg.transforms:
            if "TokenizationTransform" in transform_cfg.get("_target_", ""):
                transforms.append(instantiate(transform_cfg, tokenizer=self.tokenizer))
            else:
                transforms.append(instantiate(transform_cfg))

        def _apply_transforms(dataset_split: Any) -> Any:
            for transform in transforms:
                dataset_split = transform(dataset_split)
            return dataset_split

        processed_dataset = DatasetDict(
            {
                "train": _apply_transforms(raw_train),
                "validation": _apply_transforms(raw_val),
                "test": _apply_transforms(raw_test),
            }
        )

        processed_dataset.save_to_disk(str(self.processed_dir))
        logger.info("Данные успешно обработаны и сохранены в %s", self.processed_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Загружает закэшированные данные для нужной стадии."""
        processed_dataset = load_from_disk(str(self.processed_dir))

        if stage == "fit" or stage is None:
            self.train_dataset = processed_dataset["train"]
            self.val_dataset = processed_dataset["validation"]

        if stage == "test" or stage is None:
            self.test_dataset = processed_dataset["test"]

        if stage == "validate" or stage is None:
            self.val_dataset = processed_dataset["validation"]

        self.collator = instantiate(self.data_cfg.collator, tokenizer=self.tokenizer)

    def train_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.train_dataset,
            collate_fn=self.collator,
            shuffle=True,
        )

    def val_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.val_dataset,
            collate_fn=self.collator,
            shuffle=False,
        )

    def test_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.test_dataset,
            collate_fn=self.collator,
            shuffle=False,
        )