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
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class NLPDataModule(pl.LightningDataModule):
    """Универсальный DataModule для работы с NLP датасетами.

    Делегирует получение сырых данных fetcher'у, разбиение — сплиттеру,
    а подготовку данных — пайплайну трансформаций. 
    Обработанные данные кэшируются на диске по хэшу конфигурации.
    """

    def __init__(self, data_cfg: Any, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer

        # Хэшируем конфигурацию данных для DVC/кэширования
        hash_dict = {
            "source": OmegaConf.to_container(self.data_cfg.source, resolve=True),
            "splitter": OmegaConf.to_container(self.data_cfg.splitter, resolve=True),
            # transforms — DictConfig с именованными ключами (validation, deduplication, ...),
            # порядок определяется defaults в sft/cpt.yaml
            "transforms": OmegaConf.to_container(self.data_cfg.transforms, resolve=True),
            "seed": self.data_cfg.get("seed"),
            "tokenizer_name": getattr(tokenizer, "name_or_path", "custom_tokenizer"),
        }

        hash_str = json.dumps(hash_dict, sort_keys=True)
        config_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()[:8]

        dataset_name = self.data_cfg.get("dataset_name", "nlp_dataset")
        self.processed_dir = Path(self.data_cfg.paths.processed_data_dir) / f"{dataset_name}_processed_{config_hash}"

    def _maybe_subsample(self, dataset: Any, name: str) -> Any:
        max_samples = self.data_cfg.get("max_samples", None)
        if max_samples is None:
            return dataset

        n = len(dataset)
        k = max(1, int(n * max_samples)) if isinstance(max_samples, float) else min(int(max_samples), n)

        logger.info("max_samples: %s %d → %d примеров (%s)", name, n, k, max_samples)
        return dataset.select(range(k))

    def prepare_data(self) -> None:
        if self.processed_dir.exists() and not self.data_cfg.get("force_reprocess", False):
            logger.info("Нашли кэш обработанных данных: %s. Подготовка пропущена.", self.processed_dir)
            return

        logger.info("Начинаем загрузку и применение трансформаций...")

        # 1. Загрузка данных
        fetcher = instantiate(self.data_cfg.source)
        raw_datasets = fetcher.load()

        # 2. Разбиение датасета (логика вынесена в сплиттер)
        splitter = instantiate(self.data_cfg.splitter)
        split_datasets = splitter(raw_datasets)

        # 3. Инициализация трансформаций
        # data.transforms — DictConfig: {validation: {...}, deduplication: {...}, ...}
        # Порядок применения = порядок defaults в sft.yaml / cpt.yaml (Hydra его сохраняет).
        # TokenizationTransform требует tokenizer как runtime-аргумент — пробрасываем отдельно.

        transforms = []
        for transform_cfg in self.data_cfg.transforms.values():
            if "TokenizationTransform" in transform_cfg.get("_target_", ""):
                transforms.append(instantiate(transform_cfg, tokenizer=self.tokenizer))
            else:
                transforms.append(instantiate(transform_cfg))

        def _apply_transforms(dataset_split: Any) -> Any:
            for transform in transforms:
                dataset_split = transform(dataset_split)
            return dataset_split

        # 4. Применение пайплайна
        processed_dataset = DatasetDict({
            "train": _apply_transforms(self._maybe_subsample(split_datasets["train"], "train")),
            "validation": _apply_transforms(self._maybe_subsample(split_datasets["validation"], "validation")),
            "test": _apply_transforms(self._maybe_subsample(split_datasets["test"], "test")),
        })

        processed_dataset.save_to_disk(str(self.processed_dir))
        logger.info("Данные успешно обработаны и сохранены в %s", self.processed_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        processed_dataset = load_from_disk(str(self.processed_dir))

        if stage == "fit" or stage is None:
            self.train_dataset = processed_dataset["train"]
            self.val_dataset = processed_dataset["validation"]

        if stage == "test" or stage is None:
            self.test_dataset = processed_dataset["test"]

        if stage == "validate" or stage is None:
            self.val_dataset = processed_dataset["validation"]

        self.collator = instantiate(self.data_cfg.collator, tokenizer=self.tokenizer)

    def _dataloader_kwargs(self) -> dict:
        dl_cfg = OmegaConf.to_container(self.data_cfg.dataloader, resolve=True)
        for key in ("_target_", "dataset", "collate_fn", "shuffle"):
            dl_cfg.pop(key, None)
        return dl_cfg

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collator,
            shuffle=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collator,
            shuffle=False,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collator,
            shuffle=False,
            **self._dataloader_kwargs(),
        )