# src/utils/config_schema.py
"""Structured Config схема для валидации Hydra-конфигурации.

Каждый датакласс соответствует одной секции в configs/.
Поля помечены MISSING (???) если обязательны и не имеют дефолта.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


# ---------------------------------------------------------------------------
# Paths  →  configs/paths/default.yaml
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    root_dir: str = "."
    data_dir: str = "${paths.root_dir}/data"
    hf_cache_dir: str = "${paths.data_dir}/cache/hf_models"
    processed_data_dir: str = "${paths.data_dir}/processed"
    model_dir: str = "${paths.root_dir}/models"
    log_dir: str = "${paths.root_dir}/logs"
    output_dir: str = "${paths.root_dir}/outputs"


# ---------------------------------------------------------------------------
# System  →  configs/system/default.yaml
# ---------------------------------------------------------------------------


@dataclass
class SystemConfig:
    num_proc: int = 4  # CPU для datasets.map
    num_workers: int = 4  # DataLoader workers (0 на Windows)
    pin_memory: bool = True  # false при num_workers=0


# ---------------------------------------------------------------------------
# Environment  →  configs/environment/{local,prod}.yaml
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentConfig:
    name: str = "prod"  # local | prod


# ---------------------------------------------------------------------------
# Logger  →  configs/logger/mlflow.yaml
# ---------------------------------------------------------------------------


@dataclass
class MLFlowRegistryConfig:
    model_name: str = "${model.architecture.mlflow_model_name}"
    register_on_success: bool = True
    artifact_path: str = "lora_weights"
    promote_to_staging: bool = True


@dataclass
class MLFlowLoggerConfig:
    _target_: str = "lightning.pytorch.loggers.MLFlowLogger"
    experiment_name: str = "nlp_decoder_template"
    tracking_uri: str = "sqlite:///${paths.log_dir}/mlflow.db"
    run_name: str = "${now:%Y-%m-%d_%H-%M-%S}"
    log_model: bool = False
    tags: dict[str, str] = field(default_factory=lambda: {"env": "${environment.name}"})


@dataclass
class RootLoggerConfig:
    pylightning: MLFlowLoggerConfig = field(default_factory=MLFlowLoggerConfig)
    registry: MLFlowRegistryConfig = field(default_factory=MLFlowRegistryConfig)


# ---------------------------------------------------------------------------
# Data  →  configs/data/{sft,cpt}.yaml
# ---------------------------------------------------------------------------


@dataclass
class DataSourceConfig:
    _target_: str = "src.core.data.fetcher.RawDataFetcher"
    source_type: str = "local"
    raw_dir: str = "${paths.data_dir}/raw"
    dataset_name: str | None = None
    file_name: str | None = None
    token: str | None = None


@dataclass
class DataSplitterConfig:
    _target_: str = "src.core.data.splitters.RandomDatasetSplitter"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = "${data.seed}"  # type: ignore[assignment]


@dataclass
class DataCollatorConfig:
    _target_: str = "src.core.data.collators.InstructionDataCollator"
    max_sequence_length: int = 2048
    mask_prompt: bool = False
    response_template: str | None = None
    tokenizer: Any = None


@dataclass
class DataLoaderConfig:
    batch_size: int = 8
    num_workers: int = "${system.num_workers}"  # type: ignore[assignment]
    pin_memory: bool = "${system.pin_memory}"  # type: ignore[assignment]


@dataclass
class CleanerConfig:
    _target_: str = "src.core.data.cleaners.TextCleaningPipeline"
    cleaners: Any = field(default_factory=list)


@dataclass
class DataConfig:
    # Источник и сплит
    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    splitter: DataSplitterConfig = field(default_factory=DataSplitterConfig)
    collator: DataCollatorConfig = field(default_factory=DataCollatorConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    # Базовые настройки
    dataset_name: str = "nlp_dataset"
    seed: int = "${seed}"  # type: ignore[assignment]  # тянем из корня
    max_samples: Any = None  # int | float | null
    force_reprocess: bool = False
    val_size: float = 0.1
    test_size: float = 0.1

    # Колонки датасета
    text_column: str | None = None
    prompt_column: str | None = None
    target_column: str | None = None

    # Параметры трансформаций — тянутся в transforms/*.yaml
    max_length: int = "${max_length}"  # type: ignore[assignment]  # из корня
    num_proc: int = "${system.num_proc}"  # type: ignore[assignment]  # из system
    batch_size: int = 1000
    use_chat_template: bool = False
    messages_column: str | None = None
    separator: str = "\n\n"
    writer_batch_size: int = 1000
    drop_remainder: bool = False

    # Пайплайн очистки
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)

    # Пути к обработанным данным (переопределяются в sft/cpt)
    paths: dict[str, str] = field(
        default_factory=lambda: {"processed_data_dir": "${paths.processed_data_dir}"}
    )

    # Список трансформаций — собирается в sft.yaml / cpt.yaml
    # transforms — DictConfig[str, Any]: ключи = имена подгрупп (validation, deduplication, ...)
    # Порядок ключей = порядок defaults в sft/cpt.yaml. НЕ list — Hydra мержит как dict.
    transforms: Any = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model  →  configs/model/default.yaml + subgroups
# ---------------------------------------------------------------------------


@dataclass
class ModelArchitectureConfig:
    # Обязательные — задаются в architecture/*.yaml
    model_name_or_path: str = MISSING
    mlflow_model_name: str = MISSING
    auto_model_class: str = "transformers.AutoModelForCausalLM"
    # Параметры загрузки
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = False
    rope_scaling: dict[str, Any] | None = None
    # gradient_checkpointing в architecture — legacy-поле из старых конфигов.
    # Реально используется только в modifiers/finetuning/{lora,full}.yaml.
    gradient_checkpointing: bool = True


@dataclass
class TokenizerConfig:
    _target_: str = "src.core.models.tokenization.HFTokenizerBuilder"
    tokenizer_name: str = "${model.architecture.model_name_or_path}"
    use_fast: bool = True
    padding_side: str = "right"
    add_eos_token: bool = False
    chat_template: str | None = None


@dataclass
class QuantizationConfig:
    # 4bit (configs/model/quantization/4bit.yaml)
    load_in_4bit: bool | None = None
    bnb_4bit_compute_dtype: str | None = None
    bnb_4bit_quant_type: str | None = None
    bnb_4bit_use_double_quant: bool | None = None
    # 8bit (configs/model/quantization/8bit.yaml)
    load_in_8bit: bool | None = None


@dataclass
class PEFTLoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class EmbeddingResizeModifierConfig:
    _target_: str = "src.core.models.modifiers.EmbeddingResizeModifier"
    tokenizer: Any = "${model.tokenizer}"  # type: ignore[assignment]


@dataclass
class PEFTModifierConfig:
    _target_: str = "src.core.models.modifiers.PEFTModifier"
    gradient_checkpointing: bool = True
    is_quantized: bool = True
    lora_resume_path: str | None = None
    peft_config: PEFTLoraConfig = field(default_factory=PEFTLoraConfig)


@dataclass
class FullFTModifierConfig:
    _target_: str = "src.core.models.modifiers.FullFinetuningModifier"
    gradient_checkpointing: bool = True


@dataclass
class ModelModifiersConfig:
    embedding_resize: EmbeddingResizeModifierConfig = field(
        default_factory=EmbeddingResizeModifierConfig
    )
    finetuning: PEFTModifierConfig | FullFTModifierConfig = field(
        default_factory=PEFTModifierConfig
    )


@dataclass
class LoraResumeConfig:
    enabled: bool = False
    run_id: str = ""
    artifact_path: str = "lora_weights"


@dataclass
class HFModelBuilderConfig:
    """Конфигурация параметров, передаваемых строго в HFModelBuilder.__init__"""

    _target_: str = "src.core.models.builder.HFModelBuilder"
    _recursive_: bool = False
    model_name_or_path: str = MISSING
    auto_model_class: str = "transformers.AutoModelForCausalLM"
    cache_dir: str = "${paths.data_dir}/weights"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    quantization_config: Any = None
    modifiers: Any = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Верхнеуровневый контейнер для конфигурации модели."""

    # Узел непосредственно для инстанциирования
    builder: HFModelBuilderConfig = field(default_factory=HFModelBuilderConfig)

    # Подгруппы — только для хранения параметров и интерполяций.
    # Они больше не будут передаваться в __init__ билдера.
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    quantization: Any = None
    lora_resume: LoraResumeConfig = field(default_factory=LoraResumeConfig)
    compile: bool = False

    # Оставляем modifiers на верхнем уровне, так как Hydra подтянет их
    # сюда через defaults list из yaml
    modifiers: Any = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler  →  configs/optimizer/ configs/scheduler/
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    _partial_: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    _target_: str = "transformers.get_cosine_schedule_with_warmup"
    _partial_: bool = True
    num_warmup_steps: int = 100


# ---------------------------------------------------------------------------
# Trainer + Callbacks  →  configs/trainer/
# ---------------------------------------------------------------------------


@dataclass
class ModelCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = "${paths.log_dir}/checkpoints"
    filename: str = "step={step}-val_loss={val_loss:.4f}"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 2
    save_last: bool = True
    auto_insert_metric_name: bool = False
    every_n_train_steps: int = 200
    save_on_train_epoch_end: bool = False


@dataclass
class LRMonitorConfig:
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: str = "step"


@dataclass
class RichProgressBarConfig:
    _target_: str = "pytorch_lightning.callbacks.RichProgressBar"


@dataclass
class EarlyStoppingConfig:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"
    min_delta: float = 0.001


@dataclass
class GenerationEvalConfig:
    _target_: str = "src.training.callbacks.GenerationEvaluationCallback"
    model_name: str = "${model.architecture.mlflow_model_name}"
    num_random: int = 5
    generation_batch_size: int = 2
    mode: str = "auto"
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    fixed_samples: list[dict[str, str]] = field(default_factory=list)


@dataclass
class CallbacksConfig:
    model_checkpoint: ModelCheckpointConfig = field(default_factory=ModelCheckpointConfig)
    lr_monitor: LRMonitorConfig = field(default_factory=LRMonitorConfig)
    rich_progress_bar: RichProgressBarConfig = field(default_factory=RichProgressBarConfig)
    generation_eval: GenerationEvalConfig = field(default_factory=GenerationEvalConfig)
    # early_stopping подключается опционально через +trainer/callbacks/early_stopping=default


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"

    # Железо
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "bf16-mixed"

    # Длина обучения (max_steps-режим — не max_epochs)
    max_steps: int = 2000
    max_epochs: int = -1

    # Валидация
    val_check_interval: int = 200
    check_val_every_n_epoch: Any = None

    # Батчинг и градиенты
    accumulate_grad_batches: int = 4
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"

    # Логирование
    log_every_n_steps: int = 10
    logger: Any = None  # подтягивается через ${logger}
    default_root_dir: str = "${paths.log_dir}"

    # Прочее
    num_sanity_val_steps: int = 2
    deterministic: bool = False

    # Переопределяются из environment/*.yaml
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0

    # callbacks — DictConfig при загрузке из Hydra, List после _resolve_trainer_callbacks.
    # Объявляем как Any чтобы схема не конфликтовала ни с тем ни с другим.
    callbacks: Any = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inference  →  configs/inference/default.yaml
# ---------------------------------------------------------------------------


@dataclass
class ResponseCleanerConfig:
    _target_: str = "src.core.models.response_cleaner.ResponseCleaner"
    strip_prompt: bool = True
    remove_special_tokens: bool = True
    remove_markdown_blocks: bool = False
    remove_extra_spaces: bool = True
    trim_incomplete_sentence: bool = True


@dataclass
class InferenceConfig:
    _target_: str = "src.core.models.generator.HFTextGenerator"
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
    )
    cleaner_cfg: ResponseCleanerConfig = field(default_factory=ResponseCleanerConfig)


# ---------------------------------------------------------------------------
# API  →  configs/api/
# ---------------------------------------------------------------------------


@dataclass
class TelegramConfig:
    bot_token: str = ""
    webhook_url: str = ""


@dataclass
class TelegramWebhookConfig:
    path: str = "/webhook/telegram"
    url: str = "${api.domain}${api.telegram_webhook.path}"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    domain: str = "http://localhost:8000"
    concurrency_limit: int = 1
    title: str = "Industrial NLP Template API"
    description: str = "NLP API for Text Generation and LLM Inference"
    version: str = "0.1.0"
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080",
        ]
    )
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    telegram_webhook: TelegramWebhookConfig = field(default_factory=TelegramWebhookConfig)
    generation_template: str = "rag_qa"
    generation_static_context: str = ""
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Prompts / Strings  →  configs/prompts/ configs/strings/
# ---------------------------------------------------------------------------


@dataclass
class PromptsConfig:
    rag_qa: str = ""
    summarization: str = ""
    translation: str = ""
    telegram_welcome: str = ""


@dataclass
class BotStringsConfig:
    welcome: str = ""
    error: str = ""
    processing: str = ""


@dataclass
class ErrorStringsConfig:
    gpu_unavailable: str = ""
    no_checkpoint: str = ""


@dataclass
class StringsConfig:
    bot: BotStringsConfig = field(default_factory=BotStringsConfig)
    errors: ErrorStringsConfig = field(default_factory=ErrorStringsConfig)


# ---------------------------------------------------------------------------
# Hydra  →  управляется Hydra автоматически
# ---------------------------------------------------------------------------


@dataclass
class HydraRunConfig:
    dir: str = "${paths.log_dir}/hydra/${now:%Y-%m-%d_%H-%M-%S}"


@dataclass
class HydraJobConfig:
    chdir: bool = True


@dataclass
class HydraConfig:
    run: HydraRunConfig = field(default_factory=HydraRunConfig)
    job: HydraJobConfig = field(default_factory=HydraJobConfig)


# ---------------------------------------------------------------------------
# Root Schema  →  configs/main.yaml
# ---------------------------------------------------------------------------


@dataclass
class ConfigSchema:
    # Глобальные скаляры — объявлены в main.yaml, тянутся через ${seed}, ${max_length}
    seed: int = 42
    max_length: int = 2048
    project_name: str = "industrial_nlp_template"
    resume_training: bool = False

    # Группы конфигов
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logger: RootLoggerConfig = field(default_factory=RootLoggerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: Any = field(default_factory=dict)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    strings: StringsConfig = field(default_factory=StringsConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)

    # eval.py / infer.py — CLI-параметры
    ckpt_path: str | None = None
    metrics_output_path: str = "metrics.json"
    drift_threshold: float | None = None
    drift_metric_key: str = "test_perplexity"
    text: str | None = None
