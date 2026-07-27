# src/utils/config_schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Paths
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
# Logger
# ---------------------------------------------------------------------------


@dataclass
class MLFlowLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    experiment_name: str = "nlp_decoder_template"
    tracking_uri: str = "sqlite:///logs/mlflow.db"
    run_name: str = ""
    log_model: bool = False
    tags: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentConfig:
    name: str = "local"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class DataSourceConfig:
    _target_: str = "src.core.data.fetcher.RawDataFetcher"
    source_type: str = "local"
    raw_dir: str = ""
    # Опциональные поля — зависят от source_type
    dataset_name: str | None = None
    file_name: str | None = None
    token: str | None = None


@dataclass
class DataTransformConfig:
    _target_: str = ""
    # Общие параметры для всех трансформаций
    num_proc: int | None = None
    batch_size: int | None = None
    writer_batch_size: int | None = None
    # TokenizationTransform
    use_chat_template: bool | None = None
    text_column: str | None = None
    prompt_column: str | None = None
    target_column: str | None = None
    messages_column: str | None = None
    separator: str | None = None
    # LengthFilterTransform
    max_length: int | None = None
    # SequencePackingTransform
    packing_chunk_size: int | None = None
    drop_remainder: bool | None = None


@dataclass
class DataCollatorConfig:
    _target_: str = ""
    # InstructionDataCollator
    max_sequence_length: int | None = None
    mask_prompt: bool | None = None
    response_template: str | None = None
    # DataCollatorForLanguageModeling
    mlm: bool | None = None


@dataclass
class DataLoaderConfig:
    _target_: str = "torch.utils.data.DataLoader"
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class DataConfig:
    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    dataset_name: str = "nlp_dataset"
    max_length: int = 2048
    val_split_size: float = 0.1
    max_samples: Any = None
    seed: int = 42
    paths: Any = None
    force_reprocess: bool = False
    text_column: str | None = None
    prompt_column: str | None = None
    target_column: str | None = None
    messages_column: str | None = None
    preprocessing_num_workers: int = 4
    preprocessing_batch_size: int = 1000
    writer_batch_size: int = 200
    transforms: list[Any] = field(default_factory=list)
    collator: DataCollatorConfig = field(default_factory=DataCollatorConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ModelArchitectureConfig:
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True
    rope_scaling: dict[str, Any] | None = None


@dataclass
class TokenizerConfig:
    _target_: str = "src.core.models.tokenization.HFTokenizerBuilder"
    tokenizer_name: str = ""  # Подтягивается из model.model_name
    use_fast: bool = True
    padding_side: str = "right"
    add_eos_token: bool = False
    chat_template: str | None = None


@dataclass
class PEFTLoraConfig:
    _target_: str = "peft.LoraConfig"
    task_type: str = "CAUSAL_LM"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)


@dataclass
class QuantizationConfig:
    _target_: str = "transformers.BitsAndBytesConfig"
    # 4bit
    load_in_4bit: bool | None = None
    bnb_4bit_compute_dtype: str | None = None
    bnb_4bit_quant_type: str | None = None
    bnb_4bit_use_double_quant: bool | None = None
    # 8bit
    load_in_8bit: bool | None = None


@dataclass
class ModelBuilderConfig:
    _target_: str = "src.core.models.builder.HFModelBuilder"
    model_name_or_path: str = ""
    cache_dir: str | None = None
    auto_model_class: str = "transformers.AutoModelForCausalLM"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    rope_scaling: dict[str, Any] | None = None
    gradient_checkpointing: bool = True
    peft_config: Any | None = None  # PEFTLoraConfig или None (full FT)
    quantization_config: Any | None = None  # QuantizationConfig или None
    lora_resume_path: str | None = None


@dataclass
class LoraResumeConfig:
    enabled: bool = False
    run_id: str = ""
    artifact_path: str = "lora_weights"


@dataclass
class ModelConfig:
    model_name: str = "???"  # Обязателен, задаётся в architecture/*.yaml
    finetuning_type: str = "peft"  # full | peft | lm_head_only | frozen_embeddings
    compile: bool = False
    mlflow_model_name: str = "GenerativeLLM"
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    peft: Any | None = None  # PEFTLoraConfig или None
    quantization: Any | None = None  # QuantizationConfig или None
    builder: ModelBuilderConfig = field(default_factory=ModelBuilderConfig)
    lora_resume: LoraResumeConfig = field(default_factory=LoraResumeConfig)


# ---------------------------------------------------------------------------
# Trainer + Callbacks
# ---------------------------------------------------------------------------


@dataclass
class ModelCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = ""
    filename: str = "epoch_{epoch:02d}-val_loss_{val_loss:.4f}"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 2
    save_last: bool = True
    auto_insert_metric_name: bool = False


@dataclass
class LRMonitorConfig:
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: str = "step"


@dataclass
class DeviceStatsConfig:
    _target_: str = "pytorch_lightning.callbacks.DeviceStatsMonitor"


@dataclass
class ModelFreezingConfig:
    _target_: str = "src.training.callbacks.ModelFreezingCallback"
    finetuning_type: str = "peft"


@dataclass
class GenerationEvalConfig:
    _target_: str = "src.training.callbacks.GenerationEvaluationCallback"
    model_name: str = ""
    num_random: int = 5
    generation_batch_size: int = 2
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    fixed_samples: list[dict[str, str]] = field(default_factory=list)
    mode: str = "auto"


@dataclass
class CallbacksConfig:
    model_checkpoint: ModelCheckpointConfig = field(default_factory=ModelCheckpointConfig)
    lr_monitor: LRMonitorConfig = field(default_factory=LRMonitorConfig)
    device_stats: DeviceStatsConfig = field(default_factory=DeviceStatsConfig)
    model_freezing: ModelFreezingConfig = field(default_factory=ModelFreezingConfig)
    generation_eval: GenerationEvalConfig = field(default_factory=GenerationEvalConfig)


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 3
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
    log_every_n_steps: int = 10
    val_check_interval: float = 0.25
    logger: Any = None  # Подтягивается через ${logger}
    callbacks: Any = None
    # Опциональные переопределения из environment/*.yaml
    limit_train_batches: Any | None = None
    limit_val_batches: Any | None = None
    limit_test_batches: Any | None = None


# ---------------------------------------------------------------------------
# LightningModule + DataModule
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    _partial_: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    _target_: str = "transformers.get_cosine_schedule_with_warmup"
    _partial_: bool = True
    num_warmup_steps: int = 100


@dataclass
class ModelModuleConfig:
    _target_: str = "src.training.module.CausalLMLightningModule"
    optimizer_cfg: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler_cfg: SchedulerConfig | None = field(default_factory=SchedulerConfig)


@dataclass
class DataModuleConfig:
    _target_: str = "src.core.data.builder.NLPDataModule"
    _recursive_: bool = False
    data_cfg: Any = None  # Ссылка на DataConfig через ${data}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@dataclass
class TelegramWebhookConfig:
    path: str = "/webhook/telegram"
    url: str = ""


@dataclass
class TelegramConfig:
    bot_token: str = ""
    webhook_url: str = ""


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    domain: str = "http://localhost:8000"
    concurrency_limit: int = 1
    title: str = "Industrial NLP Template API"
    description: str = ""
    version: str = "0.1.0"
    cors_origins: list[str] = field(default_factory=list)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    telegram_webhook: TelegramWebhookConfig = field(default_factory=TelegramWebhookConfig)
    generation_template: str = "rag_qa"
    generation_static_context: str = ""
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Prompts + Strings
# ---------------------------------------------------------------------------


@dataclass
class PromptsConfig:
    rag_qa: str = ""
    summarization: str = ""
    translation: str = ""


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
# Inference (infer.py)
# ---------------------------------------------------------------------------


@dataclass
class InferenceQuantizationConfig:
    enabled: bool = False
    bits: int = 4


@dataclass
class InferenceConfig:
    quantization: InferenceQuantizationConfig = field(default_factory=InferenceQuantizationConfig)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    input_file: str | None = None
    output_file: str = "predictions.jsonl"


# ---------------------------------------------------------------------------
# Hydra
# ---------------------------------------------------------------------------


@dataclass
class HydraRunConfig:
    dir: str = ""


@dataclass
class HydraJobConfig:
    chdir: bool = True


@dataclass
class HydraConfig:
    run: HydraRunConfig = field(default_factory=HydraRunConfig)
    job: HydraJobConfig = field(default_factory=HydraJobConfig)


# ---------------------------------------------------------------------------
# Root Schema
# ---------------------------------------------------------------------------


@dataclass
class ConfigSchema:
    seed: int = 42
    project_name: str = "industrial_nlp_template"
    resume_training: bool = False
    paths: PathsConfig = field(default_factory=PathsConfig)
    logger: MLFlowLoggerConfig = field(default_factory=MLFlowLoggerConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # trainer: TrainerConfig = field(default_factory=TrainerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    strings: StringsConfig = field(default_factory=StringsConfig)

    model_module: ModelModuleConfig = field(default_factory=ModelModuleConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)

    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # eval.py
    ckpt_path: str | None = None
    metrics_output_path: str = "metrics.json"
    drift_threshold: float | None = None
    drift_metric_key: str = "test_perplexity"

    # infer.py — одиночный запрос через CLI
    text: str | None = None

    hydra: HydraConfig = field(default_factory=HydraConfig)
