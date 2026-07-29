# Russian → Abkhaz Machine Translation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.2%2B-792EE5?style=flat-square&logo=lightning)
![Hydra](https://img.shields.io/badge/Hydra-1.3-89B4FA?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2?style=flat-square&logo=mlflow)
![PEFT](https://img.shields.io/badge/PEFT_LoRA-0.8%2B-FF6F00?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Helm-326CE5?style=flat-square&logo=kubernetes)
![Airflow](https://img.shields.io/badge/Airflow-2.8%2B-017CEE?style=flat-square&logo=apacheairflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Production-ready пайплайн CPT + SFT дообучения генеративных LLM для машинного перевода с русского на абхазский язык с полным MLOps-циклом.

---

## Контекст задачи

Абхазский язык — один из наиболее морфологически сложных языков Кавказа и критически малоресурсный язык: менее 150 000 носителей, минимальное количество параллельных корпусов и практически полное отсутствие поддержки в популярных сервисах перевода. Большинство современных LLM не распознают уникальный алфавит — `ӷ ӡ қ ҟ ԥ ҭ ҳ ҵ ҷ ½ ҿ ҩ џ ь ә ҕ ҧ`.

Цель проекта — создание масштабируемого, production-ready пайплайна адаптации генеративных моделей к задаче перевода через этапы Continual Pre-Training и Supervised Fine-Tuning, с оценкой качества по метрике sacrebleu BLEU.

| Подход | BLEU |
|---|---|
| Копирование исходного текста (русский) | 5.2 |
| Данный проект (частичное обучение, ограничения Colab) | ~7 |
| Лучший известный результат на задаче | 17.6 |

Пайплайн спроектирован для масштабирования: все компоненты production-grade и запускаются без изменений кода на полноценной инфраструктуре (GPU L4 24 GB) через единственный CLI-оверрайд.

---

## Архитектура системы

```
                         Training Pipeline
                                 |
  Raw Data -> Transforms -> CPT -> SFT -> MLflow Registry -> Merge
      |            |          |     |           |               |
  fetcher.py  dedup/pack  train  train     promote.py     merge_lora

                                 |
                          Serving Stack
                                 |
     FastAPI REST  <->  LLMGenerationPipeline (SDK)
           |                     |
     Telegram Bot          Streamlit Demo
           |
     Prometheus -> Grafana

                                 |
                      Orchestration (Kubernetes)
                                 |
  Airflow DAG: weekly_llm_finetuning
    train -> merge -> evaluate -> Slack notify
  Airflow DAG: promote_llm_to_prod (ручной запуск)
    promote -> kubectl rollout restart
```

---

## Стек технологий

| Слой | Инструмент | Роль |
|---|---|---|
| Оркестрация обучения | PyTorch Lightning | Тренировочный цикл, коллбэки, чекпоинтинг |
| Управление конфигурацией | Hydra + OmegaConf | Композируемые конфиги, CLI-оверрайды |
| Трекинг экспериментов | MLflow (SQLite / server) | Метрики, таблицы генераций, Model Registry |
| Эффективное дообучение | PEFT / LoRA | Обучение только адаптеров; чекпоинт сохраняет только дельта-веса |
| Квантизация | BitsAndBytes | 4-bit NF4 double-quant для Colab; 8-bit для полной инфраструктуры |
| Смешанная точность | bf16-mixed | Обучение на GPU Ampere+ |
| Attention | Flash Attention 2 | Автоматический откат на SDPA при недоступности |
| Дедупликация данных | MinHash LSH (datasketch) | Удаление нечётких дубликатов (128 перестановок, Jaccard 0.9) |
| Версионирование данных | DVC | Воспроизводимость датасетов |
| Оценка качества | sacrebleu | BLEU логируется на каждом шаге валидации |
| REST API | FastAPI + uvicorn | `/api/v1/generate`, `/generate/stream` (SSE), `/health` |
| Бот | aiogram 3 | Telegram webhook-интеграция |
| Демо | Streamlit | Браузерный интерфейс с поддержкой стриминга |
| Наблюдаемость | Prometheus + Grafana | Задержка запросов, утилизация GPU, ML-метрики |
| Rate limiting | SlowAPI + Redis | 5 запросов/мин на клиента |
| Контейнеризация | Docker + docker-compose | api / trainer / airflow / prometheus / grafana / demo |
| Деплой в Kubernetes | Helm chart | `decoder-api-chart`: PVC, RBAC, Secrets, ConfigMap |
| CI | GitHub Actions + uv | ruff format, ruff lint, pytest на каждый PR |
| Пакетный менеджер | uv | Быстрая установка, lockfile (uv.lock) |
| Качество кода | ruff, mypy, pre-commit | Проверка на коммит и в CI |

---

## Структура проекта

```
machine_translate_rus_abkhaz/
|
+-- configs/                        # Дерево конфигов Hydra (без хардкодов)
|   +-- main.yaml                   # Корень: связывает data / model / trainer / logger
|   +-- data/
|   |   +-- cpt.yaml / sft.yaml     # Пайплайны данных под каждую задачу
|   |   +-- source/                 # Загрузчики: local / kaggle / hf / mixed
|   |   +-- transforms/             # deduplication / tokenization / packing / filtering
|   +-- model/
|   |   +-- architecture/           # Конфиги моделей: Qwen2.5-1.5B, Qwen3-4B, phi-4
|   |   +-- modifiers/              # LoRA (r=16, alpha=32) + EmbeddingResize
|   |   +-- quantization/           # 4bit NF4 double-quant / 8bit / none
|   +-- trainer/
|   |   +-- default.yaml            # max_steps, val_check_interval, bf16, grad clip
|   |   +-- callbacks/              # checkpoint / generation / early_stopping / lr_monitor
|   +-- logger/
|   |   +-- pylightning/            # MLflow PL logger (tracking_uri из конфига)
|   |   +-- registry/               # Model Registry: имя, artifact_path, флаги промоута
|   +-- environment/
|       +-- local.yaml              # CPU smoke-test: 200 семплов, max_steps=1500
|       +-- prod.yaml               # GPU: bf16, полный датасет, limit_val_batches=200
|
+-- src/
|   +-- core/
|   |   +-- data/
|   |   |   +-- builder.py          # NLPDataModule (PyTorch Lightning DataModule)
|   |   |   +-- fetcher.py          # RawDataFetcher: local / Kaggle / HuggingFace
|   |   |   +-- collators.py        # InstructionDataCollator с маскировкой prompt_len
|   |   |   +-- mixers.py           # Утилиты смешивания датасетов
|   |   |   +-- transforms/         # Модульные: tokenization / packing / dedup / filtering
|   |   +-- models/
|   |   |   +-- builder.py          # HFModelBuilder: fallback flash-attn, BnB quант, rope
|   |   |   +-- modifiers.py        # PEFTModifier (LoRA) + EmbeddingResizeModifier
|   |   +-- inference/
|   |   |   +-- generator.py        # HFTextGenerator (батчированная, чанковая генерация)
|   |   |   +-- response_cleaner.py # Удаление промпта, специальных токенов
|   |   +-- prompts/
|   |       +-- manager.py          # PromptManager (шаблоны из конфига Hydra)
|   +-- training/
|   |   +-- module.py               # CausalLMLightningModule (LoRA-aware чекпоинтинг)
|   |   +-- callbacks.py            # GenerationEvaluationCallback (BLEU + MLflow таблицы)
|   +-- api/
|   |   +-- rest/                   # FastAPI: generate / stream / health эндпоинты
|   |   |   +-- server.py           # App factory: lifespan, Prometheus, rate limiting
|   |   +-- tg_bot/                 # aiogram 3 Telegram бот (webhook + local polling)
|   +-- sdk/
|   |   +-- inference.py            # LLMGenerationPipeline — переиспользуемый SDK
|   +-- tools/
|   |   +-- promote.py              # Staging -> Production (сравнение val_loss)
|   |   +-- merge_lora.py           # Слияние LoRA + экспорт в safetensors
|   +-- utils/
|       +-- mlflow.py               # log_lora_to_mlflow, resolve_lora_resume_path
|       +-- checkpoint_utils.py     # Безопасная загрузка чекпоинтов
|       +-- hydra_utils.py          # Вспомогательные утилиты для конфигурации
|
+-- scripts/
|   +-- train.py                    # Точка входа Hydra: CPT / SFT
|   +-- eval.py                     # Автономный скрипт оценки
|   +-- infer.py                    # CLI инференс
|   +-- run_api.py                  # Точка входа uvicorn
|
+-- dags/
|   +-- retrain_model_dag.py        # Еженедельно: train -> merge -> eval -> Slack
|   +-- promote_to_prod.py          # Ручной шлюз: promote -> kubectl rollout restart
|   +-- batch_analytics.py          # Аналитика BLEU по production-логам
|   +-- quality_control.py          # Мониторинг качества данных
|   +-- system_maintenance.py       # Ротация кэша и артефактов
|
+-- helm/decoder-api-chart/         # Helm chart для деплоя в Kubernetes
|   +-- templates/                  # deployment / service / pvc / rbac / secrets / configmap
|
+-- notebooks/
|   +-- 01_eda_and_tokens.ipynb     # Анализ распределения длин, шума в данных
|   +-- 02_generation_baseline.ipynb
|   +-- 03_prompt_engineering.ipynb
|   +-- 04_peft_lora_sandbox.ipynb
|   +-- 05_evaluation_and_errors.ipynb
|   +-- 06_merge_and_export.ipynb
|
+-- tests/                          # pytest: api / core / training / dags / sdk
+-- demo/                           # Streamlit UI (поддержка SSE стриминга)
+-- deploy/                         # K8s манифесты мониторинга, переменные Airflow
+-- docker-compose.yml              # api / trainer / airflow / prometheus / grafana / demo
+-- Dockerfile                      # Мультицелевая сборка (api / training)
+-- Makefile                        # install / train / api / mlflow / docker_* команды
+-- pyproject.toml                  # uv-зависимости: core / training / api / orchestration
+-- .github/workflows/ci.yml        # uv + ruff + pytest на push/PR
```

---

## Этап 1 — Скрининг архитектур (Smoke Tests)

Перед полным CPT проведён 100-шаговый smoke-test кандидатов для выбора лучшей базовой модели под абхазскую адаптацию.

| Модель | Test Loss | Test Perplexity |
|---|---|---|
| Qwen3-4B | 3.779 | 44.1 |
| **Qwen3-4B-Instruct-2507** | **3.758** | **43.2** |
| phi-4-mini-instruct | 7.460 | 1756.8 |

phi-4-mini-instruct показала критическую нестабильность на нестандартных Unicode-токенах (всплеск перплексии до ~1757) — модель не способна представлять абхазский алфавит. Выбрана **Qwen3-4B-Instruct-2507** как лучшая по адаптивности к расширенной кириллице «из коробки».

---

## Этап 2 — Анализ данных и предобработка (CPT)

Анализ распределения длин токенов на монолингвальном абхазском корпусе (выборка 1000 предложений, токенизатор Qwen):

| Метрика | Токенов |
|---|---|
| Медиана | 70 |
| P95 | 167 |
| P99 | 217 |
| Максимум | 302 |

Выбран `packing_chunk_size = 512` — ближайшая степень двойки выше P99. При медиане 70 токенов в один чанк упаковывается ~7 предложений, что даёт модели реальный межпредложенческий контекст без значимой обрезки.

Анализ шума (1000 обработанных записей):

| Сигнал | Результат | Примечание |
|---|---|---|
| Лишние пробелы | 0% | Чисто |
| HTML-теги | 0% | Чисто |
| URL | 0% | Чисто |
| Непечатаемые символы | 100% | Ожидаемо — расширенная кириллица вне ASCII |

Данные не требовали активной очистки. `TextCleaningPipeline` запущен с пустым списком клинеров. Дедупликация: MinHash LSH (128 перестановок, Jaccard >= 0.9, 5-граммовые шинглы).

---

## Этап 3 — Continual Pre-Training (CPT)

CPT обучает модель абхазской морфологии, статистике скрипта и межтокенным зависимостям до введения задачи перевода.

Ключевые решения:

- `SequencePackingTransform` — устраняет паддинг, максимизирует утилизацию GPU
- Без chat template — цель: продолжение сырого текста
- EOS-токен как PAD — стандарт для CPT декодерных моделей
- `val_perplexity` логируется на каждом шаге валидации как основная метрика здоровья

Сравнение конфигураций по памяти и качеству:

| Конфигурация | VRAM | Perplexity @ 500 шагов |
|---|---|---|
| Qwen2.5-1.5B (bf16, без квантизации) | ~9 GB | >30, медленная сходимость |
| Qwen3-4B (8-bit) | ~14 GB | Быстрее сходится; итерация слишком медленная для Colab |
| **Qwen3-4B (4-bit NF4 double-quant)** | **~8.7 GB** | **28–29 (loss ~3.33)** |

Выбран Qwen3-4B 4-bit — оптимальный баланс сходимости и бюджета памяти. Ожидаемая перплексия при полном обучении на production-железе: ~15.

---

## Этап 4 — Supervised Fine-Tuning (SFT)

SFT обучает маппинг перевода на **147 894 параллельных парах** русский–абхазский.

Формат данных (разделитель включён в маскировку промпта):

```
Иисус даже отдал за людей жизнь, хотя многие его ненавидели
Перевод: [МАСКИРОВАНО] -> Иисус аӡәырҩы дшырцәымӷызгьы, ауаатәыҩса рзы иԥсҭазаараҵәҟьа дамеигӡеит
```

Маскировка промпта: `InstructionDataCollator` маскирует токены промпта и разделителя (`prompt_len`) значением `-100`. Loss считается исключительно по абхазскому таргету.

Параметры SFT:

| Параметр | Значение |
|---|---|
| Максимальная длина последовательности | 512 |
| Эффективный размер батча | 4 (batch=2 x accumulate=2) |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | q_proj, v_proj |
| Оптимизатор | AdamW (beta=0.9/0.999, wd=0.01) |
| Планировщик | Cosine warmup (100 шагов) |
| Точность | bf16-mixed |
| Потребление VRAM | ~14.2 GB |

Аблация learning rate:

| LR | Поведение |
|---|---|
| 2e-4 | Быстрый начальный спад; всплески val_loss после ~800 шагов (catastrophic forgetting) |
| 1e-5 | Гладкая кривая; слишком медленная сходимость за 1500 шагов |
| **5e-5** | Стабильная сходимость — выбран как оптимальный |

Промежуточный результат: ~7 BLEU на шаге 1000 — выше baseline копирования (5.2), что подтверждает работоспособность пайплайна.

---

## Система оценки качества

`GenerationEvaluationCallback` запускается каждые `val_check_interval` шагов и логирует:

- **sacrebleu BLEU** (шкала 0–100) — основная метрика; корректно работает с Unicode
- **avg_gen_length** — индикатор здоровья генерации
- **Таблицы генераций** — триплеты промпт / таргет / сгенерированный текст как артефакты MLflow (`generations/val_step_N_results.json`)

Примечание по RougeL: стандартная токенизация `rouge_score` не обрабатывает расширенную кириллицу (ԥ, ҳ, ӡ и др.), возвращая ноль вне зависимости от качества перевода. RougeL исключён из метрик; sacrebleu является единственным сигналом качества.

---

## MLflow Model Lifecycle

```
train.py
  +-- trainer.fit()
        +-- log_lora_to_mlflow()
              +-- model.save_pretrained()  -> adapter_config.json + adapter_model.safetensors
              +-- mlflow.log_artifacts()   -> lora_weights/ в артефактах run
              +-- mlflow.register_model()  -> Registry: {name}_LoRA vN
              +-- set_alias("Staging")

promote.py  (Hydra, tracking_uri из cfg)
  +-- get_model_version_by_alias("Staging")
        +-- сравнение тега val_loss с "Production"
        +-- [если лучше] set_alias("Production")

merge_lora.py  (Hydra)
  +-- resolve_lora_resume_path()  -> скачать адаптер из Registry
        +-- PeftModel.from_pretrained(base, adapter)
        +-- model.merge_and_unload()
        +-- save_pretrained()  -> models/merged_{model_name}/

Airflow: weekly_llm_finetuning DAG
  train (KubernetesPodOperator, GPU)
    -> merge (KubernetesPodOperator, CPU)
    -> evaluate
    -> Slack: "Staging готов — проверьте MLflow"

Airflow: promote_llm_to_prod DAG  (schedule=None, ручной запуск)
  promote -> kubectl rollout restart deployment/decoder-template-api
```

Все скрипты резолвят `tracking_uri` из `cfg.logger.pylightning.tracking_uri` — единственный источник правды, никаких хардкодов.

---

## Быстрый старт

```bash
# Установка (рекомендуется uv)
make install
# или: pip install -e ".[dev,training,api]"

# Smoke-test — CPU, 200 семплов, ~2 мин, GPU не нужен
python scripts/train.py environment=local data=sft model/architecture=Qwen2.5-1.5B

# CPT на полном корпусе (GPU)
python scripts/train.py environment=prod data=cpt model/architecture=Qwen3-4B-Instruct-2507

# SFT после CPT (загрузка LoRA-адаптера из Registry)
python scripts/train.py environment=prod data=sft \
  model/architecture=Qwen3-4B-Instruct-2507 \
  model.lora_resume.enabled=true \
  model.lora_resume.model_name=Qwen3-4B-Instruct-2507_LoRA \
  model.lora_resume.alias=Production

# Промоут Staging -> Production
python src/tools/promote.py model/architecture=Qwen3-4B-Instruct-2507

# Слияние LoRA с базовой моделью
python src/tools/merge_lora.py model/architecture=Qwen3-4B-Instruct-2507 \
  ~model.modifiers.embedding_resize

# Запуск REST API
make api

# Запуск MLflow UI
make mlflow

# Docker: полный стек (api + airflow + prometheus + grafana + demo)
make docker_airflow && make docker_api
```

---

## Docker-сервисы

| Сервис | Порт | Описание |
|---|---|---|
| api | 8000 | FastAPI: generate / stream / health / metrics |
| trainer | — | Батчевое обучение (profile: training) |
| airflow | 8080 | UI оркестрации DAG |
| prometheus | 9090 | Сбор метрик |
| grafana | 3000 | Дашборды (admin/admin) |
| demo | 8501 | Streamlit браузерный интерфейс |

---

## Тесты

```bash
pytest tests/ -v
```

Покрытие: `tests/training/` (module, callbacks), `tests/core/` (transforms, collators, cleaners), `tests/api/` (endpoints, schemas), `tests/dags/` (структура DAG, K8s-контракты), `tests/sdk/` (inference pipeline). CI запускается на каждый push/PR через GitHub Actions.

---

## Ограничения и план развития

Проект разработан в условиях ограничений Google Colab (~15 GB VRAM). Разрыв между достигнутым (~7 BLEU) и лучшим известным результатом (17.6 BLEU) объясняется прежде всего неполным обучением, а не архитектурой.

| Причина | Детали |
|---|---|
| Неполный CPT | Достигнутая перплексия ~30–40 vs. ожидаемые ~15 при полном прогоне |
| Низкое покрытие SFT | 2000 шагов x батч 4 = ~8k примеров из 147k |
| 4-bit квантизация | Снижает эффективную ёмкость модели по сравнению с bf16 |

Планы (при наличии L4 24 GB):

- Полный CPT до перплексии < 15 (bf16, без квантизации)
- 3–5 полных эпох SFT на полном параллельном корпусе
- Подбор LoRA rank/alpha через Optuna (r: 16 -> 128)
- Расширение target modules: добавить k_proj, o_proj, gate_proj
- Back-translation: синтетические пары ru->ab из монолингвального абхазского
- Расширение словаря токенизатора под абхазские символы

---

## Требования

- Python 3.10–3.11
- PyTorch 2.x
- CUDA 11.8+ (GPU Ampere рекомендуется для bf16 + Flash Attention 2)
- Полный список зависимостей: `pyproject.toml`

---

Автор: Максим Новиков
