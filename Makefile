# Makefile
# ==========================================
# КОМАНДЫ ДЛЯ РАЗРАБОТКИ (Task Runner)
# ==========================================
.PHONY: help install train api mlflow clean docker_train docker_api docker_airflow docker_down

help:
	@echo "Доступные команды:"
	@echo "--- Локальная разработка ---"
	@echo "  make install         - Установить зависимости локально через uv"
	@echo "  make train           - Запустить обучение локально"
	@echo "  make api             - Запустить локальный сервер API (FastAPI)"
	@echo "  make mlflow          - Запустить локальный сервер MLflow UI"
	@echo "  make clean           - Очистить кэш, логи и временные файлы"
	@echo ""
	@echo "--- Docker окружение ---"
	@echo "  make docker_train    - Запустить изолированное обучение в контейнере"
	@echo "  make docker_api      - Запустить продакшен сервер API в Docker"
	@echo "  make docker_airflow  - Поднять локальный оркестратор (Airflow) в Docker"
	@echo "  make docker_down     - Остановить все Docker контейнеры"

# --- Локальные команды ---

# Убрали rag из extras
install:
	uv venv
	uv pip install -e ".[dev,training,api]"

train:
	@echo "🧠 Запуск локального обучения..."
	python -m scripts.train $(ARGS)

api:
	@echo "🚀 Запуск локального API (uvicorn)..."
	uvicorn scripts.run_api.main:app

mlflow:
	@echo "📊 Запуск MLflow UI..."
	mlflow ui --backend-store-uri sqlite:///logs/mlflow.db --default-artifact-root ./logs/mlartifacts --host 127.0.0.1 --port 5000

clean:
	@echo "🧹 Очистка временных файлов и кэша..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .ruff_cache/

# --- Docker команды ---

docker_train:
	@echo "🐳 Запуск изолированного обучения (Docker)..."
	docker compose run --rm trainer python -m scripts.train $(ARGS)

docker_api:
	@echo "🐳 Запуск API в Docker..."
	docker compose up -d --build api

docker_airflow:
	@echo "⏳ Поднятие Airflow в Docker..."
	docker compose up -d airflow

docker_down:
	@echo "🛑 Остановка всех Docker сервисов..."
	docker compose down