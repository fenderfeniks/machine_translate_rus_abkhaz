# tests/conftest.py
import asyncio
import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# Настройка окружения до импортов
os.environ["PROJECT_ROOT"] = os.getcwd()
os.environ.setdefault("PROJECT_NAME", "Test NLP Generator API")
os.environ.setdefault("PROJECT_VERSION", "0.1.0")
os.environ.setdefault("API_PORT", "8000")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("HF_TOKEN", "test_token")
os.environ.setdefault("TG_BOT_TOKEN", "test_bot_token")

from src.api.rest.dependencies import get_generator, get_prompt_manager
from src.api.rest.server import create_app


# ---------------------------------------------------------------------------
# App fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def test_app():
    """Чистое приложение для каждого теста без загрузки ML-моделей."""
    app = create_app(load_ml=False)
    app.state.ml_models = {}
    return app


# ---------------------------------------------------------------------------
# ML mocks
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_generator() -> MagicMock:
    """Мок пайплайна — имитирует LLMGenerationPipeline.__call__"""
    generator = MagicMock(name="LLMGenerationPipeline")
    generator.return_value = [
        {"prompt": "test prompt", "generated_text": "test generated response"}
    ]
    # Мок для HFTextGenerator.generate() который используется внутри
    generator.generator = MagicMock()
    generator.generator.generate.return_value = ["test generated response"]
    return generator


@pytest.fixture
def mock_prompt_manager() -> MagicMock:
    """Мок PromptManager."""
    manager = MagicMock(name="PromptManager")
    manager.render.return_value = "Rendered prompt: test"
    return manager


@pytest.fixture
def override_ml_deps(test_app, mock_generator, mock_prompt_manager):
    """Переопределяем ML-зависимости для test_app."""
    test_app.dependency_overrides[get_generator] = lambda: mock_generator
    test_app.dependency_overrides[get_prompt_manager] = lambda: mock_prompt_manager
    test_app.state.ml_models = {"generator": mock_generator}
    test_app.state.prompt_manager = mock_prompt_manager
    test_app.state.gpu_semaphore = asyncio.Semaphore(1)
    yield test_app
    test_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_app, override_ml_deps):
    """Async HTTP-клиент с мокнутыми ML-зависимостями."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Shared model fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Минимальный реальный токенизатор для тестов коллатора и генерации."""
    pytest.importorskip("transformers", reason="transformers not installed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
