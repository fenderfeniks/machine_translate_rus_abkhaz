# tests/api/test_endpoints.py
"""Тесты REST-эндпоинтов генеративного API."""

import os
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        response = await async_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_has_status_field(self, async_client):
        response = await async_client.get("/health")
        assert "status" in response.json()


class TestGenerateEndpointSuccess:
    @pytest.mark.asyncio
    async def test_returns_200_with_valid_prompt(self, async_client):
        response = await async_client.post(
            "/api/v1/generate", json={"prompt": "Explain gradient descent."}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_contains_generated_text(self, async_client):
        response = await async_client.post("/api/v1/generate", json={"prompt": "What is Python?"})
        assert "generated_text" in response.json()

    @pytest.mark.asyncio
    async def test_generated_text_is_string(self, async_client):
        response = await async_client.post("/api/v1/generate", json={"prompt": "Tell me a joke."})
        assert isinstance(response.json()["generated_text"], str)

    @pytest.mark.asyncio
    async def test_unicode_prompt_accepted(self, async_client):
        response = await async_client.post(
            "/api/v1/generate", json={"prompt": "Объясни что такое нейронная сеть."}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_long_prompt_within_limit(self, async_client):
        response = await async_client.post("/api/v1/generate", json={"prompt": "word " * 100})
        assert response.status_code == 200


class TestGenerateEndpointValidation:
    @pytest.mark.asyncio
    async def test_missing_prompt_returns_422(self, async_client):
        response = await async_client.post("/api/v1/generate", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_field_name_returns_422(self, async_client):
        response = await async_client.post("/api/v1/generate", json={"text": "some text"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_null_body_returns_422(self, async_client):
        response = await async_client.post("/api/v1/generate", json=None)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_numeric_prompt_returns_422(self, async_client):
        response = await async_client.post("/api/v1/generate", json={"prompt": 42})
        assert response.status_code == 422


class TestGenerateEndpointAuth:
    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_403(self, test_app, override_ml_deps):
        os.environ["API_KEY"] = "correct-key"
        try:
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/generate", json={"prompt": "test"}, headers={"X-API-Key": "wrong-key"}
                )
            assert response.status_code == 403
        finally:
            del os.environ["API_KEY"]

    @pytest.mark.asyncio
    async def test_correct_api_key_returns_200(self, test_app, override_ml_deps):
        os.environ["API_KEY"] = "correct-key"
        try:
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/generate",
                    json={"prompt": "test"},
                    headers={"X-API-Key": "correct-key"},
                )
            assert response.status_code == 200
        finally:
            del os.environ["API_KEY"]


class TestGenerateEndpointMLUnavailable:
    @pytest.mark.asyncio
    async def test_returns_503_when_generator_is_none(self, test_app):
        from src.api.rest.dependencies import get_generator

        test_app.dependency_overrides[get_generator] = lambda: None
        transport = ASGITransport(app=test_app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/v1/generate", json={"prompt": "test"})
            assert response.status_code in (503, 500)
        finally:
            test_app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_503_when_generator_raises(self, test_app):
        from src.api.rest.dependencies import get_generator

        broken = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        test_app.dependency_overrides[get_generator] = lambda: broken
        transport = ASGITransport(app=test_app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/v1/generate", json={"prompt": "test"})
            assert response.status_code in (503, 500)
        finally:
            test_app.dependency_overrides.clear()
