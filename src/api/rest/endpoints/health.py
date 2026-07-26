# src/api/rest/endpoints/health.py
from fastapi import APIRouter


router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Эндпоинт для Kubernetes / Docker Healthcheck."""
    return {"status": "ok", "message": "ML API is running"}
