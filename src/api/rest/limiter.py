# src/api/rest/limiter.py
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_real_ip(request: Request) -> str:
    """Надежное извлечение реального IP-адреса клиента за Ingress/Proxy.

    Args:
        request: Входящий HTTP-запрос.

    Returns:
        IP-адрес клиента в виде строки.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Берем первый IP из списка (истинный клиент)
        return forwarded.split(",")[0].strip()
    # Fallback на стандартную функцию (если запрос пришел напрямую)
    return get_remote_address(request)


# Создаем глобальный инстанс лимитера
limiter = Limiter(
    key_func=get_real_ip,  # <-- Используем нашу функцию
    default_limits=["10/minute"],
    # Отключаем лимитер для тестов, чтобы CI не падал на 429 ошибке
    enabled=os.getenv("ENVIRONMENT") != "testing",
)
