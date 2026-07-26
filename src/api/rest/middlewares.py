# src/api/rest/middlewares.py
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class RequestTimeLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования времени выполнения запросов инференса."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Перехватывает запрос, замеряет время и пишет в лог.

        Args:
            request: Входящий HTTP-запрос.
            call_next: Функция вызова следующего обработчика в цепочке.

        Returns:
            Ответ сервера с добавленным заголовком X-Model-Process-Time.
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        if "generate" in request.url.path:
            logger.info(
                "[%s] %s | Статус: %d | Время инференса: %.2f сек.",
                request.method,
                request.url.path,
                response.status_code,
                process_time,
            )

        response.headers["X-Model-Process-Time"] = str(process_time)
        return response


def setup_middlewares(app: FastAPI, cors_origins: list[str]) -> None:
    """Регистрирует все Middleware в приложении FastAPI.

    Args:
        app: Экземпляр приложения FastAPI.
        cors_origins: Список разрешенных источников для CORS.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTimeLoggingMiddleware)
