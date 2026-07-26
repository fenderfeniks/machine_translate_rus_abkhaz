# scripts/api.py
import logging
import os

import uvicorn
from dotenv import load_dotenv


load_dotenv()
from src.utils.logger import setup_logging  # noqa E402


setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Точка входа для запуска REST API сервера."""
    api_port = int(os.getenv("API_PORT", "8000"))

    logger.info("Запуск Uvicorn сервера на порту %d", api_port)
    uvicorn.run(
        "src.api.rest.server:app",
        host="0.0.0.0",
        port=api_port,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
