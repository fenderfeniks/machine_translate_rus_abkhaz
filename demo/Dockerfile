# demo/Dockerfile
FROM python:3.10-slim

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копируем только зависимости для кэширования слоя
COPY demo/requirements.txt .
RUN uv pip install --system -r requirements.txt

# Копируем сам код интерфейса
COPY demo/app.py .

EXPOSE 8501

# Команда запуска Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]