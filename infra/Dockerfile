# Imagen de desarrollo del backend.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Capa de dependencias: se reconstruye solo si cambia el pyproject o el lock.
COPY pyproject.toml uv.lock* README.md ./
COPY backend/ ./backend/
RUN uv sync --frozen --no-dev || uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
