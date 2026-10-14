FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

# Install dependencies only (no package build)
RUN uv pip install --system --no-build-isolation -c /dev/null \
    apscheduler fastapi httpx \
    opentelemetry-api opentelemetry-sdk \
    pydantic pydantic-settings uvicorn

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
