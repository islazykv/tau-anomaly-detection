FROM python:3.13-slim AS base

# System deps for PyTorch CPU and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev extras)
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY run.py ./

# Expose default server port
EXPOSE 8000

# Default: serve model (user mounts checkpoint volume)
CMD ["uv", "run", "python", "run.py", "stage=serve", "model=ae"]
