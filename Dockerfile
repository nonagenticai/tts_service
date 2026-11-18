# Multi-stage Dockerfile for tts_service
#
# Single-context build: w2v-common is resolved from pypiserver.test
# via the [[tool.uv.index]] pypiserver-internal index and [tool.uv.sources]
# in pyproject.toml. torch is resolved from the pytorch-cpu index.
#
# Build with: docker buildx build -f Dockerfile --add-host pypiserver.test:192.168.1.241 .

ARG BASE_REGISTRY=registry.test:5000
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Build-arg gating provider extras.
# - "full" (default, production): kokoro+kitten+torch+spacy baked in.
# - "stub": minimal image for test-namespace deploys where TTS_PROVIDER=stub
#   is set at runtime. Cuts pod image size from multi-GB to ~500MB so hp1
#   image-pull doesn't time out the deploy job.
ARG INSTALL_TTS_PROVIDERS=full

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps; w2v-common comes from pypiserver.test,
# torch from pytorch-cpu index. Stub variant skips kokoro/kitten extras.
RUN if [ "$INSTALL_TTS_PROVIDERS" = "stub" ]; then \
        uv sync --frozen --no-dev --no-install-project; \
    else \
        uv sync --frozen --no-dev --extra kokoro --no-install-project; \
    fi

# Now copy source and install the project itself
COPY src/ ./src/
COPY alembic.ini ./alembic.ini

RUN if [ "$INSTALL_TTS_PROVIDERS" = "stub" ]; then \
        uv sync --frozen --no-dev && \
        /app/.venv/bin/python -m ensurepip --upgrade; \
    else \
        uv sync --frozen --no-dev --extra kokoro && \
        /app/.venv/bin/python -m ensurepip --upgrade && \
        /app/.venv/bin/python -m spacy download en_core_web_sm; \
    fi

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 ttsservice && \
    mkdir -p /app /app/logs /app/data/audio && \
    chown -R ttsservice:ttsservice /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=ttsservice:ttsservice /app/.venv /app/.venv

# Copy application code
COPY --chown=ttsservice:ttsservice src/ ./src/
COPY --chown=ttsservice:ttsservice alembic.ini ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH"

USER 1000:1000

# Port 6194 (TTS Service)
EXPOSE 6194

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

CMD ["python", "-m", "src.asgi"]
