"""ASGI application for tts_service.

Built on the shared ``w2v_common.asgi.create_app`` factory: it wires the
canonical middleware stack (RequestId → Auth → SecurityHeaders → CORS →
error-envelope) and the BearerAuth/TenantIdHeader OpenAPI customization, so
this module only supplies the service-specific lifespan + routers.

Request-id correlation in structlog logs comes from the shared
``w2v_common.observability.structlog_bridge.add_request_id`` processor
(configured in ``src.logging_config``), which reads the request-id ContextVar
the factory's ``RequestIdMiddleware`` binds — replacing the old bespoke
RequestId middleware that bound structlog contextvars itself.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from redis.exceptions import RedisError
from w2v_common.asgi import create_app
from w2v_common.observability import database_endpoint, redis_endpoint

from src.cache.redis_client import close_redis, get_redis_client, init_redis
from src.config import get_settings
from src.database.connection import close_engine, get_engine
from src.logging_config import configure_logging
from src.providers.registry import create_registry
from src.services.metering import close_metering_redis, init_metering_redis
from src.services.storage_service import StorageService
from src.services.tts_service import TTSService

load_dotenv()

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)


# ============================================================================
# Health Check Logging Filter
# ============================================================================


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress noisy health-check request logs from uvicorn access logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for health-check paths to prevent them from being logged."""
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/healthz", "/readyz"]
        )


uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckLoggingFilter())


# ============================================================================
# Lifespan helpers
# ============================================================================


async def _init_database(app: FastAPI) -> None:
    """Connect to the database, create tables, and reset orphaned jobs."""
    try:
        db_engine = await get_engine()
        app.state.db_engine = db_engine
        logger.info("Database connected: %s", database_endpoint(settings.database.url))
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        raise

    from src.database.models import Base

    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")

    await _reset_orphaned_jobs()


async def _reset_orphaned_jobs() -> None:
    """Mark jobs left in 'processing' state from a previous crash as failed."""
    from sqlalchemy import select

    from src.database.models import TTSJob
    from src.database.session import get_db_session

    async with get_db_session() as db:
        orphaned = await db.execute(select(TTSJob).where(TTSJob.status == "processing"))
        orphaned_jobs = orphaned.scalars().all()
        if orphaned_jobs:
            for job in orphaned_jobs:
                job.status = "failed"
                job.error_message = "Service restarted during processing"
            await db.commit()
            logger.warning("Reset %d orphaned jobs to failed", len(orphaned_jobs))


async def _init_redis(app: FastAPI) -> Any:
    """Connect to Redis. Returns the client, or None if Redis is unavailable."""
    try:
        redis_client = await init_redis()
        await redis_client.ping()
        app.state.redis_client = get_redis_client()
        logger.info("Redis connected: %s", redis_endpoint(settings.redis.url))
        return redis_client
    except (ImportError, OSError, RuntimeError, ValueError, RedisError) as e:
        logger.error("Failed to connect to Redis: %s", e, exc_info=True)
        logger.warning("Continuing without Redis — caching/notifications will be disabled")
        return None


async def _prewarm_kokoro(registry: Any) -> None:
    """Eagerly initialize the Kokoro pipeline to fail fast on OOM."""
    if not (settings.kokoro.enabled and "kokoro" in registry.provider_names):
        return
    try:
        from src.providers.kokoro_provider import KokoroProvider

        provider = registry.get_provider("kokoro")
        if isinstance(provider, KokoroProvider):
            logger.info("Pre-warming Kokoro pipeline...")
            # Load the native extension modules HERE, on the main thread, before
            # handing pipeline construction to a worker thread below.
            #
            # DO NOT "tidy" these imports back into the ``asyncio.to_thread`` call.
            # ``torch._C`` is a single-phase-init C extension; importing such a
            # module off the main thread is a documented CPython fragility, and it
            # is precisely what made a ``patch.dict("sys.modules")`` eviction lethal
            # (libtorch re-ran ``initModule()`` -> SIGSEGV, exit 139, PR #92).
            # Doing it here also moves a multi-second native load off the first
            # request and into startup, where a failure is visible immediately.
            #
            # ``torch``/``kokoro`` are optional (``tts-service[kokoro]`` extra); a
            # missing extra raises ImportError, which the handler below already
            # tolerates exactly as it did when the import only happened in-thread.
            import torch  # noqa: F401
            from kokoro import KPipeline  # noqa: F401

            await asyncio.to_thread(provider._get_pipeline, "a")
            logger.info("Kokoro pipeline ready")
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Kokoro pre-warm failed: %s", e)


def _build_router(registry: Any) -> Any:
    """Construct the optional provider router. Returns None if disabled or init fails."""
    if not settings.routing.enabled:
        return None
    try:
        from src.services.router import CostEstimator, ProviderRouter

        cost_estimator = CostEstimator(settings.routing.cost_per_1k_chars)
        router_instance = ProviderRouter(
            registry=registry,
            cost_estimator=cost_estimator,
            quality_scores=settings.routing.quality_scores,
            default_strategy=settings.routing.default_strategy,
            max_fallback_attempts=settings.routing.max_fallback_attempts,
        )
        logger.info("Provider routing enabled (strategy=%s)", settings.routing.default_strategy)
        return router_instance
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Provider routing init failed: %s", e)
        return None


async def _init_notifications(app: FastAPI, redis_client: Any) -> Any:
    """Start the notification service. Returns None if init fails.

    Always assigns ``app.state.notification_service`` (the started service or
    ``None``) so ``_shutdown_hook`` can read it back off ``app.state`` and get
    the same value the old lifespan passed as a local.
    """
    try:
        from src.services.notification_service import NotificationService

        notification_service = NotificationService(redis_client=redis_client)
        await notification_service.start()
        app.state.notification_service = notification_service
        logger.info("NotificationService started")
        return notification_service
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("NotificationService init failed: %s", e)
        app.state.notification_service = None
        return None


def _init_connection_manager(app: FastAPI) -> None:
    """Create the WebSocket connection manager."""
    try:
        from src.services.connection_manager import ConnectionManager

        app.state.connection_manager = ConnectionManager(
            max_per_tenant=settings.websocket.max_connections_per_tenant
        )
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("ConnectionManager init failed: %s", e)


async def _shutdown(registry: Any, notification_service: Any, metering_redis: Any) -> None:
    """Tear down providers, notifications, Redis, metering, and the database."""
    logger.info("Shutting down TTS Service")
    try:
        await registry.close_all()
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.error("Error closing providers: %s", e, exc_info=True)
    if notification_service:
        try:
            await notification_service.stop()
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error("Error stopping notification service: %s", e, exc_info=True)
    await close_metering_redis(metering_redis)
    try:
        await close_redis()
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.error("Error closing Redis: %s", e, exc_info=True)
    try:
        await close_engine()
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.error("Error closing database: %s", e, exc_info=True)
    logger.info("TTS Service shutdown complete")


# ============================================================================
# create_app lifespan hooks
# ============================================================================


async def _startup(app: FastAPI) -> None:
    """Init DB, Redis, providers, storage, services (create_app on_startup)."""
    logger.info("=" * 70)
    logger.info("Starting TTS Service")
    logger.info("=" * 70)

    await _init_database(app)
    redis_client = await _init_redis(app)

    registry = create_registry(settings)
    app.state.provider_registry = registry
    await _prewarm_kokoro(registry)

    storage = StorageService()
    await storage.initialize()
    app.state.storage_service = storage

    router_instance = _build_router(registry)
    notification_service = await _init_notifications(app, redis_client)
    _init_connection_manager(app)

    metering_redis = await init_metering_redis(settings.metering.redis_url)
    app.state.metering_redis = metering_redis

    app.state.settings = settings
    app.state.tts_service = TTSService(
        registry=registry,
        storage=storage,
        redis_client=redis_client,
        router=router_instance,
        notification_service=notification_service,
        concurrency_settings=settings.concurrency,
        metering_redis=metering_redis,
    )

    logger.info("=" * 70)
    logger.info("TTS Service started successfully")
    logger.info("Environment: %s", settings.environment)
    logger.info("API: %s:%d", settings.api.host, settings.api.port)
    logger.info("Providers: %s", ", ".join(registry.provider_names))
    logger.info("Storage backend: %s", settings.storage.backend)
    logger.info("Routing: %s", "enabled" if router_instance else "disabled")
    logger.info("Notifications: %s", "enabled" if notification_service else "disabled")
    logger.info("=" * 70)


async def _shutdown_hook(app: FastAPI) -> None:
    """Graceful teardown (create_app on_shutdown), reading state off ``app.state``."""
    await _shutdown(
        app.state.provider_registry,
        getattr(app.state, "notification_service", None),
        getattr(app.state, "metering_redis", None),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Back-compat lifespan composing the create_app on_startup/on_shutdown hooks.

    The app itself is built via ``create_app`` (which wires its own lifespan
    from the same hooks); this wrapper exists so the asgi-coverage tests that
    drive ``async with lifespan(app)`` keep working against one code path.
    """
    await _startup(app)
    try:
        yield
    finally:
        await _shutdown_hook(app)


# ============================================================================
# Routers (imported here so create_app receives them via routers=)
# ============================================================================

from src.api.health import router as health_router  # noqa: E402 -- imported for create_app
from src.api.tts import router as tts_router  # noqa: E402 -- imported for create_app
from src.api.usage import router as usage_router  # noqa: E402 -- imported for create_app
from src.api.ws import router as ws_router  # noqa: E402 -- imported for create_app

# ============================================================================
# Create App
# ============================================================================

app = create_app(
    settings,
    routers=[health_router, tts_router, usage_router, ws_router],
    title=settings.app_name,
    description="Text-to-Speech microservice with multiple provider support",
    version=settings.app_version,
    on_startup=[_startup],
    on_shutdown=[_shutdown_hook],
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "tts",
            "description": (
                "Text-to-speech generation: sync + async, SSE/raw streaming, "
                "batching/concatenation, job polling, audio download."
            ),
        },
        {
            "name": "tts-ws",
            "description": "WebSocket channel for real-time job/batch notifications.",
        },
        {
            "name": "usage",
            "description": (
                "Per-tenant usage analytics, cost breakdown, dashboard, quotas, "
                "alerts, and provider cost rates."
            ),
        },
    ],
)


@app.get(
    "/",
    operation_id="service_root",
    tags=["health"],
    summary="Service banner + endpoint index",
)
async def root() -> dict[str, Any]:
    """Return service metadata and available endpoint paths."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "TTS microservice with multiple provider support",
        "endpoints": {
            "health": "/health",
            "providers": "/api/tts/providers",
            "generate": "/api/tts/generate",
            "stream": "/api/tts/stream",
            "batch": "/api/tts/generate/batch",
            "usage": "/api/tts/usage",
            "ws": "/api/tts/ws",
            "docs": "/docs",
        },
        "status": "running",
        "environment": settings.environment,
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
