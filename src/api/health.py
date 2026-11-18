"""Health check endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.cache.redis_client import get_redis_client
from src.config import get_settings
from src.database.connection import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# The library exception trees below are NOT optional and must never be
# "simplified" away — see the long note in ``src/cache/redis_client.py`` and
# ``tests/unit/test_redis_error_hierarchy.py``. None of these inherit from the
# builtins whose names they shadow::
#
#     sqlalchemy.exc.OperationalError.__mro__
#         -> DatabaseError -> DBAPIError -> StatementError -> SQLAlchemyError
#            -> HasDescriptionCode -> Exception
#     issubclass(sqlalchemy.exc.OperationalError, OSError | RuntimeError) -> False
#     issubclass(redis.exceptions.RedisError,     OSError)                -> False
#     pydantic.ValidationError.__mro__ -> ValueError -> Exception
#
# So a builtins-only ``except (OSError, ConnectionError, TimeoutError,
# RuntimeError)`` let every REAL database fault escape these probes, and
# FastAPI turned it into a 500. On /health/ready (and its /readyz alias) that
# breaks the Kubernetes readiness contract: a transient DB fault must be a 503
# ("not ready"), never a 500 ("the app is broken").
#
# Probe failure = the dependency is down, not the process. The builtins stay in
# the tuples because a socket-level error can still surface as one.
_DB_PROBE_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, SQLAlchemyError)
#
# HONEST SCOPE NOTE for ``_REDIS_PROBE_ERRORS``: on the ``/health`` redis probe
# it does NOT guard ``ping()`` — ``w2v_common.cache.redis_client.RedisClient.
# ping()`` swallows ``RuntimeError`` and ``RedisError`` and returns ``False``,
# so no exception ever reaches this handler from that call. A dead Redis is
# detected by CHECKING THE RETURN VALUE, not by catching. What the tuple does
# genuinely guard in that block is ``get_redis_client()``, which constructs the
# client and raises ``RuntimeError`` when ``REDIS_URL`` is unset (and
# ``ValueError`` on a non-integer ``REDIS_*`` env, caught by the outer handler
# via ``_SETTINGS_ERRORS``). ``RedisError`` is kept because ``ping()`` is not a
# contractually non-raising API — a client substituted here (tests do exactly
# this) or a future override in ``src/cache/redis_client.py`` can raise it, and
# a 500 from a health endpoint is the failure this module exists to prevent.
_REDIS_PROBE_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, RedisError)
_SETTINGS_ERRORS = (ValidationError,)


def _check_storage() -> str:
    """Verify the configured local storage path exists and is writable."""
    try:
        storage_path = Path(get_settings().storage.local_path)
        if not storage_path.exists():
            return "path not found"
        if not os.access(str(storage_path), os.W_OK):
            return "not writable"
        return "healthy"
    except (OSError, *_SETTINGS_ERRORS) as e:
        # ``get_settings()`` constructs ``Settings()`` on a cache miss, which
        # raises pydantic ``ValidationError`` (a ``ValueError``) on bad env —
        # not an ``OSError``. ``PermissionError`` was redundant here: it is
        # already an ``OSError`` subclass.
        return f"unhealthy: {e}"


@router.get("/health", response_model=None, operation_id="health_check")
async def health_check() -> dict[str, Any] | JSONResponse:
    """Combined health check for all components."""
    try:
        db_status = "healthy"
        try:
            engine = await get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except _DB_PROBE_ERRORS as e:
            db_status = f"unhealthy: {e}"

        redis_status = "healthy"
        try:
            redis_client = get_redis_client()
            if redis_client:
                # ``RedisClient.ping()`` (w2v_common) is FAIL-SOFT: it catches
                # RuntimeError from ``_ensure_connected()`` and RedisError from
                # the ping itself and returns ``False``. It never raises on a
                # dead backend — so the exception handler below can NOT detect
                # one, and discarding this bool reported a completely dead
                # Redis as "healthy". The return value is the only signal.
                if not await redis_client.ping():
                    redis_status = "unhealthy: ping returned False"
            else:
                redis_status = "not configured"
        except _REDIS_PROBE_ERRORS as e:
            redis_status = f"unhealthy: {e}"

        storage_status = _check_storage()

        overall = (
            "healthy"
            if all(s == "healthy" for s in (db_status, redis_status, storage_status))
            else "degraded"
        )

        return {
            "status": overall,
            "components": {
                "database": db_status,
                "redis": redis_status,
                "storage": storage_status,
                "api": "healthy",
            },
        }
    except (*_DB_PROBE_ERRORS, *_REDIS_PROBE_ERRORS, *_SETTINGS_ERRORS) as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@router.get("/health/live", response_model=None, operation_id="health_live")
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=None, operation_id="health_ready")
async def health_ready() -> dict[str, Any] | JSONResponse:
    """Kubernetes readiness probe — checks database connectivity."""
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "ready": True, "checks": {"database": {"status": "up"}}}
    except _DB_PROBE_ERRORS as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ready": False,
                "checks": {"database": {"status": "down", "error": str(e)}},
            },
        )


@router.get("/healthz", response_model=None, operation_id="health.live")
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return await health_live()


@router.get("/readyz", response_model=None, operation_id="health.ready")
async def readyz() -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready()
