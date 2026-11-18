"""Shared test fixtures for TTS service."""

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.config import clear_settings_cache

# Set test environment variables before importing any app code.
# Port 5475, not 5432: a 5432 default resolves to whatever postgres the dev
# happens to be running, which answers the TCP connect and THEN fails auth —
# disguising a config error as a connection error.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5475/tts_service_ci",  # pragma: allowlist secret
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6380/0")
os.environ.setdefault("API_PORT", "6194")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_PATH", "/tmp/tts_test_audio")
# w2v_common >= 0.15.0 makes StorageSettings.minio_endpoint required (no default).
# Seed a placeholder even when STORAGE_BACKEND=local so Settings() instantiation
# in tests doesn't ValidationError on the field being absent.
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("ENVIRONMENT", "test")

# Capture the test DB URL BEFORE the autouse _clear_settings fixture
# monkeypatches DATABASE_URL to its in-test default. CI sets DATABASE_URL
# to its tts_service_ci postgres; locally falls back to the default above
# (`docker compose up -d tts_service_test_postgres` — :5475).
_TEST_DATABASE_URL = os.environ["DATABASE_URL"]

# Ensure the configured local storage path exists so health checks see a writable dir.
Path(os.environ["STORAGE_LOCAL_PATH"]).mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch):
    """Remove CI env vars that interfere with config-default tests."""
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "DATABASE_",
                "AUTH_",
                "REDIS_",
                "API_",
                "STORAGE_",
                "LOG_",
                "CACHE_",
                "TRACING_",
                "LLM_",
                "FFMPEG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    # Restore test-only env vars that were cleared above
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5475/tts_service_ci",  # pragma: allowlist secret
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6380/0")
    monkeypatch.setenv("API_PORT", "6194")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_LOCAL_PATH", "/tmp/tts_test_audio")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")
    Path("/tmp/tts_test_audio").mkdir(parents=True, exist_ok=True)
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from src.config import Settings

    settings = Settings()
    yield settings


@pytest.fixture
def mock_provider_registry():
    """Create a mock provider registry."""
    from src.providers.registry import ProviderRegistry

    return ProviderRegistry()


# ---------------------------------------------------------------------------
# Real database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine():
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, localhost:5434 locally). Runs Base.metadata.create_all (idempotent)
    then TRUNCATEs every model table so each test starts from a clean slate.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Real async_sessionmaker for tts_service tests."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory):
    """Real AsyncSession for a single test."""
    async with db_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Real Redis fixtures (fakeredis)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_redis_client():
    """RedisClient backed by fakeredis (real commands, in-memory)."""
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client


# ---------------------------------------------------------------------------
# Tenant identity constants
# ---------------------------------------------------------------------------
# w2v_common >= 0.67.0 parses every tenant id as a ``uuid.UUID`` and RAISES on
# anything else, so a slug fixture like ``"tenant-1"`` now dies at auth and
# turns an unrelated assertion into a 401. These are real (distinct) uuids so
# the request authenticates and the original assertion is exercised again.
TENANT_A = "11111111-1111-4111-8111-111111111111"
TENANT_B = "22222222-2222-4222-8222-222222222222"

#: Namespace for :func:`tenant_uuid`. Any fixed uuid works; this is the RFC 4122
#: example namespace, chosen so it is obviously not a real tenant.
_TENANT_SLUG_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def tenant_uuid(slug: str) -> str:
    """A stable, distinct uuid standing in for a legacy test-tenant slug.

    ``tenant_id`` is a postgres ``uuid`` column (revision ``005_tenant_id_uuid``),
    so the slug fixtures this suite grew up on (``"tenant-1"``, ``"victim-tenant"``,
    ``"tenant-test"``) can no longer be persisted at all — psycopg refuses them
    before the row is written.

    Substituting one shared uuid everywhere would have been the wrong repair:
    a third of this suite's tenant assertions are ISOLATION tests
    (``test_tenant_isolation``, ``test_404_when_alert_belongs_to_other_tenant``,
    the whole of ``test_cross_tenant_job_access``) whose entire premise is that
    two tenants are DIFFERENT. Collapsing them would leave those tests green
    while proving nothing. ``uuid5`` keeps every slug distinct and deterministic,
    so the slug still names the role it plays in the test.

    Deliberately NOT used by ``tests/unit/test_tenant_uuid_census.py``: that
    module's tenant literals are dirty on purpose — they are the census's input.
    """
    return str(uuid.uuid5(_TENANT_SLUG_NS, slug))


# ---------------------------------------------------------------------------
# Trusted-network configuration
# ---------------------------------------------------------------------------
# w2v-common 0.89.0 made TRUSTED_CIDRS default to EMPTY: no peer is trusted
# unless the operator names the networks. Tests that exercise the trusted-
# network lane must configure it explicitly — they may not lean on a default.
# These are the lanes the tests were written against (the pre-0.89 default);
# 192.168.0.0/16 stays out (SSR-proxy bypass vector, 2026-04-18).
EXPLICIT_TRUSTED_CIDRS = "127.0.0.1/32,::1/128,172.16.0.0/12,10.0.0.0/8"


@pytest.fixture
def explicit_trusted_cidrs(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure TRUSTED_CIDRS explicitly for the duration of one test.

    ``TRUSTED_NETWORKS`` is parsed from the env once, at import, and
    ``_is_trusted_network`` / the dispatch lanes read that module global at
    call time — so set the env (for anything re-reading it) AND pin the parsed
    global, instead of reloading the module mid-suite.
    """
    import ipaddress

    import w2v_common.auth.middleware as auth_middleware

    monkeypatch.setenv("TRUSTED_CIDRS", EXPLICIT_TRUSTED_CIDRS)
    monkeypatch.setattr(
        auth_middleware,
        "TRUSTED_NETWORKS",
        [ipaddress.ip_network(c, strict=False) for c in EXPLICIT_TRUSTED_CIDRS.split(",")],
    )
    return EXPLICIT_TRUSTED_CIDRS
