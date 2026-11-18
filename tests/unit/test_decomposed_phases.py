"""Tests for the helper phase functions extracted during the C901 refactor.

Covers extracted methods from:
- src.asgi.lifespan        — _build_router, _init_connection_manager, _prewarm_kokoro
- src.api.tts.generate     — _validate_provider_or_404, _resolve_quality_profile,
                             _build_audio_response
- src.services.tts_service.generate_sync — _try_cache_hit (cache-miss path)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    PostProcessingConfig,
    QualityProfile,
    TTSGenerateRequest,
)
from src.api.tts import (
    _build_audio_response,
    _resolve_quality_profile,
    _validate_provider_or_404,
)
from src.asgi import _build_router, _init_connection_manager, _prewarm_kokoro
from src.providers.base import GenerationResult
from tests.conftest import tenant_uuid

# ---------------------------------------------------------------------------
# api.tts helpers
# ---------------------------------------------------------------------------


def test_validate_provider_or_404_skips_auto() -> None:
    registry = MagicMock()
    registry.get_provider.side_effect = AssertionError("must not be called for auto")
    _validate_provider_or_404(registry, "auto")  # no exception
    registry.get_provider.assert_not_called()


def test_validate_provider_or_404_raises_404_for_unknown() -> None:
    registry = MagicMock()
    registry.get_provider.side_effect = KeyError("kokoro")
    with pytest.raises(HTTPException) as ei:
        _validate_provider_or_404(registry, "kokoro")
    assert ei.value.status_code == 404


def test_validate_provider_or_404_passes_for_known() -> None:
    registry = MagicMock()
    registry.get_provider.return_value = MagicMock()
    _validate_provider_or_404(registry, "openai")
    registry.get_provider.assert_called_once_with("openai")


def test_resolve_quality_profile_passthrough_when_no_profile() -> None:
    body = TTSGenerateRequest(text="hi", provider="kokoro")
    pp, params = _resolve_quality_profile(body)
    assert pp is None
    assert params is None


def test_resolve_quality_profile_narration_elevenlabs_sets_defaults() -> None:
    body = TTSGenerateRequest(
        text="hi",
        provider="elevenlabs",
        quality_profile=QualityProfile.NARRATION,
    )
    pp, params = _resolve_quality_profile(body)
    assert isinstance(pp, PostProcessingConfig)
    assert params is not None
    assert params["stability"] == 0.7
    assert params["similarity_boost"] == 0.85


def test_resolve_quality_profile_does_not_override_explicit_post_processing() -> None:
    explicit = PostProcessingConfig()
    body = TTSGenerateRequest(
        text="hi",
        provider="elevenlabs",
        quality_profile=QualityProfile.NARRATION,
        post_processing=explicit,
    )
    pp, _ = _resolve_quality_profile(body)
    assert pp is explicit  # untouched when caller supplied one


def test_build_audio_response_sets_cache_hit_and_provider_headers() -> None:
    result = GenerationResult(
        audio_data=b"\x00\x01\x02\x03",
        format="mp3",
        sample_rate=24000,
        duration_seconds=1.5,
    )
    meta = {
        "cached": True,
        "resolved_provider": "kokoro",
        "detected_language": "en",
        "estimated_cost": 0.0,
    }
    resp = _build_audio_response(result, meta, fallback_provider="kokoro")
    assert resp.media_type == "audio/mpeg"
    assert resp.headers["x-cache"] == "HIT"
    assert resp.headers["x-provider"] == "kokoro"
    assert resp.headers["x-detected-language"] == "en"
    assert resp.headers["x-estimated-cost-usd"] == "0.0"
    assert resp.headers["x-audio-size"] == "4"


def test_build_audio_response_wav_miss() -> None:
    result = GenerationResult(
        audio_data=b"RIFF",
        format="wav",
        sample_rate=22050,
        duration_seconds=0.1,
    )
    resp = _build_audio_response(result, {"cached": False}, fallback_provider="kitten")
    assert resp.media_type == "audio/wav"
    assert resp.headers["x-cache"] == "MISS"
    assert "x-provider" not in {k.lower() for k in resp.headers}


# ---------------------------------------------------------------------------
# asgi helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _async_cm(value: object):
    """Minimal async context manager yielding ``value``."""
    yield value


def _patched_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    """Mutate the module-level `settings` used by asgi helpers."""
    import src.asgi as asgi_mod

    snap = asgi_mod.settings
    for k, v in overrides.items():
        # walk nested attributes via dotted key
        target: object = snap
        parts = k.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        monkeypatch.setattr(target, parts[-1], v)


def test_build_router_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_settings(monkeypatch, **{"routing.enabled": False})
    assert _build_router(registry=MagicMock()) is None


def test_build_router_returns_instance_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_settings(
        monkeypatch,
        **{
            "routing.enabled": True,
            "routing.cost_per_1k_chars": {"kokoro": 0.0},
            "routing.quality_scores": {"kokoro": 0.9},
            "routing.default_strategy": "balanced",
            "routing.max_fallback_attempts": 1,
        },
    )
    registry = MagicMock()
    router = _build_router(registry=registry)
    assert router is not None
    # ProviderRouter constructed successfully with our registry
    from src.services.router import ProviderRouter

    assert isinstance(router, ProviderRouter)


def test_init_connection_manager_attaches_to_app_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_settings(monkeypatch, **{"websocket.max_connections_per_tenant": 7})
    app = FastAPI()
    _init_connection_manager(app)
    assert hasattr(app.state, "connection_manager")
    assert app.state.connection_manager._max_per_tenant == 7


@pytest.mark.asyncio
async def test_prewarm_kokoro_no_op_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_settings(monkeypatch, **{"kokoro.enabled": False})
    registry = SimpleNamespace(provider_names=set(), get_provider=lambda n: None)
    # Should not raise / should not call get_provider
    await _prewarm_kokoro(registry)


# ---------------------------------------------------------------------------
# tts_service._try_cache_hit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_cache_hit_no_cache_returns_empty_key() -> None:
    from src.services.tts_service import TTSService

    svc = TTSService.__new__(TTSService)
    # We don't need internal state for the no_cache=True branch; the cache
    # checker should not be invoked at all when no_cache=True.
    svc.set_cache_checker(AsyncMock(side_effect=AssertionError("must not be called")))

    provider = MagicMock()
    provider.sample_rate = 24000
    meta: dict = {"cached": False, "cache_hit": False}

    cached, key = await svc._try_cache_hit(
        no_cache=True,
        tenant_id=tenant_uuid("t"),
        processed_text="hi",
        resolved_provider="kokoro",
        resolved_voice="af_heart",
        speed=1.0,
        output_format="mp3",
        provider_obj=provider,
        meta=meta,
        start_time=0.0,
    )
    assert cached is None
    assert key == ""
    assert meta["cached"] is False


@pytest.mark.asyncio
async def test_try_cache_hit_returns_result_on_hit() -> None:
    from src.services.tts_service import TTSService

    svc = TTSService.__new__(TTSService)
    svc.set_cache_checker(AsyncMock(return_value=(b"audio-bytes", "cache-key", 2.5)))

    provider = MagicMock()
    provider.sample_rate = 24000
    meta: dict = {"cached": False, "cache_hit": False}

    cached, key = await svc._try_cache_hit(
        no_cache=False,
        tenant_id=tenant_uuid("t"),
        processed_text="hi",
        resolved_provider="kokoro",
        resolved_voice="af_heart",
        speed=1.0,
        output_format="mp3",
        provider_obj=provider,
        meta=meta,
        start_time=0.0,
    )
    assert cached is not None
    assert cached.audio_data == b"audio-bytes"
    assert cached.duration_seconds == 2.5
    assert key == ""
    assert meta["cached"] is True
    assert meta["cache_hit"] is True
    assert meta["estimated_cost"] == 0.0
    assert "latency_ms" in meta


@pytest.mark.asyncio
async def test_try_cache_hit_returns_key_on_miss() -> None:
    from src.services.tts_service import TTSService

    svc = TTSService.__new__(TTSService)
    svc.set_cache_checker(AsyncMock(return_value=(None, "fresh-key", 0.0)))

    provider = MagicMock()
    provider.sample_rate = 24000
    meta: dict = {"cached": False, "cache_hit": False}

    cached, key = await svc._try_cache_hit(
        no_cache=False,
        tenant_id=tenant_uuid("t"),
        processed_text="hi",
        resolved_provider="kokoro",
        resolved_voice="af_heart",
        speed=1.0,
        output_format="mp3",
        provider_obj=provider,
        meta=meta,
        start_time=0.0,
    )
    assert cached is None
    assert key == "fresh-key"
    assert meta["cached"] is False


# ---------------------------------------------------------------------------
# Startup log redaction -- CALL SITE, not the helper
# ---------------------------------------------------------------------------

_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_USER = "tts_app"
_FAKE_HOST = "pgbouncer.pgbouncer"
_FAKE_DB = "tts_service"
_FAKE_DSN = f"postgresql+psycopg://{_FAKE_USER}:{_FAKE_PASSWORD}@{_FAKE_HOST}:6432/{_FAKE_DB}"


@pytest.mark.asyncio
async def test_init_database_log_line_does_not_leak_dsn_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup 'Database connected' line must never carry DSN userinfo.

    This is deliberately a test of the CALL SITE in ``src.asgi._init_database``
    rather than of ``w2v_common.observability.database_endpoint``: a
    helper-only test stays green for the entire duration of the leak.

    ``caplog`` is not used on purpose -- ``configure_logging`` re-points root
    handlers at import time, so caplog can capture nothing and the assertions
    would pass vacuously. We attach our own handler to the module logger.
    """
    import io
    import logging

    import src.asgi as asgi_mod

    _patched_settings(monkeypatch, **{"database.url": _FAKE_DSN})

    engine = MagicMock()
    conn = AsyncMock()
    engine.begin = MagicMock(return_value=_async_cm(conn))

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    asgi_mod.logger.addHandler(handler)
    # setLevel(), never `.level = ...`: Logger memoizes isEnabledFor in
    # ``._cache`` and only setLevel() invalidates it.
    previous_level = asgi_mod.logger.level
    asgi_mod.logger.setLevel(logging.INFO)
    try:
        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=engine),
            patch("src.asgi._reset_orphaned_jobs", new_callable=AsyncMock),
        ):
            await asgi_mod._init_database(MagicMock())
    finally:
        asgi_mod.logger.removeHandler(handler)
        asgi_mod.logger.setLevel(previous_level)

    captured = stream.getvalue()

    # Vacuity guard FIRST: without it the test passes when the line never emits.
    # The colon is load-bearing -- the same function also logs
    # "Database tables created/verified", and a looser match would be satisfied
    # by the wrong line.
    assert "Database connected: " in captured, f"startup line never emitted: {captured!r}"

    # The secret and the DB username must be gone.
    assert _FAKE_PASSWORD not in captured
    assert _FAKE_USER not in captured

    # ...but the line must stay useful: host and database still identify the target.
    assert _FAKE_HOST in captured
    assert _FAKE_DB in captured


# ---------------------------------------------------------------------------
# THE SIBLING LEAK: the Redis URL carries a password in the same slot the DSN
# does (``redis://[<user>]:<password>@host:port/db``), and ``_init_redis`` sits
# a few lines below the ``_init_database`` call site guarded above -- guarded
# beside unguarded, in the same module.
# ---------------------------------------------------------------------------

_FAKE_REDIS_USER = "default"
_FAKE_REDIS_HOST = "redis.w2v"
_FAKE_REDIS_URL = f"redis://{_FAKE_REDIS_USER}:{_FAKE_PASSWORD}@{_FAKE_REDIS_HOST}:6379/0"


@pytest.mark.asyncio
async def test_init_redis_log_line_does_not_leak_redis_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup 'Redis connected' line must never carry Redis userinfo.

    Like its DSN sibling above, this is deliberately a test of the CALL SITE in
    ``src.asgi._init_redis`` rather than of
    ``w2v_common.observability.redis_endpoint``: a helper-only test stays green
    for the entire duration of the leak, and the helper already has its own
    unit tests upstream in w2v_common.

    ``caplog`` is not used on purpose -- ``configure_logging`` re-points root
    handlers at import time, so caplog can capture nothing and the assertions
    would pass vacuously. We attach our own handler to the module logger.
    """
    import io
    import logging

    import src.asgi as asgi_mod

    _patched_settings(monkeypatch, **{"redis.url": _FAKE_REDIS_URL})

    redis_client = AsyncMock()
    redis_client.ping = AsyncMock(return_value=True)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    asgi_mod.logger.addHandler(handler)
    # setLevel(), never `.level = ...` and never
    # ``monkeypatch.setattr(logger, "level", ...)``: Logger memoizes
    # isEnabledFor in ``._cache`` and ONLY setLevel() invalidates it. A stale
    # "INFO disabled" entry means the record never reaches this handler, and
    # then every negative assertion below passes on EMPTY output with the leak
    # fully intact. The vacuity guard is what makes that impossible.
    previous_level = asgi_mod.logger.level
    asgi_mod.logger.setLevel(logging.INFO)
    try:
        with (
            patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=redis_client),
            patch("src.asgi.get_redis_client", return_value=redis_client),
        ):
            await asgi_mod._init_redis(MagicMock())
    finally:
        asgi_mod.logger.removeHandler(handler)
        asgi_mod.logger.setLevel(previous_level)

    captured = stream.getvalue()

    # Vacuity guard FIRST. ``_init_redis`` is fail-open -- it swallows the
    # exception and logs a warning instead -- so without this the test passes
    # when the success line never emits at all.
    assert "Redis connected: " in captured, f"startup line never emitted: {captured!r}"

    # The secret and the Redis username must be gone.
    assert _FAKE_PASSWORD not in captured
    assert f"{_FAKE_REDIS_USER}:" not in captured

    # ...but the line must stay useful: host and port still identify the target.
    assert f"{_FAKE_REDIS_HOST}:6379" in captured
