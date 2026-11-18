"""G5: every tts cost event carries app, funding and a deterministic event_key.

Real emits through w2v_common's cost bus into fakeredis -- the XADD fields are
what billing reads, so they are what is asserted.
"""

from __future__ import annotations

import asyncio
import uuid

import fakeredis.aioredis
import pytest
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.metering import MissingAttributionError, resolve_stream

from src.services.metering import (
    UnknownFundingProviderError,
    funding_for_provider,
    synthesis_event_key,
)
from src.services.tts_service import TTSService
from tests.conftest import TENANT_A

_LICENCE = "0199a000-0000-7000-8000-00000000000a"
_SUBJECT = "0199a000-0000-7000-8000-00000000000b"


@pytest.fixture
async def redis_conn():
    conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield conn
    await conn.aclose()


@pytest.fixture(autouse=True)
def _identity():
    set_identity_context(uuid.UUID(TENANT_A), _LICENCE, _SUBJECT)
    yield
    clear_identity_context()


def _svc(conn) -> TTSService:
    svc = TTSService.__new__(TTSService)
    svc._metering_redis = conn
    return svc


def _emit(svc: TTSService, *, provider: str = "kokoro", call_kind="job", call_id="job-1") -> None:
    svc._emit_cost_event(
        tenant_id=TENANT_A,
        provider=provider,
        model=provider,
        text="hello",
        duration_s=1.0,
        audio_format="mp3",
        latency_ms=10,
        call_kind=call_kind,
        call_id=call_id,
    )


async def _events(conn) -> list[dict[str, str]]:
    for _ in range(50):
        await asyncio.sleep(0)
    return [fields for _id, fields in await conn.xrange(resolve_stream())]


@pytest.mark.parametrize(
    ("provider", "funding"),
    [
        ("kokoro", "local_gpu"),
        ("kitten", "local_gpu"),
        ("openai", "platform_external"),
        ("elevenlabs", "platform_external"),
    ],
)
async def test_event_carries_full_attribution(redis_conn, provider: str, funding: str) -> None:
    _emit(_svc(redis_conn), provider=provider)
    (fields,) = await _events(redis_conn)
    assert fields["app_id"] == "w2v"
    assert fields["funding"] == funding
    assert fields["license_id"] == _LICENCE
    assert fields["subject_id"] == _SUBJECT
    assert fields["event_key"] == synthesis_event_key("job", "job-1")


async def test_event_key_is_deterministic_per_call(redis_conn) -> None:
    svc = _svc(redis_conn)
    _emit(svc, call_id="job-1")
    _emit(svc, call_id="job-1")
    _emit(svc, call_id="job-2")
    _emit(svc, call_kind="stream", call_id="job-1")
    keys = [f["event_key"] for f in await _events(redis_conn)]
    assert keys[0] == keys[1]
    assert len({keys[0], keys[2], keys[3]}) == 3


def test_unknown_provider_has_no_guessed_funding() -> None:
    with pytest.raises(UnknownFundingProviderError):
        funding_for_provider("some-new-vendor")


async def test_strict_mode_full_attribution_emits(redis_conn, monkeypatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    _emit(_svc(redis_conn))
    assert len(await _events(redis_conn)) == 1


async def test_strict_mode_refuses_a_missing_licence(redis_conn, monkeypatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(uuid.UUID(TENANT_A), None, _SUBJECT)
    with pytest.raises(MissingAttributionError, match="license_id"):
        _emit(_svc(redis_conn))
