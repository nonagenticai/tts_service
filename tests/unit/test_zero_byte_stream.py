"""Regression tests: a stream that produced no audio must not read as success.

Three independent holes let a failed synthesis surface as a clean HTTP 200 with
zero audio bytes, fully billed:

1. ``KokoroProvider._generate_segments_to_queue`` swallowed every pipeline
   exception (``except Exception: logger.exception``) and its ``finally`` still
   emitted the ``is_final=True`` sentinel — a first-segment espeak/OOM failure
   was byte-for-byte indistinguishable from a successful empty synthesis.
2. ``TTSService.generate_stream`` set ``produced = True`` on *any* chunk,
   including that empty sentinel, so no fallback provider was tried and
   ``_meter_completed_stream`` billed the full text.
3. The cloud providers' SYNC paths raise on an empty body but their STREAM
   paths did not, so an upstream 200-with-no-body yielded only the sentinel.

Plus the monthly budget ceiling, which never fired on the default request
because ``provider="auto"`` is absent from ``cost_per_1k_chars`` and estimated
to $0.00.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas import RoutingStrategy
from src.providers.base import AudioChunk, GenerationResult
from src.providers.kokoro_provider import KokoroProvider
from src.services.router import ProviderRouter, RoutingDecision
from src.services.tts_service import TTSService, _estimate_cost
from tests.conftest import tenant_uuid

# ---------------------------------------------------------------------------
# 1. Kokoro: a pipeline error must propagate, not become a success sentinel
# ---------------------------------------------------------------------------


class _FakeSegment:
    def __init__(self, audio: object) -> None:
        self.audio = audio


@pytest.fixture
def _stub_soundfile():
    """Stub ``soundfile`` (not installed locally) so the thread body can import it."""
    module = types.ModuleType("soundfile")

    def _write(buf, data, samplerate, format=None):  # noqa: A002
        buf.write(b"RIFFfake-wav-bytes")

    setattr(module, "write", _write)  # noqa: B010 - ModuleType has no static attr
    existing = sys.modules.get("soundfile")
    sys.modules["soundfile"] = module
    try:
        yield module
    finally:
        if existing is None:
            del sys.modules["soundfile"]
        else:
            sys.modules["soundfile"] = existing


@pytest.mark.asyncio
async def test_kokoro_first_segment_failure_propagates(_stub_soundfile):
    """An espeak/OOM failure on the FIRST segment must raise, not yield is_final."""

    def _exploding_pipeline(text, voice=..., speed=...):
        raise RuntimeError("espeak-ng phonemization failed")
        yield  # pragma: no cover - generator marker

    provider = KokoroProvider()
    chunks: list[AudioChunk] = []
    with (
        patch.object(KokoroProvider, "_get_pipeline", return_value=_exploding_pipeline),
        pytest.raises(RuntimeError, match="espeak-ng phonemization failed"),
    ):
        async for chunk in provider.generate_stream(
            text="hello", voice="af_heart", output_format="wav"
        ):
            chunks.append(chunk)

    # And it must never have claimed a clean finish.
    assert not any(c.is_final for c in chunks), (
        "a failed synthesis emitted is_final=True — indistinguishable from success"
    )


@pytest.mark.asyncio
async def test_kokoro_mid_stream_failure_propagates_after_real_audio(_stub_soundfile):
    """Bytes already sent are kept, but the stream must still end in an error."""

    def _partial_pipeline(text, voice=..., speed=...):
        yield _FakeSegment(audio=[0.0, 0.1])
        raise RuntimeError("kokoro OOM mid-stream")

    provider = KokoroProvider()
    chunks: list[AudioChunk] = []
    with (
        patch.object(KokoroProvider, "_get_pipeline", return_value=_partial_pipeline),
        pytest.raises(RuntimeError, match="kokoro OOM mid-stream"),
    ):
        async for chunk in provider.generate_stream(
            text="hello", voice="af_heart", output_format="wav"
        ):
            chunks.append(chunk)

    assert [c.data for c in chunks] == [b"RIFFfake-wav-bytes"]
    assert not any(c.is_final for c in chunks)


# ---------------------------------------------------------------------------
# 2. TTSService.generate_stream: the single chokepoint
# ---------------------------------------------------------------------------


def _make_provider(name: str) -> AsyncMock:
    provider = AsyncMock()
    provider.name = name
    provider.provider_type = "local"
    provider.sample_rate = 24000
    provider.is_available = AsyncMock(return_value=True)
    provider.get_default_voice = MagicMock(return_value=f"{name}_default")
    provider.list_voices = AsyncMock(return_value=[])
    return provider


def _sentinel_only_stream(**_kwargs) -> AsyncIterator[AudioChunk]:
    """A provider that emits ONLY the empty is_final sentinel — zero audio."""

    async def _gen() -> AsyncIterator[AudioChunk]:
        yield AudioChunk(
            data=b"",
            chunk_index=0,
            is_final=True,
            format="mp3",
            sample_rate=24000,
            total_bytes=0,
        )

    return _gen()


def _real_audio_stream(**_kwargs) -> AsyncIterator[AudioChunk]:
    async def _gen() -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"realaudio", chunk_index=0, format="mp3", sample_rate=24000)
        yield AudioChunk(
            data=b"",
            chunk_index=1,
            is_final=True,
            format="mp3",
            sample_rate=24000,
            total_bytes=9,
        )

    return _gen()


def _auto_router(primary: str, fallback_chain: list[str]) -> AsyncMock:
    """A ProviderRouter stand-in — ``spec=`` so the isinstance gate in
    ``_resolve_provider_chain`` accepts it."""
    router = AsyncMock(spec=ProviderRouter)
    router.select_provider = AsyncMock(
        return_value=RoutingDecision(
            provider_name=primary,
            voice="custom",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=list(fallback_chain),
        )
    )
    return router


def _service(providers: dict[str, AsyncMock], router: object | None = None) -> TTSService:
    registry = MagicMock()
    registry.get_provider.side_effect = lambda name: providers[name]
    return TTSService(registry=registry, storage=AsyncMock(), router=router)


@pytest.mark.asyncio
async def test_zero_byte_stream_is_not_billed_and_not_reported_as_produced():
    """A provider that yields only the empty sentinel must not bill the tenant."""
    kokoro = _make_provider("kokoro")
    kokoro.generate_stream = _sentinel_only_stream
    service = _service({"kokoro": kokoro})

    with patch.object(TTSService, "_meter_completed_stream", new_callable=AsyncMock) as mock_meter:
        with pytest.raises(RuntimeError):
            async for _chunk in service.generate_stream(
                text="a long paid sentence",
                provider="kokoro",
                db=AsyncMock(),
                tenant_id=tenant_uuid("t1"),
            ):
                pass

    mock_meter.assert_not_called()


@pytest.mark.asyncio
async def test_zero_byte_stream_falls_back_to_the_next_provider():
    """Nothing was sent on the wire, so the fallback chain must still be usable."""
    kokoro = _make_provider("kokoro")
    kokoro.generate_stream = _sentinel_only_stream
    openai = _make_provider("openai")
    openai.generate_stream = _real_audio_stream

    service = _service(
        {"kokoro": kokoro, "openai": openai},
        router=_auto_router("kokoro", ["openai"]),
    )

    with patch.object(TTSService, "_meter_completed_stream", new_callable=AsyncMock) as mock_meter:
        received = [
            c.data
            async for c in service.generate_stream(
                text="hello", provider="auto", db=AsyncMock(), tenant_id=tenant_uuid("t1")
            )
        ]

    assert b"realaudio" in received
    assert mock_meter.await_count == 1
    meter_call = mock_meter.await_args
    assert meter_call is not None
    assert meter_call.kwargs["provider"] == "openai"


@pytest.mark.asyncio
async def test_stream_raw_route_does_not_serve_an_empty_200():
    """PRODUCTION CALL SITE: /api/tts/stream/raw must not finish a zero-byte body."""
    from src.api.schemas import TTSStreamRequest
    from src.api.tts import stream_generate_raw

    kokoro = _make_provider("kokoro")
    kokoro.generate_stream = _sentinel_only_stream
    service = _service({"kokoro": kokoro})

    request = MagicMock()
    request.state.tenant_id = tenant_uuid("t1")
    request.app.state.tts_service = service
    registry = MagicMock()
    registry.get_provider.return_value = kokoro
    request.app.state.provider_registry = registry

    body = TTSStreamRequest(text="a long paid sentence", provider="kokoro")

    with patch("src.api.tts.check_quota", new_callable=AsyncMock):
        response = await stream_generate_raw(body, request, AsyncMock())

    emitted = b""
    with pytest.raises(RuntimeError):
        async for part in response.body_iterator:
            emitted += part.encode() if isinstance(part, str) else bytes(part)

    assert emitted == b"", "no audio should have been served"


# ---------------------------------------------------------------------------
# 3. Cloud providers: the STREAM path had no empty-body guard
# ---------------------------------------------------------------------------


def _empty_body_client() -> MagicMock:
    """An httpx-like client whose streamed response carries a 200 and no bytes."""
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.status_code = 200

    async def _aiter_bytes(chunk_size=None):
        return
        yield b""  # pragma: no cover - generator marker

    response.aiter_bytes = _aiter_bytes

    @asynccontextmanager
    async def _stream(*_args, **_kwargs):
        yield response

    client = MagicMock()
    client.stream = _stream
    return client


@pytest.mark.asyncio
async def test_openai_stream_rejects_an_empty_body():
    from src.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-test")
    provider._client = _empty_body_client()

    chunks: list[AudioChunk] = []
    with pytest.raises((RuntimeError, ValueError)):
        async for chunk in provider.generate_stream(text="hi", voice="alloy"):
            chunks.append(chunk)

    assert not any(c.is_final for c in chunks)


@pytest.mark.asyncio
async def test_elevenlabs_stream_rejects_an_empty_body():
    from src.providers.elevenlabs_provider import ElevenLabsProvider

    provider = ElevenLabsProvider(api_key="el-test")
    provider._client = _empty_body_client()

    chunks: list[AudioChunk] = []
    with pytest.raises((RuntimeError, ValueError)):
        async for chunk in provider.generate_stream(text="hi", voice="rachel"):
            chunks.append(chunk)

    assert not any(c.is_final for c in chunks)


# ---------------------------------------------------------------------------
# 4. The monthly budget ceiling must fire on the DEFAULT request
# ---------------------------------------------------------------------------


def test_estimate_cost_for_auto_is_not_free():
    """``auto`` is absent from cost_per_1k_chars; estimating $0 disables the cap."""
    cheapest_priced = 0.015  # openai
    dearest = 0.030  # elevenlabs
    estimate = _estimate_cost("auto", 100_000)
    assert estimate >= (100_000 / 1000.0) * dearest, (
        f"auto estimated {estimate}; a pre-flight ceiling must not under-estimate "
        f"(cheapest priced provider alone is {(100_000 / 1000.0) * cheapest_priced})"
    )


@pytest.mark.asyncio
async def test_generate_sync_prices_the_routed_provider_not_the_raw_request():
    """PRODUCTION CALL SITE: generate_sync's pre-flight ceiling with provider='auto'.

    ``auto`` routed to elevenlabs must project elevenlabs' rate, not $0.00.
    """
    elevenlabs = _make_provider("elevenlabs")
    elevenlabs.generate = AsyncMock(
        return_value=GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
    )
    service = _service({"elevenlabs": elevenlabs}, router=_auto_router("elevenlabs", []))
    db = AsyncMock()
    text = "x" * 100_000

    with patch("src.services.quota_service.check_quota", new_callable=AsyncMock) as mock_quota:
        await service.generate_sync(
            text=text, provider="auto", db=db, tenant_id=tenant_uuid("t1"), no_cache=True
        )

    mock_quota.assert_called_once()
    # elevenlabs = $0.030 / 1k chars * 100k chars = $3.00 (was $0.00)
    assert mock_quota.call_args.kwargs["projected_cost"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_enforce_stream_quota_projects_a_real_cost_for_auto():
    """PRODUCTION CALL SITE: the streaming routes' pre-flight ceiling."""
    from src.api.tts import _enforce_stream_quota

    with patch("src.api.tts.check_quota", new_callable=AsyncMock) as mock_quota:
        await _enforce_stream_quota(AsyncMock(), tenant_uuid("t1"), "x" * 100_000, "auto")

    assert mock_quota.call_args.kwargs["projected_cost"] > 0.0
