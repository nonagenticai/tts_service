"""Regression tests for multi-provider fallback in the production generate paths.

The ``fallback_chain`` computed by :class:`ProviderRouter` used to be discarded:
every generate path resolved a single provider and called it once, so a primary
failure hard-failed instead of degrading to the next provider. These tests pin
the wired behaviour for the sync, streaming, and async paths, and prove the
fallback ``except`` catches the *real* provider error type re-raised by
``utils.retry`` (``_RetryableHTTPError``, an ``httpx.HTTPStatusError``).
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.schemas import RoutingStrategy
from src.providers.base import AudioChunk, GenerationResult
from src.providers.openai_provider import _RetryableHTTPError
from src.services.router import ProviderRouter, RoutingDecision
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid


def _retryable_error(message: str = "upstream 503") -> _RetryableHTTPError:
    """Build the real error type providers re-raise after retries are exhausted."""
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")
    response = httpx.Response(503, request=request)
    return _RetryableHTTPError(message, request=request, response=response)


def _make_provider(name: str, *, available: bool = True) -> AsyncMock:
    provider = AsyncMock()
    provider.name = name
    provider.provider_type = "cloud"
    provider.sample_rate = 24000
    provider.is_available = AsyncMock(return_value=available)
    provider.get_default_voice = MagicMock(return_value=f"{name}_default")
    provider.list_voices = AsyncMock(return_value=[])
    return provider


def _auto_router(primary: str, fallback_chain: list[str]) -> AsyncMock:
    """A ProviderRouter stand-in whose selection yields a primary + fallback chain."""
    router = AsyncMock(spec=ProviderRouter)
    decision = RoutingDecision(
        provider_name=primary,
        voice="custom",
        strategy_used=RoutingStrategy.BALANCED,
        reason="test",
        fallback_chain=list(fallback_chain),
    )
    router.select_provider = AsyncMock(return_value=decision)
    return router


class TestGenerateSyncFallback:
    def _make_service(self, providers: dict[str, AsyncMock], router: AsyncMock) -> TTSService:
        registry = MagicMock()
        registry.get_provider.side_effect = lambda name: providers[name]
        return TTSService(registry=registry, storage=AsyncMock(), router=router)

    @pytest.mark.asyncio
    async def test_primary_error_degrades_to_secondary(self) -> None:
        primary = _make_provider("openai")
        primary.generate = AsyncMock(side_effect=_retryable_error())
        secondary = _make_provider("kokoro")
        secondary.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"fallback-audio",
                format="mp3",
                sample_rate=24000,
                duration_seconds=1.0,
            )
        )
        service = self._make_service(
            {"openai": primary, "kokoro": secondary},
            _auto_router("openai", ["kokoro"]),
        )

        result, meta = await service.generate_sync(
            text="Hello world",
            provider="auto",
            voice="custom",
            no_cache=True,
            tenant_id=tenant_uuid("tenant-test"),
        )

        # (a) result came from the secondary provider
        assert result.audio_data == b"fallback-audio"
        # (b) metadata reflects the provider that actually produced audio
        assert meta["resolved_provider"] == "kokoro"
        assert meta["fallback_used"] is True
        # (c) no exception propagated; the primary was in fact attempted first
        primary.generate.assert_awaited_once()
        secondary.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail_propagates(self) -> None:
        primary = _make_provider("openai")
        primary.generate = AsyncMock(side_effect=_retryable_error("primary down"))
        secondary = _make_provider("kokoro")
        secondary.generate = AsyncMock(side_effect=_retryable_error("secondary down"))
        service = self._make_service(
            {"openai": primary, "kokoro": secondary},
            _auto_router("openai", ["kokoro"]),
        )

        with pytest.raises(RuntimeError, match="All TTS providers failed"):
            await service.generate_sync(
                text="Hello world",
                provider="auto",
                voice="custom",
                no_cache=True,
                tenant_id=tenant_uuid("tenant-test"),
            )
        # No silent success: both were tried.
        primary.generate.assert_awaited_once()
        secondary.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retryable_http_error_is_caught_for_pinned_provider(self) -> None:
        # An explicit provider has no fallback chain, but the broadened except
        # must still catch the httpx-derived provider error (not leak it raw).
        only = _make_provider("openai")
        only.generate = AsyncMock(side_effect=_retryable_error())
        registry = MagicMock()
        registry.get_provider.return_value = only
        service = TTSService(registry=registry, storage=AsyncMock(), router=None)

        with pytest.raises(RuntimeError, match="All TTS providers failed"):
            await service.generate_sync(
                text="Hello world",
                provider="openai",
                voice="custom",
                no_cache=True,
                tenant_id=tenant_uuid("tenant-test"),
            )


class TestGenerateStreamFallback:
    def _make_service(
        self, providers: dict[str, AsyncMock], router: AsyncMock | None
    ) -> TTSService:
        registry = MagicMock()
        registry.get_provider.side_effect = lambda name: providers[name]
        return TTSService(registry=registry, storage=AsyncMock(), router=router)

    @pytest.mark.asyncio
    async def test_falls_back_before_first_chunk(self) -> None:
        primary = _make_provider("openai")

        async def failing_stream(**_kwargs: object) -> AsyncIterator[AudioChunk]:
            raise _retryable_error()
            yield  # pragma: no cover -- makes this an async generator

        primary.generate_stream = failing_stream

        secondary = _make_provider("kokoro")

        async def ok_stream(**_kwargs: object) -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"kokoro-chunk", chunk_index=0)
            yield AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=12)

        secondary.generate_stream = ok_stream

        service = self._make_service(
            {"openai": primary, "kokoro": secondary},
            _auto_router("openai", ["kokoro"]),
        )

        collected = [
            chunk
            async for chunk in service.generate_stream(
                tenant_id=tenant_uuid("tenant-test"),
                text="Hello world",
                provider="auto",
                voice="custom",
            )
        ]
        assert [c.data for c in collected] == [b"kokoro-chunk", b""]

    @pytest.mark.asyncio
    async def test_error_after_first_chunk_propagates(self) -> None:
        # Once bytes are on the wire we cannot fall back: the error must surface.
        provider = _make_provider("kokoro")

        async def partial_then_fail(**_kwargs: object) -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"partial", chunk_index=0)
            raise _retryable_error()

        provider.generate_stream = partial_then_fail
        service = self._make_service({"kokoro": provider}, router=None)

        collected: list[AudioChunk] = []
        with pytest.raises(httpx.HTTPStatusError):
            async for chunk in service.generate_stream(
                tenant_id=tenant_uuid("tenant-test"),
                text="Hello world",
                provider="kokoro",
                voice="custom",
            ):
                collected.append(chunk)
        assert len(collected) == 1
        assert collected[0].data == b"partial"


class TestGenerateAsyncFallback:
    @pytest.mark.asyncio
    async def test_auto_routing_is_deferred_to_process_time(self) -> None:
        # The job must store the request verbatim ("auto") so process_job can
        # re-route with the fallback chain, rather than freezing one provider.
        registry = MagicMock()
        service = TTSService(registry=registry, storage=AsyncMock(), router=None)
        db = AsyncMock()
        job = MagicMock()
        job.job_id = "job-1"

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.create_job = AsyncMock(return_value=job)
            await service.generate_async(
                db, tenant_id=tenant_uuid("t1"), text="Hi", provider="auto"
            )

        call_kwargs = mock_jm.create_job.call_args[1]
        assert call_kwargs["provider"] == "auto"
        # No premature provider resolution happened.
        registry.get_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_job_attributes_usage_to_fallback_provider(self) -> None:
        registry = MagicMock()
        storage = AsyncMock()
        storage.store = AsyncMock(return_value="/path/audio.mp3")
        service = TTSService(registry=registry, storage=storage, router=None)

        db = AsyncMock()
        db.commit = AsyncMock()

        job = MagicMock()
        job.text = "Hello world"
        job.provider = "auto"
        job.voice = "default"
        job.speed = 1.0
        job.output_format = "mp3"
        job.model = None
        job.text_length = 11
        job.routing_strategy = "balanced"

        result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        gen_meta = {
            "resolved_provider": "kokoro",
            "estimated_cost": 0.0,
            "latency_ms": 12,
            "cache_hit": False,
        }

        with (
            patch.object(service, "generate_sync", AsyncMock(return_value=(result, gen_meta))),
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            await service.process_job(db, "job-1", tenant_uuid("t1"))

        usage_kwargs = mock_jm.record_usage.call_args[1]
        # Usage is billed to the provider that actually produced audio, not "auto".
        assert usage_kwargs["provider"] == "kokoro"
