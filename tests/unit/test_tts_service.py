"""Unit tests for TTS service orchestration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.providers.base import GenerationResult
from src.services.tts_service import TTSService, _estimate_cost
from tests.conftest import tenant_uuid


class TestTTSService:
    def _make_service(self):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage)

    @pytest.mark.asyncio
    async def test_generate_sync_calls_provider(self):
        service = self._make_service()

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        expected_result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        mock_provider.generate = AsyncMock(return_value=expected_result)
        service._registry.get_provider.return_value = mock_provider

        result, meta = await service.generate_sync(
            text="Hello world",
            provider="kokoro",
            voice="af_heart",
            tenant_id=tenant_uuid("tenant-test"),
        )

        # Verify the result passes through from the provider unchanged
        assert result.audio_data == b"audio"
        assert result.format == "mp3"
        assert result.duration_seconds == 1.0

        # Verify provider was called with the correct text and voice
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["text"] == "Hello world"
        assert call_kwargs["voice"] == "af_heart"
        assert call_kwargs["speed"] == 1.0

        # Verify metadata was populated
        assert meta["resolved_provider"] == "kokoro"
        assert "latency_ms" in meta
        assert meta["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_generate_sync_unavailable_provider(self):
        service = self._make_service()

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=False)
        service._registry.get_provider.return_value = mock_provider

        with pytest.raises(RuntimeError, match="not available"):
            await service.generate_sync(
                text="Hello",
                provider="kokoro",
                voice="af_heart",
                tenant_id=tenant_uuid("tenant-test"),
            )

    @pytest.mark.asyncio
    async def test_generate_sync_with_model_kwarg(self):
        service = self._make_service()

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="Bella")
        mock_provider.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
            )
        )
        service._registry.get_provider.return_value = mock_provider

        _result, _meta = await service.generate_sync(
            text="Hello",
            provider="kitten",
            voice="Bella",
            model="mini",
            tenant_id=tenant_uuid("tenant-test"),
        )

        _, kwargs = mock_provider.generate.call_args
        assert kwargs["model"] == "mini"

    @pytest.mark.asyncio
    async def test_emit_cost_event_delegates_to_shared_scheduler(self, monkeypatch):
        """When metering_redis is present, the helper delegates to the shared
        GC-safe scheduler with this service's routing identity. The
        background-task lifecycle now lives in w2v_common.metering."""
        metering_redis = MagicMock()
        service = TTSService(
            registry=MagicMock(),
            storage=MagicMock(),
            metering_redis=metering_redis,
        )

        calls = []
        monkeypatch.setattr(
            "src.services.tts_service.schedule_cost_event",
            lambda conn, **kwargs: calls.append((conn, kwargs)),
        )

        service._emit_cost_event(
            tenant_id=tenant_uuid("t"),
            provider="kokoro",
            model="kokoro",
            text="hello",
            duration_s=1.0,
            audio_format="mp3",
            latency_ms=10,
            call_kind="sync",
            call_id="req-1",
        )

        assert len(calls) == 1
        conn, kwargs = calls[0]
        assert conn is metering_redis
        assert kwargs["service_key"] == "w2v-tts"  # pragma: allowlist secret -- routing slug
        assert kwargs["cost_source"] == "w2v-tts"
        assert kwargs["tenant_id"] == tenant_uuid("t")
        assert kwargs["provider"] == "kokoro"
        assert kwargs["input_tokens"] == len("hello")

    @pytest.mark.asyncio
    async def test_emit_cost_event_noop_when_metering_disabled(self, monkeypatch):
        """No metering_redis means the shared scheduler is never invoked."""
        service = TTSService(
            registry=MagicMock(),
            storage=MagicMock(),
            metering_redis=None,
        )

        calls = []
        monkeypatch.setattr(
            "src.services.tts_service.schedule_cost_event",
            lambda *a, **k: calls.append((a, k)),
        )

        service._emit_cost_event(
            tenant_id=tenant_uuid("t"),
            provider="kokoro",
            model="kokoro",
            text="hello",
            duration_s=1.0,
            audio_format="mp3",
            latency_ms=10,
            call_kind="sync",
            call_id="req-1",
        )
        assert calls == []

    @pytest.mark.asyncio
    async def test_emit_cost_event_bills_estimated_provider_spend_not_zero(self, monkeypatch):
        """Regression: a paid provider must be billed its estimated cost, not $0.

        ``_emit_cost_event`` previously passed ``cost_usd=0.0`` to
        ``schedule_cost_event`` — every OpenAI/ElevenLabs synthesis reported a
        free charge on the cost bus. It must now pass
        ``_estimate_cost(provider, len(text))`` (> 0 for cloud providers).
        """
        metering_redis = MagicMock()
        service = TTSService(
            registry=MagicMock(),
            storage=MagicMock(),
            metering_redis=metering_redis,
        )

        calls: list[dict] = []
        monkeypatch.setattr(
            "src.services.tts_service.schedule_cost_event",
            lambda conn, **kwargs: calls.append(kwargs),
        )

        provider = "elevenlabs"
        text = "The quick brown fox jumps over the lazy dog."
        expected_cost = _estimate_cost(provider, len(text))
        assert expected_cost > 0.0  # guard: elevenlabs is priced, not free

        service._emit_cost_event(
            tenant_id=tenant_uuid("t"),
            provider=provider,
            model="eleven_multilingual_v2",
            text=text,
            duration_s=1.0,
            audio_format="mp3",
            latency_ms=10,
            call_kind="sync",
            call_id="req-1",
        )

        assert len(calls) == 1
        assert calls[0]["cost_usd"] == expected_cost
        assert calls[0]["cost_usd"] > 0.0

    @pytest.mark.asyncio
    async def test_generate_sync_returns_metadata(self):
        service = self._make_service()

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
            )
        )
        service._registry.get_provider.return_value = mock_provider

        _result, meta = await service.generate_sync(
            text="Hello", provider="kokoro", voice="af_heart", tenant_id=tenant_uuid("tenant-test")
        )

        assert "resolved_provider" in meta
        assert meta["resolved_provider"] == "kokoro"
        assert "latency_ms" in meta
        assert "estimated_cost" in meta
