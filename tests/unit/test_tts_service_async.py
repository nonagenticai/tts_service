"""Unit tests for TTSService async job and streaming generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache.redis_client import RedisClient
from src.providers.base import AudioChunk, GenerationResult
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid


class TestTTSServiceGenerateStream:
    def _make_service(
        self, router: object = None, redis: RedisClient | None = None
    ) -> tuple[TTSService, MagicMock, MagicMock]:
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage, redis_client=redis, router=router)
        return service, registry, storage

    @pytest.mark.asyncio
    async def test_generate_stream_basic(self) -> None:
        service, registry, _storage = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.list_voices = AsyncMock(return_value=[])

        chunks = [
            AudioChunk(data=b"chunk1", chunk_index=0),
            AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=6),
        ]

        async def mock_stream(**_kwargs: object):
            for c in chunks:
                yield c

        mock_provider.generate_stream = mock_stream
        registry.get_provider.return_value = mock_provider

        collected = []
        async for chunk in service.generate_stream(
            tenant_id=tenant_uuid("tenant-test"), text="Hello", provider="kokoro"
        ):
            collected.append(chunk)

        assert len(collected) == 2
        assert collected[0].data == b"chunk1"
        assert collected[1].is_final is True

    @pytest.mark.asyncio
    async def test_generate_stream_unavailable(self) -> None:
        service, registry, _storage = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=False)
        registry.get_provider.return_value = mock_provider

        with pytest.raises(RuntimeError, match="not available"):
            async for _ in service.generate_stream(
                tenant_id=tenant_uuid("tenant-test"), text="Hello", provider="kokoro"
            ):
                pass

    @pytest.mark.asyncio
    async def test_generate_stream_with_model(self) -> None:
        service, registry, _storage = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.list_voices = AsyncMock(return_value=[])

        async def mock_stream(**kwargs: object):
            assert kwargs.get("model") == "mini"
            # Must emit REAL audio: a provider that yields only the empty
            # is_final sentinel is now (correctly) treated as a failed stream.
            yield AudioChunk(data=b"audio", chunk_index=0)
            yield AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=5)

        mock_provider.generate_stream = mock_stream
        registry.get_provider.return_value = mock_provider

        async for _chunk in service.generate_stream(
            tenant_id=tenant_uuid("tenant-test"), text="Hello", provider="kitten", model="mini"
        ):
            pass

    @pytest.mark.asyncio
    async def test_generate_stream_with_explicit_voice(self) -> None:
        service, registry, _storage = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="default")

        async def mock_stream(**kwargs: object):
            assert kwargs["voice"] == "custom_voice"
            yield AudioChunk(data=b"audio", chunk_index=0)
            yield AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=5)

        mock_provider.generate_stream = mock_stream
        registry.get_provider.return_value = mock_provider

        async for _chunk in service.generate_stream(
            tenant_id=tenant_uuid("tenant-test"),
            text="Hello",
            provider="kokoro",
            voice="custom_voice",
        ):
            pass


class TestTTSServiceGenerateAsync:
    def _make_service(self, router: object = None) -> tuple[TTSService, MagicMock, MagicMock]:
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage, router=router)
        return service, registry, storage

    @pytest.mark.asyncio
    async def test_generate_async_creates_job(self) -> None:
        service, registry, _storage = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        registry.get_provider.return_value = mock_provider

        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "job-abc"

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.create_job = AsyncMock(return_value=mock_job)

            job_id = await service.generate_async(
                db,
                tenant_id=tenant_uuid("t1"),
                text="Hello",
                provider="kokoro",
                voice="af_heart",
            )
            assert job_id == "job-abc"
            mock_jm.create_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_async_with_batch(self) -> None:
        service, registry, _storage = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        registry.get_provider.return_value = mock_provider

        db = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-batch"

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.create_job = AsyncMock(return_value=mock_job)

            job_id = await service.generate_async(
                db,
                tenant_id=tenant_uuid("t1"),
                text="Hello",
                provider="kokoro",
                batch_id="batch-1",
                caller_item_id="item-1",
            )
            assert job_id == "job-batch"
            call_kwargs = mock_jm.create_job.call_args[1]
            assert call_kwargs["batch_id"] == "batch-1"
            assert call_kwargs["caller_item_id"] == "item-1"


class TestTTSServiceProcessJob:
    def _make_service(self) -> tuple[TTSService, MagicMock, AsyncMock]:
        registry = MagicMock()
        storage = AsyncMock()
        service = TTSService(registry=registry, storage=storage)
        return service, registry, storage

    @pytest.mark.asyncio
    async def test_process_job_not_found(self) -> None:
        service, _registry, _storage = self._make_service()
        db = AsyncMock()

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(ValueError, match="Job not found"):
                await service.process_job(db, "nonexistent", tenant_uuid("t1"))

    @pytest.mark.asyncio
    async def test_process_job_success(self) -> None:
        service, _registry, storage = self._make_service()
        db = AsyncMock()
        db.commit = AsyncMock()

        mock_job = MagicMock()
        mock_job.text = "Hello"
        mock_job.provider = "kokoro"
        mock_job.voice = "af_heart"
        mock_job.speed = 1.0
        mock_job.output_format = "mp3"
        mock_job.model = None
        mock_job.text_length = 5

        mock_result = GenerationResult(
            audio_data=b"audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.0,
        )

        storage.store = AsyncMock(return_value="/path/audio.mp3")

        with (
            patch.object(
                service,
                "generate_sync",
                AsyncMock(
                    return_value=(
                        mock_result,
                        {"cached": False, "estimated_cost": 0.0, "latency_ms": 50},
                    )
                ),
            ),
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            await service.process_job(db, "j1", tenant_uuid("t1"))

            # Verify job lifecycle: processing -> completed
            mock_jm.mark_processing.assert_called_once_with(db, "j1")

            # Verify mark_completed received correct audio metadata
            completed_kwargs = mock_jm.mark_completed.call_args[1]
            assert completed_kwargs["audio_path"] == "/path/audio.mp3"
            assert completed_kwargs["audio_size_bytes"] == len(b"audio")
            assert completed_kwargs["audio_duration_seconds"] == 1.0
            assert completed_kwargs["cached"] is False

            # Verify usage was recorded with correct data
            usage_kwargs = mock_jm.record_usage.call_args[1]
            assert usage_kwargs["tenant_id"] == tenant_uuid("t1")
            assert usage_kwargs["provider"] == "kokoro"
            assert usage_kwargs["characters"] == 5
            assert usage_kwargs["audio_seconds"] == 1.0

            # Verify storage received the right audio
            storage.store.assert_called_once_with("j1", b"audio", "mp3")

            # Verify db was committed
            assert db.commit.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_job_failure(self) -> None:
        service, _registry, storage = self._make_service()
        db = AsyncMock()
        db.commit = AsyncMock()

        mock_job = MagicMock()
        mock_job.text = "Hello"
        mock_job.provider = "kokoro"
        mock_job.voice = "af_heart"
        mock_job.speed = 1.0
        mock_job.output_format = "mp3"
        mock_job.model = None
        mock_job.text_length = 5

        # Generate fails
        with (
            patch.object(
                service,
                "generate_sync",
                AsyncMock(side_effect=RuntimeError("generation failed")),
            ),
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            with pytest.raises(RuntimeError, match="generation failed"):
                await service.process_job(db, "j1", tenant_uuid("t1"))

            # Verify mark_failed received (db, job_id, error_message)
            failed_args = mock_jm.mark_failed.call_args[0]
            assert failed_args[0] is db
            assert failed_args[1] == "j1"
            assert "generation failed" in failed_args[2]

            # Verify error usage was recorded with is_error=True and 0.0 audio_seconds
            usage_kwargs = mock_jm.record_usage.call_args[1]
            assert usage_kwargs["is_error"] is True
            assert usage_kwargs["audio_seconds"] == 0.0
            assert usage_kwargs["characters"] == 5
            assert usage_kwargs["provider"] == "kokoro"

            # Storage should NOT have been called since generation failed
            storage.store.assert_not_called()


class TestTTSServiceConcatenateBatch:
    def _make_service(self) -> tuple[TTSService, MagicMock, AsyncMock]:
        registry = MagicMock()
        storage = AsyncMock()
        service = TTSService(registry=registry, storage=storage)
        return service, registry, storage

    @pytest.mark.asyncio
    async def test_concatenate_batch_no_jobs(self) -> None:
        service, _registry, _storage = self._make_service()
        db = AsyncMock()
        db.commit = AsyncMock()

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[])
            with pytest.raises(ValueError, match="No completed jobs"):
                await service.concatenate_batch(db, "batch-1", tenant_id=tenant_uuid("tenant-a"))

    @pytest.mark.asyncio
    async def test_concatenate_batch_no_audio(self) -> None:
        service, _registry, storage = self._make_service()
        db = AsyncMock()
        db.commit = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/path/audio.mp3"
        mock_job.job_id = "job-1"

        storage.retrieve = AsyncMock(return_value=None)

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
            with pytest.raises(ValueError, match="audio retrieval failed"):
                await service.concatenate_batch(db, "batch-1", tenant_id=tenant_uuid("tenant-a"))


class TestTTSServiceGenerateSyncFull:
    def _make_service(
        self, redis: RedisClient | None = None
    ) -> tuple[TTSService, MagicMock, AsyncMock]:
        registry = MagicMock()
        storage = AsyncMock()
        service = TTSService(registry=registry, storage=storage, redis_client=redis)
        return service, registry, storage

    @pytest.mark.asyncio
    async def test_generate_sync_cache_hit(self) -> None:
        mock_redis = AsyncMock()
        service, registry, _storage = self._make_service(redis=mock_redis)

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.sample_rate = 24000
        mock_provider.list_voices = AsyncMock(return_value=[])
        registry.get_provider.return_value = mock_provider

        # Mock cache to return data (audio_data, cache_key, cached_duration)
        service.set_cache_checker(AsyncMock(return_value=(b"cached audio", "cache-key", 3.5)))

        result, meta = await service.generate_sync(
            tenant_id=tenant_uuid("tenant-test"), text="Hello", provider="kokoro"
        )
        assert result.audio_data == b"cached audio"
        assert result.duration_seconds == 3.5
        assert meta["cached"] is True
        assert meta["cache_hit"] is True
        assert meta["estimated_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_sync_with_post_processing(self) -> None:
        service, registry, _storage = self._make_service()

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.list_voices = AsyncMock(return_value=[])
        gen_result = GenerationResult(
            audio_data=b"raw audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.0,
        )
        mock_provider.generate = AsyncMock(return_value=gen_result)
        registry.get_provider.return_value = mock_provider

        # Mock post-processing (returns tuple of (bytes, float))
        from src.api.schemas import PostProcessingConfig

        pp_config = PostProcessingConfig()  # Default = no processing
        with patch.object(
            service, "_apply_post_processing", AsyncMock(return_value=(b"raw audio", 1.5))
        ):
            result, _meta = await service.generate_sync(
                tenant_id=tenant_uuid("tenant-test"),
                text="Hello",
                provider="kokoro",
                post_processing=pp_config,
                no_cache=True,
            )
        assert result.audio_data == b"raw audio"
        assert result.duration_seconds == 1.5

    @pytest.mark.asyncio
    async def test_generate_sync_cache_store_error(self) -> None:
        """Cache store errors should not prevent the response."""
        mock_redis = AsyncMock()
        service, registry, storage = self._make_service(redis=mock_redis)

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.list_voices = AsyncMock(return_value=[])
        gen_result = GenerationResult(
            audio_data=b"audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.0,
        )
        mock_provider.generate = AsyncMock(return_value=gen_result)
        registry.get_provider.return_value = mock_provider

        # Cache miss, then store fails
        service.set_cache_checker(AsyncMock(return_value=(None, "cache-key", 0.0)))
        storage.store = AsyncMock(side_effect=OSError("storage error"))

        # Should still return result despite cache store failure
        result, _meta = await service.generate_sync(
            tenant_id=tenant_uuid("tenant-test"), text="Hello", provider="kokoro"
        )
        assert result.audio_data == b"audio"

    @pytest.mark.asyncio
    async def test_generate_sync_auto_detect_language(self) -> None:
        service, registry, _storage = self._make_service()

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.list_voices = AsyncMock(return_value=[])
        gen_result = GenerationResult(
            audio_data=b"audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.0,
        )
        mock_provider.generate = AsyncMock(return_value=gen_result)
        registry.get_provider.return_value = mock_provider

        # Mock language detection to detect Spanish
        with patch.object(
            service, "_detect_language", MagicMock(return_value=("es", "spanish_voice"))
        ):
            _result, meta = await service.generate_sync(
                tenant_id=tenant_uuid("tenant-test"),
                text="Hola mundo",
                provider="kokoro",
                no_cache=True,
            )
        assert meta.get("detected_language") == "es"


class TestTTSServiceResolveProviderAuto:
    def _make_service(self, router: object = None) -> tuple[TTSService, MagicMock, MagicMock]:
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage, router=router)
        return service, registry, storage

    @pytest.mark.asyncio
    async def test_auto_with_router(self) -> None:
        from src.api.schemas import RoutingStrategy
        from src.services.router import ProviderRouter, RoutingDecision

        mock_router = AsyncMock(spec=ProviderRouter)
        decision = RoutingDecision(
            provider_name="openai",
            voice="alloy",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=[],
        )
        mock_router.select_provider = AsyncMock(return_value=decision)

        service, registry, _storage = self._make_service(router=mock_router)
        mock_provider = MagicMock()
        registry.get_provider.return_value = mock_provider

        _p, name = await service._resolve_provider("auto", "Hello")
        assert name == "openai"
