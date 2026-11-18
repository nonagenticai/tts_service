"""Unit tests for concurrency semaphore controls."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import ConcurrencySettings
from src.providers.base import GenerationResult
from src.providers.registry import ProviderRegistry
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid


def _make_tts_service(concurrency_settings=None):
    """Create a TTSService with mocked dependencies."""
    registry = ProviderRegistry()
    storage = MagicMock()
    storage.store = AsyncMock(return_value="/fake/path")
    storage.retrieve = AsyncMock(return_value=b"audio")
    return TTSService(
        registry=registry,
        storage=storage,
        redis_client=None,
        router=None,
        notification_service=None,
        concurrency_settings=concurrency_settings,
    )


class TestConcurrencySettings:
    def test_defaults(self):
        cs = ConcurrencySettings()
        assert cs.max_concurrent_generations == 10
        assert cs.max_concurrent_batch_jobs == 20
        assert cs.openai_max_concurrent == 5
        assert cs.elevenlabs_max_concurrent == 5
        assert cs.kokoro_max_concurrent == 3
        assert cs.kitten_max_concurrent == 3


class TestGenerationSemaphore:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_generation(self):
        """Test that the generation semaphore limits concurrent calls."""
        cs = ConcurrencySettings(max_concurrent_generations=2)
        tts = _make_tts_service(cs)

        # Verify the semaphore was created with the right value
        assert tts._generation_semaphore._value == 2

    @pytest.mark.asyncio
    async def test_provider_semaphores_created(self):
        """Test per-provider semaphores are created correctly."""
        cs = ConcurrencySettings(
            openai_max_concurrent=3,
            kokoro_max_concurrent=1,
        )
        tts = _make_tts_service(cs)

        assert "openai" in tts._provider_semaphores
        assert "kokoro" in tts._provider_semaphores
        assert tts._provider_semaphores["openai"]._value == 3
        assert tts._provider_semaphores["kokoro"]._value == 1

    @pytest.mark.asyncio
    async def test_batch_semaphore_property(self):
        """Test the batch_semaphore property."""
        cs = ConcurrencySettings(max_concurrent_batch_jobs=15)
        tts = _make_tts_service(cs)
        assert tts.batch_semaphore._value == 15


class TestSemaphoreConcurrencyEnforcement:
    @pytest.mark.asyncio
    async def test_generation_semaphore_enforced(self):
        """Test that generation calls actually go through the semaphore."""
        cs = ConcurrencySettings(max_concurrent_generations=1)
        tts = _make_tts_service(cs)

        # Track concurrent execution count
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        original_generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"test_audio",
                format="mp3",
                sample_rate=24000,
                duration_seconds=1.0,
            )
        )

        async def tracked_generate(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)
            result = await original_generate(*args, **kwargs)
            async with lock:
                current_concurrent -= 1
            return result

        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.generate = tracked_generate
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="alloy")
        mock_provider.list_voices = AsyncMock(return_value=[])
        mock_provider.sample_rate = 24000
        tts._registry.register(mock_provider)

        # Run multiple generations concurrently
        tasks = [
            asyncio.create_task(
                tts.generate_sync(
                    text="hello",
                    provider="openai",
                    voice="alloy",
                    tenant_id=tenant_uuid("test"),
                    no_cache=True,
                )
            )
            for _ in range(3)
        ]
        await asyncio.gather(*tasks)

        # With semaphore=1, max concurrent should be 1
        assert max_concurrent == 1
