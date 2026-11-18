"""Extended unit tests for TTS service orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.providers.base import GenerationResult
from src.services.tts_service import TTSService, _compute_cache_key, _estimate_cost
from tests.conftest import tenant_uuid

# ============================================================================
# _compute_cache_key
# ============================================================================


class TestComputeCacheKey:
    def test_returns_sha256_hex(self):
        key = _compute_cache_key("tenant", "Hello", "kokoro", "af_heart", 1.0, "mp3")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_inputs_different_keys(self):
        k1 = _compute_cache_key(tenant_uuid("t1"), "Hello", "kokoro", "af_heart", 1.0, "mp3")
        k2 = _compute_cache_key(tenant_uuid("t2"), "Hello", "kokoro", "af_heart", 1.0, "mp3")
        assert k1 != k2

    def test_tenant_isolation(self):
        k1 = _compute_cache_key(tenant_uuid("tenant-a"), "text", "kokoro", "v", 1.0, "mp3")
        k2 = _compute_cache_key(tenant_uuid("tenant-b"), "text", "kokoro", "v", 1.0, "mp3")
        assert k1 != k2

    def test_deterministic(self):
        k1 = _compute_cache_key("t", "text", "p", "v", 1.0, "mp3")
        k2 = _compute_cache_key("t", "text", "p", "v", 1.0, "mp3")
        assert k1 == k2


# ============================================================================
# _estimate_cost
# ============================================================================


class TestEstimateCost:
    def test_free_provider_zero_cost(self):
        cost = _estimate_cost("kokoro", 1000)
        assert cost == 0.0

    def test_free_provider_kitten_zero_cost(self):
        cost = _estimate_cost("kitten", 1000)
        assert cost == 0.0

    def test_openai_cost_matches_rate(self):
        """OpenAI should charge $0.015 per 1000 chars."""
        cost = _estimate_cost("openai", 1000)
        assert cost == 0.015

    def test_elevenlabs_cost_matches_rate(self):
        """ElevenLabs should charge $0.030 per 1000 chars."""
        cost = _estimate_cost("elevenlabs", 1000)
        assert cost == 0.030

    def test_cloud_cost_scales_linearly(self):
        """2000 chars should cost double 1000 chars."""
        cost_1k = _estimate_cost("openai", 1000)
        cost_2k = _estimate_cost("openai", 2000)
        assert cost_2k == cost_1k * 2

    def test_zero_characters_zero_cost(self):
        cost = _estimate_cost("openai", 0)
        assert cost == 0.0

    def test_unknown_provider_zero_cost(self):
        cost = _estimate_cost("nonexistent", 1000)
        assert cost == 0.0


# ============================================================================
# TTSService - Extended Tests
# ============================================================================


class TestTTSServiceResolveVoice:
    def _make_service(self):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage)

    def test_resolve_voice_with_explicit_voice(self):
        service = self._make_service()
        provider = MagicMock()
        result = service._resolve_voice(provider, "custom_voice")
        assert result == "custom_voice"
        provider.get_default_voice.assert_not_called()

    def test_resolve_voice_uses_default_when_none(self):
        service = self._make_service()
        provider = MagicMock()
        provider.get_default_voice.return_value = "default_voice"
        result = service._resolve_voice(provider, None)
        assert result == "default_voice"
        provider.get_default_voice.assert_called_once()

    def test_resolve_voice_uses_default_when_empty(self):
        service = self._make_service()
        provider = MagicMock()
        provider.get_default_voice.return_value = "default_voice"
        result = service._resolve_voice(provider, "")
        assert result == "default_voice"
        provider.get_default_voice.assert_called_once()

    def test_resolve_voice_uses_default_when_string_default(self):
        """The literal string 'default' should trigger default voice lookup."""
        service = self._make_service()
        provider = MagicMock()
        provider.get_default_voice.return_value = "af_heart"
        result = service._resolve_voice(provider, "default")
        assert result == "af_heart"
        provider.get_default_voice.assert_called_once()


class TestTTSServiceProcessText:
    def _make_service(self):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage)

    def test_plain_text_passthrough(self):
        service = self._make_service()
        result = service._process_text("Hello world", "kokoro")
        assert result == "Hello world"

    def test_ssml_auto_detection(self):
        service = self._make_service()
        result = service._process_text("<speak>Hello</speak>", "kokoro")
        # Should be processed (converted for kokoro)
        assert "Hello" in result

    def test_ssml_explicitly_false(self):
        service = self._make_service()
        result = service._process_text("<speak>Hello</speak>", "kokoro", ssml=False)
        assert result == "<speak>Hello</speak>"

    def test_ssml_explicitly_true(self):
        service = self._make_service()
        result = service._process_text("<speak>Hello</speak>", "kokoro", ssml=True)
        assert "Hello" in result


class TestTTSServiceDetectLanguage:
    def _make_service(self):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage)

    @patch("src.services.language_detector._get_detector")
    def test_detect_language_high_confidence_returns_lang(self, mock_get_detector):
        """When lingua detects a language with high confidence, _detect_language returns it."""
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "SPANISH"
        mock_top.value = 0.95
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        service = self._make_service()
        lang, _voice = service._detect_language("Hola mundo", "openai", None)
        assert lang == "es"

    @patch("src.services.language_detector._get_detector")
    def test_detect_language_low_confidence_returns_none(self, mock_get_detector):
        """When confidence is below threshold, _detect_language returns (None, None)."""
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "FRENCH"
        mock_top.value = 0.3
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        service = self._make_service()
        lang, voice = service._detect_language("ambiguous text", "kokoro", None)
        # Low confidence -> detect_language returns ("en", 0.3) -> confidence < 0.8
        # -> _detect_language returns (None, None)
        assert lang is None
        assert voice is None

    @patch("src.services.language_detector._get_detector")
    def test_detect_language_no_detector_returns_none(self, mock_get_detector):
        """When lingua is not installed, returns (None, None)."""
        mock_get_detector.return_value = None
        service = self._make_service()
        lang, voice = service._detect_language("Hello", "kokoro", None)
        assert lang is None
        assert voice is None

    @patch("src.services.language_detector._get_detector")
    def test_detect_language_picks_best_voice_when_no_explicit_voice(self, mock_get_detector):
        """When voice is None and lang is detected, should try to find best matching voice."""
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "FRENCH"
        mock_top.value = 0.95
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        service = self._make_service()
        mock_voice_fr = MagicMock(id="fr_voice", language="fr")
        mock_voice_en = MagicMock(id="en_voice", language="en")
        lang, voice = service._detect_language(
            "Bonjour le monde", "openai", None, [mock_voice_en, mock_voice_fr]
        )
        assert lang == "fr"
        assert voice == "fr_voice"

    @patch("src.services.language_detector._get_detector")
    def test_detect_language_skips_voice_selection_when_voice_provided(self, mock_get_detector):
        """When an explicit voice is passed, detected lang returned but voice is None."""
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "SPANISH"
        mock_top.value = 0.95
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        service = self._make_service()
        lang, voice = service._detect_language("Hola", "openai", "alloy")
        assert lang == "es"
        assert voice is None  # explicit voice was provided, no auto-selection


class TestTTSServiceCaching:
    def _make_service(self, redis=None):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage, redis_client=redis)

    @pytest.mark.asyncio
    async def test_check_cache_no_redis(self):
        service = self._make_service(redis=None)
        data, key, duration = await service._check_cache("t", "text", "p", "v", 1.0, "mp3")
        assert data is None
        assert key == ""
        assert duration is None

    @pytest.mark.asyncio
    async def test_store_cache_no_redis(self):
        service = self._make_service(redis=None)
        # Should not raise
        await service._store_cache("key", "/path", 1.0, "mp3")

    @pytest.mark.asyncio
    async def test_check_cache_miss(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = True
        service._settings.cache.key_prefix = "tts:"

        data, key, duration = await service._check_cache("t", "text", "p", "v", 1.0, "mp3")
        assert data is None
        # key should be a 64-char hex SHA256 hash
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
        assert duration is None
        # Redis should have been queried with the prefixed key
        mock_redis.get.assert_called_once_with(f"tts:{key}")

    @pytest.mark.asyncio
    async def test_check_cache_hit(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(
            return_value={"audio_path": "/path/to/audio.mp3", "duration": 2.5}
        )
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = True
        service._settings.cache.key_prefix = "tts:"
        service._storage.retrieve = AsyncMock(return_value=b"cached audio")

        data, _key, duration = await service._check_cache("t", "text", "p", "v", 1.0, "mp3")
        assert data == b"cached audio"
        assert duration == 2.5

    @pytest.mark.asyncio
    async def test_check_cache_hit_missing_duration(self):
        """A cache entry with no duration field reads back as UNMEASURED (None).

        It must NOT default to 0.0: absent means nobody measured it, and a
        substituted number here is indistinguishable from a real zero-length
        clip on every hit for the whole TTL."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value={"audio_path": "/path/to/audio.mp3"})
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = True
        service._settings.cache.key_prefix = "tts:"
        service._storage.retrieve = AsyncMock(return_value=b"cached audio")

        data, _key, duration = await service._check_cache("t", "text", "p", "v", 1.0, "mp3")
        assert data == b"cached audio"
        assert duration is None

    @pytest.mark.asyncio
    async def test_check_cache_hit_missing_audio(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value={"audio_path": "/path/to/audio.mp3"})
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = True
        service._settings.cache.key_prefix = "tts:"
        service._storage.retrieve = AsyncMock(return_value=None)

        data, _key, _duration = await service._check_cache("t", "text", "p", "v", 1.0, "mp3")
        assert data is None

    @pytest.mark.asyncio
    async def test_store_cache_saves_metadata_with_correct_data(self):
        mock_redis = AsyncMock()
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = True
        service._settings.cache.key_prefix = "tts:"
        service._settings.cache.default_ttl = 3600

        await service._store_cache("key123", "/path/audio.mp3", 2.0, "mp3")
        mock_redis.set.assert_called_once()

        # Validate the actual arguments passed to redis.set
        call_args = mock_redis.set.call_args
        stored_key = call_args[0][0]
        stored_value = call_args[0][1]
        stored_ttl = call_args[1]["ttl"]

        assert stored_key == "tts:key123"
        assert stored_value == {"audio_path": "/path/audio.mp3", "duration": 2.0, "format": "mp3"}
        assert stored_ttl == 3600

    @pytest.mark.asyncio
    async def test_store_cache_disabled_is_noop(self):
        mock_redis = AsyncMock()
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = False

        await service._store_cache("key123", "/path/audio.mp3", 2.0, "mp3")
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_cache_empty_key_is_noop(self):
        mock_redis = AsyncMock()
        service = self._make_service(redis=mock_redis)
        service._settings.cache.enabled = True

        await service._store_cache("", "/path/audio.mp3", 2.0, "mp3")
        mock_redis.set.assert_not_called()


class TestTTSServicePostProcessing:
    def _make_service(self):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage)

    @pytest.mark.asyncio
    async def test_no_post_processing(self):
        service = self._make_service()
        audio_data, duration = await service._apply_post_processing(b"audio", "mp3", 24000, None)
        assert audio_data == b"audio"
        assert duration == 0.0

    @pytest.mark.asyncio
    async def test_invalid_post_processing_type(self):
        service = self._make_service()
        audio_data, duration = await service._apply_post_processing(
            b"audio", "mp3", 24000, "not a config"
        )
        assert audio_data == b"audio"
        assert duration == 0.0


class TestTTSServiceNotify:
    def _make_service(self, notifications=None):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage, notification_service=notifications)

    @pytest.mark.asyncio
    async def test_notify_no_service(self):
        service = self._make_service(notifications=None)
        # Should not raise
        await service._notify("job_started", "j1", tenant_uuid("t1"))

    @pytest.mark.asyncio
    async def test_notify_with_exception(self):
        mock_ns = AsyncMock()
        mock_ns.publish_event = AsyncMock(side_effect=Exception("notification error"))
        service = self._make_service(notifications=mock_ns)
        # Notification errors should not propagate (it's a non-NotificationService instance)
        await service._notify("job_started", "j1", tenant_uuid("t1"))


class TestTTSServiceResolveProvider:
    def _make_service(self, router=None):
        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage, router=router)

    @pytest.mark.asyncio
    async def test_resolve_explicit_provider(self):
        service = self._make_service()
        mock_provider = MagicMock()
        service._registry.get_provider.return_value = mock_provider

        p, name = await service._resolve_provider("kokoro", "Hello")
        assert p is mock_provider
        assert name == "kokoro"

    @pytest.mark.asyncio
    async def test_resolve_auto_no_router(self):
        service = self._make_service(router=None)
        mock_provider = MagicMock()
        service._registry.get_provider.return_value = mock_provider

        p, name = await service._resolve_provider("auto", "Hello")
        assert p is mock_provider
        assert name == "auto"

    @pytest.mark.asyncio
    async def test_generate_sync_no_cache(self):
        service = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.list_voices = AsyncMock(return_value=[])
        expected_result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        mock_provider.generate = AsyncMock(return_value=expected_result)
        service._registry.get_provider.return_value = mock_provider

        result, meta = await service.generate_sync(
            text="Hello", provider="kokoro", no_cache=True, tenant_id=tenant_uuid("tenant-test")
        )
        assert result.audio_data == b"audio"
        assert meta.get("cached") is False

    @pytest.mark.asyncio
    async def test_generate_sync_with_language(self):
        service = self._make_service()
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        expected_result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        mock_provider.generate = AsyncMock(return_value=expected_result)
        service._registry.get_provider.return_value = mock_provider

        _result, meta = await service.generate_sync(
            text="Hello", provider="kokoro", language="en", tenant_id=tenant_uuid("tenant-test")
        )
        assert meta.get("detected_language") == "en"
