"""Unit tests for TTS providers."""

import pytest

from src.providers.base import GenerationResult, VoiceMetadata


class TestVoiceMetadata:
    def test_voice_metadata_creation(self):
        voice = VoiceMetadata(id="test", display_name="Test Voice")
        assert voice.id == "test"
        assert voice.display_name == "Test Voice"
        assert voice.gender == "unknown"
        assert voice.language == "en"
        assert voice.preview_url is None


class TestGenerationResult:
    def test_generation_result_creation(self):
        result = GenerationResult(
            audio_data=b"fake_audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.5,
        )
        assert result.audio_data == b"fake_audio"
        assert result.format == "mp3"
        assert result.sample_rate == 24000
        assert result.duration_seconds == 1.5
        assert result.metadata == {}


class TestKokoroProvider:
    def test_kokoro_voice_to_lang_code(self):
        from src.providers.kokoro_provider import KokoroProvider

        assert KokoroProvider._voice_to_lang_code("af_heart") == "a"
        assert KokoroProvider._voice_to_lang_code("bf_emma") == "b"
        assert KokoroProvider._voice_to_lang_code("") == "a"
        assert KokoroProvider._voice_to_lang_code("xf_unknown") == "a"

    def test_kokoro_default_voice(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        assert provider.get_default_voice() == "af_heart"
        assert provider.name == "kokoro"
        assert provider.provider_type == "local"
        assert provider.requires_api_key is False

    @pytest.mark.asyncio
    async def test_kokoro_list_voices(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        voices = await provider.list_voices()
        assert len(voices) > 0
        assert all(isinstance(v, VoiceMetadata) for v in voices)


class TestKittenProvider:
    def test_kitten_default_voice(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        assert provider.get_default_voice() == "Bella"
        assert provider.name == "kitten"
        assert provider.provider_type == "local"
        assert provider.requires_api_key is False

    def test_kitten_resolve_model_id(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        assert provider._resolve_model_id(None) == "KittenML/kitten-tts-mini-0.8"
        assert provider._resolve_model_id("mini") == "KittenML/kitten-tts-mini-0.8"
        assert provider._resolve_model_id("nano") == "KittenML/kitten-tts-nano-0.2"
        assert provider._resolve_model_id("custom/model") == "custom/model"

    @pytest.mark.asyncio
    async def test_kitten_list_voices(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        voices = await provider.list_voices()
        assert len(voices) == 8
        voice_ids = [v.id for v in voices]
        assert "Bella" in voice_ids
        assert "Leo" in voice_ids


class TestOpenAIProvider:
    def test_openai_defaults(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        assert provider.get_default_voice() == "alloy"
        assert provider.name == "openai"
        assert provider.provider_type == "cloud"
        assert provider.requires_api_key is True
        assert provider.max_text_length == 4096

    @pytest.mark.asyncio
    async def test_openai_is_available_with_key(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_openai_is_available_without_key(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_openai_text_length_validation(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(ValueError, match="exceeds"):
            await provider.generate(text="x" * 5000, voice="alloy")


class TestElevenLabsProvider:
    def test_elevenlabs_defaults(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test-key")
        assert provider.get_default_voice() == "rachel"
        assert provider.name == "elevenlabs"
        assert provider.max_text_length == 5000

    @pytest.mark.asyncio
    async def test_elevenlabs_text_length_validation(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test-key")
        with pytest.raises(ValueError, match="exceeds"):
            await provider.generate(text="x" * 6000, voice="rachel")


class TestProviderRegistry:
    def test_register_and_get(self):
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        provider = OpenAIProvider(api_key="test")
        registry.register(provider)

        assert registry.get_provider("openai") is provider
        assert "openai" in registry.provider_names

    def test_get_unknown_provider_raises(self):
        from src.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get_provider("nonexistent")

    def test_list_providers(self):
        from src.providers.kokoro_provider import KokoroProvider
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry.register(KokoroProvider())
        registry.register(OpenAIProvider(api_key="test"))  # pragma: allowlist secret

        providers = registry.list_providers()
        assert len(providers) == 2
