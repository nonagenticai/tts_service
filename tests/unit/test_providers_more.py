"""Additional provider tests covering more code paths."""

import pytest

# ============================================================================
# Kokoro Provider edge cases
# ============================================================================


class TestKokoroProviderEdgeCases:
    def test_voice_to_lang_code_empty(self):
        from src.providers.kokoro_provider import KokoroProvider

        assert KokoroProvider._voice_to_lang_code("") == "a"

    def test_voice_to_lang_code_unknown_prefix(self):
        from src.providers.kokoro_provider import KokoroProvider

        # 'x' is not in the lang code map -> should fallback to 'a'
        assert KokoroProvider._voice_to_lang_code("xf_unknown") == "a"

    def test_voice_to_lang_code_all_prefixes(self):
        from src.providers.kokoro_provider import KokoroProvider

        expected = {
            "a": "a",
            "b": "b",
            "e": "e",
            "f": "f",
            "h": "h",
            "i": "i",
            "j": "j",
            "p": "p",
            "z": "z",
        }
        for prefix, code in expected.items():
            assert KokoroProvider._voice_to_lang_code(f"{prefix}f_voice") == code

    @pytest.mark.asyncio
    async def test_list_voices(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        voices = await provider.list_voices()
        assert len(voices) >= 10
        voice_ids = [v.id for v in voices]
        assert "af_heart" in voice_ids

    def test_class_attributes(self):
        from src.providers.kokoro_provider import KokoroProvider

        assert KokoroProvider.name == "kokoro"
        assert KokoroProvider.provider_type == "local"
        assert KokoroProvider.requires_api_key is False
        assert KokoroProvider.supports_streaming is True


# ============================================================================
# Kitten Provider edge cases
# ============================================================================


class TestKittenProviderEdgeCases:
    def test_resolve_model_none_returns_default(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider(default_model_id="KittenML/custom-model")
        assert provider._resolve_model_id(None) == "KittenML/custom-model"

    def test_resolve_model_custom_id_passthrough(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        # A full model path not in KITTEN_MODELS should be passed through
        assert provider._resolve_model_id("org/custom-model") == "org/custom-model"

    def test_resolve_all_aliases(self):
        from src.providers.kitten_provider import KITTEN_MODELS, KittenProvider

        provider = KittenProvider()
        for alias, full_id in KITTEN_MODELS.items():
            assert provider._resolve_model_id(alias) == full_id

    @pytest.mark.asyncio
    async def test_list_voices(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        voices = await provider.list_voices()
        assert len(voices) >= 8
        voice_ids = [v.id for v in voices]
        assert "Bella" in voice_ids

    def test_get_default_voice(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider(default_voice="Hugo")
        assert provider.get_default_voice() == "Hugo"

    def test_class_attributes(self):
        from src.providers.kitten_provider import KittenProvider

        assert KittenProvider.name == "kitten"
        assert KittenProvider.provider_type == "local"
        assert KittenProvider.requires_api_key is False


# ============================================================================
# OpenAI Provider - text length validation
# ============================================================================


class TestOpenAIProviderTextValidation:
    @pytest.mark.asyncio
    async def test_generate_text_too_long(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        long_text = "x" * 5000  # exceeds 4096 limit
        with pytest.raises(ValueError, match="exceeds"):
            await provider.generate(text=long_text, voice="alloy")

    def test_get_default_voice(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test", default_voice="nova")
        assert provider.get_default_voice() == "nova"

    @pytest.mark.asyncio
    async def test_is_available_with_key(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_without_key(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        assert await provider.is_available() is False


# ============================================================================
# ElevenLabs Provider - text length validation
# ============================================================================


class TestElevenLabsProviderTextValidation:
    @pytest.mark.asyncio
    async def test_generate_text_too_long(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test-key")
        long_text = "x" * 6000  # exceeds 5000 limit
        with pytest.raises(ValueError, match="exceeds"):
            await provider.generate(text=long_text, voice="rachel")

    def test_get_default_voice(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test", default_voice="drew")
        assert provider.get_default_voice() == "drew"
