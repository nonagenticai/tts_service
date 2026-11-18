"""Unit tests for the TTSProvider abstract base class fallback streaming."""

import pytest

from src.providers.base import GenerationResult, TTSProvider, VoiceMetadata


class _ConcreteProvider(TTSProvider):
    """Minimal concrete implementation for testing the base class."""

    name = "test"
    display_name = "Test Provider"
    provider_type = "local"
    max_text_length = 5000

    async def is_available(self) -> bool:
        return True

    async def generate(
        self, text: str, voice: str, speed: float = 1.0, output_format: str = "mp3", **kwargs
    ) -> GenerationResult:
        return GenerationResult(
            audio_data=b"fake-audio",
            format=output_format,
            sample_rate=24000,
            duration_seconds=1.5,
        )

    async def list_voices(self) -> list[VoiceMetadata]:
        return [VoiceMetadata(id="v1", display_name="Voice 1")]

    def get_default_voice(self) -> str:
        return "v1"


class TestTTSProviderDefaultStream:
    @pytest.mark.asyncio
    async def test_default_generate_stream_yields_single_chunk(self):
        """The base class generate_stream should fall back to generate()."""
        provider = _ConcreteProvider()
        chunks = []
        async for chunk in provider.generate_stream(text="Hello", voice="v1"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].data == b"fake-audio"
        assert chunks[0].is_final is True
        assert chunks[0].chunk_index == 0
        assert chunks[0].format == "mp3"
        assert chunks[0].total_bytes == len(b"fake-audio")
        assert chunks[0].total_duration_seconds == 1.5

    @pytest.mark.asyncio
    async def test_default_generate_stream_wav_format(self):
        provider = _ConcreteProvider()
        chunks = []
        async for chunk in provider.generate_stream(text="Hello", voice="v1", output_format="wav"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].format == "wav"

    @pytest.mark.asyncio
    async def test_concrete_provider_is_available(self):
        provider = _ConcreteProvider()
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_concrete_provider_list_voices(self):
        provider = _ConcreteProvider()
        voices = await provider.list_voices()
        assert len(voices) == 1
        assert voices[0].id == "v1"

    def test_concrete_provider_get_default_voice(self):
        provider = _ConcreteProvider()
        assert provider.get_default_voice() == "v1"


class TestVoiceMetadataDefaults:
    def test_defaults(self):
        v = VoiceMetadata(id="test", display_name="Test")
        assert v.gender == "unknown"
        assert v.language == "en"
        assert v.languages == ["en"]
        assert v.preview_url is None

    def test_custom_values(self):
        v = VoiceMetadata(
            id="custom",
            display_name="Custom",
            gender="female",
            language="fr",
            languages=["fr", "en"],
            preview_url="https://example.com/preview.mp3",
        )
        assert v.gender == "female"
        assert v.language == "fr"
        assert v.languages == ["fr", "en"]
        assert v.preview_url == "https://example.com/preview.mp3"


class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=2.0
        )
        assert r.metadata == {}
        assert r.audio_data == b"audio"
        assert r.format == "mp3"

    def test_with_metadata(self):
        r = GenerationResult(
            audio_data=b"audio",
            format="wav",
            sample_rate=44100,
            duration_seconds=3.0,
            metadata={"provider": "test"},
        )
        assert r.metadata == {"provider": "test"}
        assert r.sample_rate == 44100
