"""Extended unit tests for TTS providers to cover more code paths."""

from unittest.mock import MagicMock

import httpx
import pytest

from src.providers.base import AudioChunk

# ============================================================================
# AudioChunk
# ============================================================================


class TestAudioChunk:
    def test_to_base64(self):
        chunk = AudioChunk(data=b"hello", chunk_index=0)
        b64 = chunk.to_base64()
        assert b64 == "aGVsbG8="  # base64 of "hello"

    def test_final_chunk(self):
        chunk = AudioChunk(data=b"", chunk_index=5, is_final=True, total_bytes=1024)
        assert chunk.is_final is True
        assert chunk.total_bytes == 1024


# ============================================================================
# OpenAI Provider
# ============================================================================


class TestOpenAIProviderGenerate:
    @pytest.mark.asyncio
    async def test_generate_no_api_key_raises(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        with pytest.raises(ValueError, match="API key"):
            await provider.generate(text="Hello", voice="alloy")

    @pytest.mark.asyncio
    async def test_list_voices(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        voices = await provider.list_voices()
        assert len(voices) >= 6
        voice_ids = [v.id for v in voices]
        assert "alloy" in voice_ids
        assert "nova" in voice_ids

    @pytest.mark.asyncio
    async def test_stream_no_api_key_raises(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        with pytest.raises(ValueError, match="API key"):
            async for _ in provider.generate_stream(text="Hello", voice="alloy"):
                pass


class TestOpenAIRaiseForStatus:
    def test_success_status(self):
        from src.providers.openai_provider import _raise_for_status_retryable

        response = MagicMock()
        response.raise_for_status = MagicMock()
        # Should not raise
        _raise_for_status_retryable(response)

    def test_retryable_status(self):
        from src.providers.openai_provider import _raise_for_status_retryable, _RetryableHTTPError

        response = MagicMock()
        mock_exc = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        response.raise_for_status.side_effect = mock_exc
        with pytest.raises(_RetryableHTTPError):
            _raise_for_status_retryable(response)

    def test_non_retryable_status(self):
        from src.providers.openai_provider import _raise_for_status_retryable

        response = MagicMock()
        mock_exc = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        response.raise_for_status.side_effect = mock_exc
        with pytest.raises(httpx.HTTPStatusError):
            _raise_for_status_retryable(response)


# ============================================================================
# ElevenLabs Provider
# ============================================================================


class TestElevenLabsProviderGenerate:
    @pytest.mark.asyncio
    async def test_generate_no_api_key_raises(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="")
        with pytest.raises(ValueError, match="API key"):
            await provider.generate(text="Hello", voice="rachel")

    @pytest.mark.asyncio
    async def test_list_voices(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test")
        voices = await provider.list_voices()
        assert len(voices) >= 8
        voice_ids = [v.id for v in voices]
        assert "rachel" in voice_ids

    @pytest.mark.asyncio
    async def test_is_available_with_key(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test")
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_without_key(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="")
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_stream_no_api_key_raises(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="")
        with pytest.raises(ValueError, match="API key"):
            async for _ in provider.generate_stream(text="Hello", voice="rachel"):
                pass


class TestElevenLabsRaiseForStatus:
    def test_retryable_429(self):
        from src.providers.elevenlabs_provider import (
            _raise_for_status_retryable,
            _RetryableHTTPError,
        )

        response = MagicMock()
        mock_exc = httpx.HTTPStatusError(
            "Rate Limited",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        response.raise_for_status.side_effect = mock_exc
        with pytest.raises(_RetryableHTTPError):
            _raise_for_status_retryable(response)

    def test_non_retryable_400(self):
        from src.providers.elevenlabs_provider import _raise_for_status_retryable

        response = MagicMock()
        mock_exc = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=MagicMock(status_code=400),
        )
        response.raise_for_status.side_effect = mock_exc
        with pytest.raises(httpx.HTTPStatusError):
            _raise_for_status_retryable(response)


# ============================================================================
# Kitten Provider
# ============================================================================


class TestKittenProviderExtended:
    def test_resolve_model_micro(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        assert provider._resolve_model_id("micro") == "KittenML/kitten-tts-micro-0.5"

    def test_resolve_model_nano_int8(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        assert provider._resolve_model_id("nano-int8") == "KittenML/kitten-tts-nano-int8-0.2"


# ============================================================================
# Kokoro Provider
# ============================================================================


class TestKokoroProviderExtended:
    def test_kokoro_init_custom_voice(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider(default_voice="bf_emma")
        assert provider.get_default_voice() == "bf_emma"

    def test_kokoro_voice_to_lang_code_various(self):
        from src.providers.kokoro_provider import KokoroProvider

        assert KokoroProvider._voice_to_lang_code("ef_voice") == "e"
        assert KokoroProvider._voice_to_lang_code("jf_voice") == "j"
        assert KokoroProvider._voice_to_lang_code("zf_voice") == "z"


# ============================================================================
# B1 — declared sample_rate must match the requested/produced format
# ============================================================================


class TestElevenLabsDeclaredSampleRate:
    """The class-level ``sample_rate`` feeds post-processing and response
    metadata. It declared 24000 while every request asked for (and got)
    ``mp3_44100_128``."""

    def test_class_sample_rate_matches_requested_output_format(self):
        from src.providers.elevenlabs_provider import (
            _ELEVENLABS_FALLBACK_FORMAT,
            _ELEVENLABS_OUTPUT_FORMATS,
            ElevenLabsProvider,
        )

        api_format = _ELEVENLABS_OUTPUT_FORMATS[_ELEVENLABS_FALLBACK_FORMAT]
        assert api_format == "mp3_44100_128"
        assert ElevenLabsProvider.sample_rate == 44100

    def test_every_supported_format_declares_its_real_rate(self):
        from src.providers.elevenlabs_provider import (
            _ELEVENLABS_OUTPUT_FORMATS,
            _ELEVENLABS_SAMPLE_RATES,
        )

        assert set(_ELEVENLABS_SAMPLE_RATES) == set(_ELEVENLABS_OUTPUT_FORMATS)
        for fmt, api_format in _ELEVENLABS_OUTPUT_FORMATS.items():
            # The API token carries the rate: mp3_<rate>_<kbps>.
            assert str(_ELEVENLABS_SAMPLE_RATES[fmt]) in api_format

    def test_rate_parser_rejects_a_format_without_a_rate(self):
        from src.providers.elevenlabs_provider import _rate_from_api_format

        assert _rate_from_api_format("mp3_44100_128") == 44100
        with pytest.raises(ValueError, match="Cannot derive sample rate"):
            _rate_from_api_format("ulaw")

    @pytest.mark.asyncio
    async def test_generate_reports_the_rate_of_the_format_it_requested(self):
        import json as _json
        from unittest.mock import patch

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="xi-test")  # pragma: allowlist secret
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["output_format"] = request.url.params.get("output_format")
            _json.loads(request.content)
            return httpx.Response(200, content=b"\xff\xfbeleven")

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("src.providers.elevenlabs_provider.probe_duration", return_value=1.0):
            result = await provider.generate("Hi", "rachel")

        assert seen["output_format"] == "mp3_44100_128"
        assert result.sample_rate == 44100


# ============================================================================
# B3 — the ffprobe temp file must not leak on the error path
# ============================================================================


class TestProviderProbeTempFileCleanup:
    @staticmethod
    def _tempfiles(prefix_dir):
        return set(prefix_dir.iterdir())

    @pytest.mark.asyncio
    async def test_elevenlabs_probe_failure_leaves_no_temp_file(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        provider = ElevenLabsProvider(api_key="xi-test")  # pragma: allowlist secret

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"\xff\xfbeleven")

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch(
            "src.providers.elevenlabs_provider.probe_duration",
            side_effect=RuntimeError("ffprobe blew up"),
        ):
            result = await provider.generate("Hi", "rachel")

        assert result.duration_seconds is None  # degraded to UNMEASURED, not raised
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_openai_probe_failure_leaves_no_temp_file(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from src.providers.openai_provider import OpenAIProvider

        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        provider = OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"\xff\xfbopenai", headers={"content-type": "audio/mpeg"}
            )

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch(
            "src.providers.openai_provider.probe_duration",
            side_effect=RuntimeError("ffprobe blew up"),
        ):
            result = await provider.generate("Hi", "alloy")

        assert result.duration_seconds is None
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_openai_probe_subprocess_timeout_degrades_not_raises(self, tmp_path, monkeypatch):
        """A hung ffprobe raises ``subprocess.TimeoutExpired`` — it must degrade
        to an UNMEASURED duration like every other probe failure, not hard-fail
        generate."""
        import subprocess
        from unittest.mock import patch

        from src.providers.openai_provider import OpenAIProvider

        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        provider = OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"\xff\xfbopenai", headers={"content-type": "audio/mpeg"}
            )

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch(
            "src.providers.openai_provider.probe_duration",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=5),
        ):
            result = await provider.generate("Hi", "alloy")

        assert result.duration_seconds is None
        assert list(tmp_path.iterdir()) == []
