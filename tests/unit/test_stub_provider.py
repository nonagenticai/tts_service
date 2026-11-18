"""Real-assertion tests for the offline stub TTS provider.

These tests exercise the stub end-to-end: bytes generation, header parsing,
sample-level silence verification, and downstream pipeline compatibility
(the same FFmpeg concat code path used for Kokoro output).
"""

from __future__ import annotations

import io
import shutil
import struct
import wave

import pytest

from src.providers.base import GenerationResult
from src.providers.stub_provider import (
    STUB_SAMPLE_RATE,
    StubProvider,
    _silent_wav_bytes,
)
from src.utils.audio import concatenate_audio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_wav(data: bytes) -> tuple[wave.Wave_read, bytes]:
    """Open a WAV bytes blob and return (handle, raw frames)."""
    buf = io.BytesIO(data)
    wav = wave.open(buf, "rb")
    frames = wav.readframes(wav.getnframes())
    return wav, frames


def _max_abs_sample(frames: bytes) -> int:
    """Return max(|sample|) over a 16-bit little-endian PCM byte string."""
    if not frames:
        return 0
    n = len(frames) // 2
    samples = struct.unpack(f"<{n}h", frames)
    return max(abs(s) for s in samples)


# ---------------------------------------------------------------------------
# _silent_wav_bytes
# ---------------------------------------------------------------------------


class TestSilentWavBytes:
    def test_returns_valid_riff_wave_header(self):
        data = _silent_wav_bytes(1.0)
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"

    def test_parses_with_stdlib_wave_module(self):
        data = _silent_wav_bytes(1.0)
        wav, frames = _read_wav(data)
        try:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == STUB_SAMPLE_RATE
            assert wav.getnframes() == STUB_SAMPLE_RATE  # exactly 1 second
            assert len(frames) == STUB_SAMPLE_RATE * 2
        finally:
            wav.close()

    def test_silence_is_actually_silent(self):
        data = _silent_wav_bytes(0.5)
        _, frames = _read_wav(data)
        assert _max_abs_sample(frames) == 0

    def test_duration_is_clamped_low(self):
        # below the 1.0s minimum
        data = _silent_wav_bytes(0.0001)
        wav, _ = _read_wav(data)
        try:
            assert wav.getnframes() == STUB_SAMPLE_RATE
        finally:
            wav.close()

    def test_duration_is_clamped_high(self):
        # absurdly long → clamped to 600s
        data = _silent_wav_bytes(10_000.0)
        wav, _ = _read_wav(data)
        try:
            assert wav.getnframes() == 600 * STUB_SAMPLE_RATE
        finally:
            wav.close()


# ---------------------------------------------------------------------------
# StubProvider — async generate
# ---------------------------------------------------------------------------


class TestStubProviderAsync:
    @pytest.mark.asyncio
    async def test_is_available_always_true(self):
        provider = StubProvider()
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_generate_wav_default_duration(self):
        provider = StubProvider()
        result = await provider.generate(
            text="hello world",
            voice="stub_default",
            output_format="wav",
        )
        assert isinstance(result, GenerationResult)
        assert result.format == "wav"
        assert result.sample_rate == STUB_SAMPLE_RATE
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 1.0
        # Output is parseable WAV with the right header.
        assert result.audio_data[:4] == b"RIFF"
        wav, frames = _read_wav(result.audio_data)
        try:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert _max_abs_sample(frames) == 0
        finally:
            wav.close()

    @pytest.mark.asyncio
    async def test_generate_with_explicit_duration(self):
        provider = StubProvider()
        result = await provider.generate(
            text="anything",
            voice="stub_default",
            output_format="wav",
            duration_seconds=2.5,
        )
        assert result.duration_seconds == pytest.approx(2.5, abs=0.01)
        wav, _ = _read_wav(result.audio_data)
        try:
            # Allow ±1 frame for rounding.
            assert abs(wav.getnframes() - int(2.5 * STUB_SAMPLE_RATE)) <= 1
        finally:
            wav.close()

    @pytest.mark.asyncio
    async def test_generate_estimates_duration_from_text(self):
        # 150 chars at 15 chars/s ≈ 10s
        provider = StubProvider()
        result = await provider.generate(
            text="x" * 150,
            voice="stub_default",
            output_format="wav",
        )
        assert result.duration_seconds == pytest.approx(10.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_generate_speed_scales_duration(self):
        provider = StubProvider()
        slow = await provider.generate(
            text="x" * 150, voice="stub_default", output_format="wav", speed=1.0
        )
        fast = await provider.generate(
            text="x" * 150, voice="stub_default", output_format="wav", speed=2.0
        )
        assert fast.duration_seconds is not None
        assert slow.duration_seconds is not None
        assert fast.duration_seconds < slow.duration_seconds
        assert fast.duration_seconds == pytest.approx(slow.duration_seconds / 2, abs=0.5)

    @pytest.mark.asyncio
    async def test_generate_rejects_non_numeric_duration(self):
        provider = StubProvider()
        # ``duration_seconds`` is declared ``object`` via ``**kwargs`` so this
        # call is type-clean. The provider must reject the value at runtime.
        bad_duration: object = "long"
        with pytest.raises(TypeError, match="duration_seconds"):
            await provider.generate(
                text="hi",
                voice="stub_default",
                output_format="wav",
                duration_seconds=bad_duration,
            )

    @pytest.mark.asyncio
    async def test_list_voices(self):
        provider = StubProvider()
        voices = await provider.list_voices()
        assert len(voices) == 1
        assert voices[0].id == "stub_default"

    def test_get_default_voice(self):
        provider = StubProvider(default_voice="stub_default")
        assert provider.get_default_voice() == "stub_default"

    @pytest.mark.asyncio
    async def test_generate_mp3_format(self):
        """MP3 path: validate real bytes when ffmpeg is on PATH; otherwise
        assert the documented :class:`FileNotFoundError` is raised. Both
        branches make a real assertion — neither is a no-op skip."""
        provider = StubProvider()
        if shutil.which("ffmpeg") is not None:
            result = await provider.generate(
                text="hello",
                voice="stub_default",
                output_format="mp3",
                duration_seconds=1.0,
            )
            assert result.format == "mp3"
            assert result.sample_rate == STUB_SAMPLE_RATE
            assert len(result.audio_data) > 0
            # MP3 frame sync: every valid MP3 starts with 0xFF 0xFx (frame
            # sync) or an ID3v2 tag ('ID3').
            head = result.audio_data[:3]
            assert head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)
        else:
            with pytest.raises(FileNotFoundError, match="ffmpeg"):
                await provider.generate(
                    text="hello",
                    voice="stub_default",
                    output_format="mp3",
                    duration_seconds=1.0,
                )


# ---------------------------------------------------------------------------
# Downstream pipeline compatibility — feed stub WAV through the same FFmpeg
# concat code path that consumes Kokoro output.
# ---------------------------------------------------------------------------


class TestStubDownstreamCompatibility:
    @pytest.mark.asyncio
    async def test_wav_segments_concatenate_via_ffmpeg(self):
        """Feed stub WAV output into ``concatenate_audio`` — the same code
        path that consumes Kokoro WAV output. When ffmpeg is present we
        validate the real concatenated bytes; when absent we assert
        ``concatenate_audio`` raises ``FileNotFoundError`` with a clear
        message (its documented contract)."""
        provider = StubProvider()
        seg1 = await provider.generate(
            text="first segment",
            voice="stub_default",
            output_format="wav",
            duration_seconds=1.0,
        )
        seg2 = await provider.generate(
            text="second segment",
            voice="stub_default",
            output_format="wav",
            duration_seconds=1.5,
        )
        # Both segments — produced offline without ffmpeg — must be valid
        # WAV that downstream code can at least open. This part of the
        # assertion never depends on ffmpeg.
        for seg in (seg1, seg2):
            assert seg.audio_data[:4] == b"RIFF"
            wav, _ = _read_wav(seg.audio_data)
            try:
                assert wav.getframerate() == STUB_SAMPLE_RATE
            finally:
                wav.close()

        if shutil.which("ffmpeg") is None:
            with pytest.raises(FileNotFoundError, match="ffmpeg"):
                concatenate_audio([seg1.audio_data, seg2.audio_data], output_format="wav")
            return

        # Same code path Kokoro WAV output flows through.
        concatenated = concatenate_audio([seg1.audio_data, seg2.audio_data], output_format="wav")
        assert concatenated[:4] == b"RIFF"
        wav, frames = _read_wav(concatenated)
        try:
            assert wav.getframerate() == STUB_SAMPLE_RATE
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            # Combined ~2.5s of silence, allow ±0.1s for ffmpeg framing.
            expected_frames = int(2.5 * STUB_SAMPLE_RATE)
            assert abs(wav.getnframes() - expected_frames) < STUB_SAMPLE_RATE // 10
            # Real silence is preserved through the concat path.
            assert _max_abs_sample(frames) == 0
        finally:
            wav.close()


# ---------------------------------------------------------------------------
# Settings + registry wiring for TTS_PROVIDER=stub.
# ---------------------------------------------------------------------------


class TestStubProviderEnvWiring:
    def test_stub_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("STUB_ENABLED", raising=False)
        monkeypatch.delenv("TTS_PROVIDER", raising=False)
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        settings = Settings()
        assert settings.stub.enabled is False

    def test_stub_enabled_directly(self, monkeypatch):
        monkeypatch.setenv("STUB_ENABLED", "true")
        monkeypatch.delenv("TTS_PROVIDER", raising=False)
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        settings = Settings()
        assert settings.stub.enabled is True

    def test_tts_provider_override_forces_stub_only(self, monkeypatch):
        monkeypatch.setenv("TTS_PROVIDER", "stub")
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        settings = Settings()
        assert settings.stub.enabled is True
        assert settings.kokoro.enabled is False
        assert settings.kitten.enabled is False
        assert settings.openai_tts.enabled is False
        assert settings.elevenlabs.enabled is False

    def test_tts_provider_unknown_raises(self, monkeypatch):
        monkeypatch.setenv("TTS_PROVIDER", "totallynotaprovider")
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        with pytest.raises(ValueError, match="not a known provider"):
            Settings()

    def test_registry_includes_stub_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TTS_PROVIDER", "stub")
        from src.config import Settings, clear_settings_cache
        from src.providers.registry import create_registry

        clear_settings_cache()
        settings = Settings()
        registry = create_registry(settings)
        assert "stub" in registry.provider_names
        assert "kokoro" not in registry.provider_names

    def test_registry_excludes_stub_when_disabled(self, monkeypatch):
        monkeypatch.delenv("STUB_ENABLED", raising=False)
        monkeypatch.delenv("TTS_PROVIDER", raising=False)
        from src.config import Settings, clear_settings_cache
        from src.providers.registry import create_registry

        clear_settings_cache()
        settings = Settings()
        registry = create_registry(settings)
        assert "stub" not in registry.provider_names


class TestStubSelectability:
    """B2: reachable when pinned/only-configured, never an implicit winner."""

    @pytest.mark.asyncio
    async def test_pinned_stub_still_passes_the_availability_gate(self):
        """The pinned path (``provider="stub"``) checks ``is_available()`` before
        generating, so the stub must stay available — the exclusion lives in
        ``implicit_selection``, not here."""
        assert await StubProvider().is_available() is True

    def test_stub_is_excluded_from_implicit_selection(self):
        assert StubProvider.implicit_selection is False

    def test_all_real_providers_remain_implicitly_selectable(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider
        from src.providers.kitten_provider import KittenProvider
        from src.providers.kokoro_provider import KokoroProvider
        from src.providers.openai_provider import OpenAIProvider

        for cls in (KokoroProvider, KittenProvider, OpenAIProvider, ElevenLabsProvider):
            assert cls.implicit_selection is True, cls.__name__
