"""Stub TTS provider — returns silent audio in the same format as Kokoro.

Designed for offline pipeline tests, CI smoke runs, and local development
without paid cloud TTS calls or heavy local-model downloads.

The stub mirrors :class:`KokoroProvider` so the rest of the pipeline (FFmpeg
concat in :mod:`src.utils.audio`, downstream video_production_service) cannot
distinguish stub output from real Kokoro output by format alone:

* sample rate: 24000 Hz (matches ``KOKORO_SAMPLE_RATE``)
* channels: mono
* sample width: 16-bit signed PCM
* WAV: written via :mod:`wave` (stdlib, no ffmpeg)
* MP3: WAV → MP3 via :func:`src.utils.audio.wav_to_mp3` (same path Kokoro uses)
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import wave
from pathlib import Path

from src.providers.base import GenerationResult, TTSProvider, VoiceMetadata
from src.utils.audio import wav_to_mp3

logger = logging.getLogger(__name__)

STUB_SAMPLE_RATE = 24000
STUB_CHANNELS = 1
STUB_SAMPLE_WIDTH = 2  # bytes — 16-bit PCM

# Heuristic mirroring Kokoro: roughly 15 characters per second of speech
# at speed=1.0. Used only when the caller does not supply ``duration_seconds``.
_CHARS_PER_SECOND = 15.0
_MIN_DURATION_SECONDS = 1.0
_MAX_DURATION_SECONDS = 600.0

STUB_VOICES: list[VoiceMetadata] = [
    VoiceMetadata(
        id="stub_default",
        display_name="Stub (Silent)",
        gender="unknown",
        language="en",
        languages=["en"],
    ),
]


def _silent_wav_bytes(duration_seconds: float, sample_rate: int = STUB_SAMPLE_RATE) -> bytes:
    """Return a valid mono 16-bit PCM WAV file containing silence.

    Args:
        duration_seconds: Length of the audio. Clamped to
            ``[_MIN_DURATION_SECONDS, _MAX_DURATION_SECONDS]``.
        sample_rate: Output sample rate in Hz.

    Returns:
        Bytes of a complete RIFF/WAVE file with zero-filled samples.
    """
    clamped = max(_MIN_DURATION_SECONDS, min(_MAX_DURATION_SECONDS, duration_seconds))
    n_frames = round(clamped * sample_rate)
    # Real silent samples — zero-filled little-endian 16-bit PCM.
    silence = b"\x00\x00" * n_frames

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(STUB_CHANNELS)
        wav.setsampwidth(STUB_SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(silence)
    return buf.getvalue()


class StubProvider(TTSProvider):
    """Offline TTS provider that emits silent audio in Kokoro's format."""

    name = "stub"
    display_name = "Stub (silent, offline)"
    provider_type = "local"
    max_text_length = 100000
    sample_rate = STUB_SAMPLE_RATE
    requires_api_key = False
    supports_streaming = False
    supports_ssml = False
    # The stub returns SILENCE. It must never win an ``auto`` routing decision or
    # a fallback hop behind a real provider — that returns HTTP 200 with a silent
    # track, which is worse than a failure because nothing downstream can tell.
    # It stays reachable when pinned by name (``provider="stub"``) or when it is
    # the only configured provider (``TTS_PROVIDER=stub``, the offline/CI path).
    implicit_selection = False

    def __init__(self, default_voice: str = "stub_default") -> None:
        self._default_voice = default_voice

    async def is_available(self) -> bool:
        """The stub is always ready — it has no external dependencies.

        Availability is deliberately unconditional: registration already means
        an operator set ``STUB_ENABLED``/``TTS_PROVIDER=stub``, and the pinned
        path checks this same flag before generating. What keeps the stub out of
        an *implicit* win is :attr:`implicit_selection`, not this method.
        """
        return True

    @staticmethod
    def _estimate_duration(text: str, speed: float) -> float:
        """Estimate output duration from text length and speed multiplier."""
        if speed <= 0:
            speed = 1.0
        raw = len(text) / _CHARS_PER_SECOND / speed
        return max(_MIN_DURATION_SECONDS, raw)

    def _generate_sync(
        self,
        text: str,
        speed: float,
        output_format: str,
        duration_override: float | None,
    ) -> GenerationResult:
        """Synchronous generation — called via :func:`asyncio.to_thread`."""
        duration = (
            duration_override
            if duration_override is not None
            else self._estimate_duration(text, speed)
        )
        wav_bytes = _silent_wav_bytes(duration, sample_rate=STUB_SAMPLE_RATE)

        if output_format == "wav":
            return GenerationResult(
                audio_data=wav_bytes,
                format="wav",
                sample_rate=STUB_SAMPLE_RATE,
                duration_seconds=duration,
                metadata={"provider": self.name, "voice_used": self._default_voice},
            )

        # Default: WAV → MP3 via ffmpeg, mirroring KokoroProvider.
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "stub.wav"
            wav_path.write_bytes(wav_bytes)
            mp3_path = wav_to_mp3(wav_path)
            mp3_bytes = mp3_path.read_bytes()

        return GenerationResult(
            audio_data=mp3_bytes,
            format="mp3",
            sample_rate=STUB_SAMPLE_RATE,
            duration_seconds=duration,
            metadata={"provider": self.name, "voice_used": self._default_voice},
        )

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> GenerationResult:
        """Generate silent audio shaped like Kokoro output.

        ``kwargs`` may include ``duration_seconds`` (float) to force a precise
        output length — useful for golden-fixture pipelines that need a
        deterministic duration regardless of input text.
        """
        duration_override_raw = kwargs.get("duration_seconds")
        duration_override: float | None
        if duration_override_raw is None:
            duration_override = None
        elif isinstance(duration_override_raw, (int, float)) and not isinstance(
            duration_override_raw, bool
        ):
            duration_override = float(duration_override_raw)
        else:
            raise TypeError(
                f"duration_seconds must be a number, got {type(duration_override_raw).__name__}"
            )

        logger.debug(
            "Stub generate: voice=%s format=%s text_len=%d duration_override=%s",
            voice,
            output_format,
            len(text),
            duration_override,
        )
        return await asyncio.to_thread(
            self._generate_sync, text, speed, output_format, duration_override
        )

    async def list_voices(self) -> list[VoiceMetadata]:
        """Return the single synthetic stub voice."""
        return list(STUB_VOICES)

    def get_default_voice(self) -> str:
        """Return the default stub voice ID."""
        return self._default_voice
