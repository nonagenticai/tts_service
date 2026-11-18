"""Abstract base class for TTS providers."""

from __future__ import annotations

import base64
import subprocess  # nosec B404 -- imported for TimeoutExpired only, no process is spawned here
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

# Exceptions that represent a provider/transport failure rather than a caller
# programming error. Both cloud providers raise ``httpx.HTTPError`` subclasses
# (including ``_RetryableHTTPError``, which ``utils.retry`` re-raises after its
# retries are exhausted); local providers raise ``OSError`` / ``RuntimeError`` /
# ``ValueError`` from torch / ffmpeg / model code. The multi-provider fallback
# loop catches exactly this set so one provider's failure degrades to the next,
# while genuine bugs (``TypeError``, ``KeyError``, ...) still propagate.
#
# ``subprocess.TimeoutExpired`` is listed EXPLICITLY: it subclasses
# ``SubprocessError``, NOT ``OSError`` and NOT the builtin ``TimeoutError``, so
# none of the other entries catch it. Every ffmpeg/ffprobe call in this repo
# passes a ``timeout=``, so a hung binary raises it — and before it was listed
# here it escaped every handler and left the job non-terminal until a restart.
PROVIDER_FAILURE_ERRORS: tuple[type[Exception], ...] = (
    httpx.HTTPError,
    OSError,
    RuntimeError,
    ValueError,
    subprocess.TimeoutExpired,
)


@dataclass
class VoiceMetadata:
    """Metadata for a TTS voice."""

    id: str
    display_name: str
    gender: str = "unknown"
    language: str = "en"
    languages: list[str] = field(default_factory=lambda: ["en"])
    preview_url: str | None = None


@dataclass
class AudioChunk:
    """A chunk of audio data for streaming."""

    data: bytes
    chunk_index: int
    is_final: bool = False
    format: str = "mp3"
    sample_rate: int = 24000
    total_duration_seconds: float | None = None
    total_bytes: int | None = None
    # Which provider actually produced these bytes. Set by
    # ``TTSService.generate_stream`` on the final sentinel, because the streaming
    # fallback loop lives there and the route has no other way to know: the
    # request may have said ``auto``, or named a provider that failed before its
    # first chunk. ``None`` = not known; consumers must omit rather than fall
    # back to what was requested. Buffered counterpart: ``meta["resolved_provider"]``.
    provider: str | None = None

    def to_base64(self) -> str:
        """Encode audio data as a base64 ASCII string for JSON transport."""
        return base64.b64encode(self.data).decode("ascii")


@dataclass
class GenerationResult:
    """Result of a TTS generation operation."""

    audio_data: bytes
    format: str
    sample_rate: int
    # ``None`` = the duration could not be measured (ffprobe unavailable or
    # failed). Distinct from ``0.0``, which is a real measurement of a real
    # zero-length file. Consumers must OMIT the value rather than substitute.
    duration_seconds: float | None
    metadata: dict[str, str] = field(default_factory=dict)


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    name: str
    display_name: str
    provider_type: str  # "local" or "cloud"
    max_text_length: int
    sample_rate: int = 24000
    requires_api_key: bool = False
    supports_streaming: bool = False
    supports_ssml: bool = False
    # May the router pick this provider on its own — as the ``auto`` winner or as
    # an entry in a fallback chain? Real providers: yes. A provider that does not
    # actually synthesize speech (the silent stub) must set this ``False`` so a
    # genuine request can never silently degrade to it and return 200-with-silence.
    # Such a provider is still reachable when a caller pins it by name, or when it
    # is the only provider configured (``TTS_PROVIDER=stub`` / stub-only registry).
    implicit_selection: bool = True

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and ready to use."""
        ...

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> GenerationResult:
        """Generate audio from text."""
        ...

    async def generate_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks. Default: single-chunk fallback using generate()."""
        result = await self.generate(
            text=text,
            voice=voice,
            speed=speed,
            output_format=output_format,
            **kwargs,
        )
        yield AudioChunk(
            data=result.audio_data,
            chunk_index=0,
            is_final=True,
            format=result.format,
            sample_rate=result.sample_rate,
            total_duration_seconds=result.duration_seconds,
            total_bytes=len(result.audio_data),
        )

    async def close(self) -> None:
        """Release any resources held by the provider (default no-op)."""
        return

    @abstractmethod
    async def list_voices(self) -> list[VoiceMetadata]:
        """List available voices for this provider."""
        ...

    @abstractmethod
    def get_default_voice(self) -> str:
        """Get the default voice ID for this provider."""
        ...
