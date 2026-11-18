"""ElevenLabs TTS provider (cloud, requires API key)."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 -- imported for TimeoutExpired only, no process is spawned here
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

import httpx

from src.providers.base import AudioChunk, GenerationResult, TTSProvider, VoiceMetadata
from src.utils.audio import probe_duration
from src.utils.retry import async_retry_with_backoff

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
ELEVENLABS_STREAM_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# ElevenLabs ``output_format`` query-parameter values, keyed by the format name
# this service speaks. Previously NO ``output_format`` was sent at all, so the
# API always applied its own default (``mp3_44100_128``) and the caller's
# requested format was silently ignored. ``mp3_44100_128`` is exactly that
# default, so sending it explicitly is byte-identical to the old behaviour while
# making the contract visible. The API cannot emit a WAV *container* (its
# non-mp3 options are raw PCM / u-law), so a ``wav`` request is served as mp3 —
# and, critically, REPORTED as mp3, so the declared format always matches the
# bytes.
_ELEVENLABS_OUTPUT_FORMATS = {"mp3": "mp3_44100_128"}
_ELEVENLABS_FALLBACK_FORMAT = "mp3"

# Sample rate of the bytes each API ``output_format`` token actually produces.
# ElevenLabs encodes the rate INTO the token (``mp3_<rate>_<kbps>``), so this is
# derived from the token rather than hand-maintained — a declared rate that
# contradicts the requested format is exactly the bug this replaces (the class
# declared 24000 while every request asked for, and got, 44.1 kHz).


def _rate_from_api_format(api_format: str) -> int:
    """Parse the sample rate out of an ElevenLabs ``output_format`` token.

    ``mp3_44100_128`` -> ``44100``. Raises ``ValueError`` on a token that does
    not carry a parsable rate, so a newly added format cannot silently reinstate
    a wrong declared rate.
    """
    parts = api_format.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Cannot derive sample rate from ElevenLabs format {api_format!r}")
    return int(parts[1])


_ELEVENLABS_SAMPLE_RATES = {
    fmt: _rate_from_api_format(api_fmt) for fmt, api_fmt in _ELEVENLABS_OUTPUT_FORMATS.items()
}
# The rate of the format we fall back to, i.e. what the provider produces by
# default. Declared on the class so callers that read the class attribute
# (post-processing, streaming chunks, API metadata) get the truth.
ELEVENLABS_SAMPLE_RATE = _ELEVENLABS_SAMPLE_RATES[_ELEVENLABS_FALLBACK_FORMAT]

ELEVENLABS_VOICES = [
    VoiceMetadata(
        id="rachel",
        display_name="Rachel",
        gender="female",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="drew",
        display_name="Drew",
        gender="male",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="clyde",
        display_name="Clyde",
        gender="male",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="paul",
        display_name="Paul",
        gender="male",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="domi",
        display_name="Domi",
        gender="female",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="dave",
        display_name="Dave",
        gender="male",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="fin",
        display_name="Fin",
        gender="male",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
    VoiceMetadata(
        id="sarah",
        display_name="Sarah",
        gender="female",
        language="en",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "zh", "ko"],
    ),
]


class _RetryableHTTPError(httpx.HTTPStatusError):
    """Raised only for retryable status codes (429, 5xx)."""


def _raise_for_status_retryable(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableHTTPError(
                message=str(exc), request=exc.request, response=exc.response
            ) from exc
        raise


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS API provider."""

    name = "elevenlabs"
    display_name = "ElevenLabs"
    provider_type = "cloud"
    max_text_length = 5000
    # 44100, derived from the ``mp3_44100_128`` output format every request asks
    # for. NOT 24000 — that stale value contradicted the bytes and was handed to
    # the post-processor as the input rate and reported as response metadata.
    sample_rate = ELEVENLABS_SAMPLE_RATE
    requires_api_key = True
    supports_streaming = True
    supports_ssml = True

    def __init__(
        self,
        api_key: str = "",
        default_voice: str = "rachel",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        timeout: float = 60.0,
        streaming_chunk_size: int = 65536,
    ) -> None:
        self._api_key = api_key
        self._default_voice = default_voice
        self._model_id = model_id
        self._stability = stability
        self._similarity_boost = similarity_boost
        self._timeout = timeout
        self._streaming_chunk_size = streaming_chunk_size
        self._client = httpx.AsyncClient(timeout=self._timeout, http2=True)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    async def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self._api_key)

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> GenerationResult:
        """Call the ElevenLabs TTS API with retry on transient errors.

        Args:
            text: Text to synthesize.
            voice: ElevenLabs voice ID (e.g. "rachel").
            speed: Playback speed multiplier.
            output_format: Requested audio format. Passed to the API as its
                ``output_format`` parameter when supported; otherwise mp3 is
                produced. ``GenerationResult.format`` always reports what was
                actually produced, never what was requested.

        Returns:
            GenerationResult with audio bytes.

        Raises:
            ValueError: If API key is missing or text exceeds limit.
            RuntimeError: If the API returns an empty response.
        """
        if not self._api_key:
            raise ValueError("ElevenLabs API key is not configured")
        if len(text) > self.max_text_length:
            raise ValueError(
                f"Text length ({len(text)} chars) exceeds ElevenLabs TTS limit "
                f"of {self.max_text_length} characters"
            )

        url = ELEVENLABS_TTS_URL_TEMPLATE.format(voice=quote(voice, safe=""))

        # Resolve the format we can actually produce, and tell the API about it.
        produced_format = (
            output_format
            if output_format in _ELEVENLABS_OUTPUT_FORMATS
            else _ELEVENLABS_FALLBACK_FORMAT
        )
        if produced_format != output_format:
            logger.warning(
                "ElevenLabs cannot produce %r audio; returning %s (format is reported honestly)",
                output_format,
                produced_format,
            )
        api_output_format = _ELEVENLABS_OUTPUT_FORMATS[produced_format]

        raw_stability = kwargs.get("stability", self._stability)
        stability = (
            float(raw_stability)
            if isinstance(raw_stability, (int, float, str))
            else self._stability
        )
        raw_sim = kwargs.get("similarity_boost", self._similarity_boost)
        similarity_boost = (
            float(raw_sim) if isinstance(raw_sim, (int, float, str)) else self._similarity_boost
        )

        async def _call() -> httpx.Response:
            resp = await self._client.post(
                url,
                params={"output_format": api_output_format},
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self._model_id,
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                    },
                    "speed": speed,
                },
            )
            _raise_for_status_retryable(resp)
            return resp

        logger.debug(
            "ElevenLabs API call: voice=%s model=%s text_len=%d", voice, self._model_id, len(text)
        )
        start = time.monotonic()
        response = await async_retry_with_backoff(
            _call,
            max_retries=2,
            retryable_exceptions=(httpx.TransportError, _RetryableHTTPError),
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        if not response.content:
            logger.error("ElevenLabs TTS returned empty response after %.1fms", elapsed_ms)
            raise RuntimeError("ElevenLabs TTS returned empty response")

        logger.info(
            "ElevenLabs API completed: voice=%s size=%d elapsed=%.1fms",
            voice,
            len(response.content),
            elapsed_ms,
        )

        # Probe actual duration via ffprobe (ElevenLabs doesn't return it).
        # ``None`` means UNMEASURED and is carried as such all the way to the API
        # boundary and the cache — never collapsed to 0.0, which is a real
        # measurement a caller is entitled to believe.
        duration: float | None = None
        # ``tmp_path`` is bound BEFORE the write so an exception anywhere after
        # NamedTemporaryFile(delete=False) still reaches the unlink. Without the
        # try/finally a probe failure leaked one temp file per request.
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{produced_format}", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(response.content)
                tmp.flush()
            duration = probe_duration(tmp_path)
        except OSError, TimeoutError, ValueError, RuntimeError, subprocess.TimeoutExpired:
            # ffprobe is invoked with a timeout; TimeoutExpired subclasses
            # SubprocessError, so none of the other arms catch it.
            logger.debug("Failed to probe audio duration", exc_info=True)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        return GenerationResult(
            # ``format`` describes the bytes we are actually returning, never the
            # format the caller asked for. Everything downstream (storage key
            # extension, Content-Type, batch concat) keys off this field.
            audio_data=response.content,
            format=produced_format,
            # Per-call rate of the format we actually requested, so adding a
            # second entry to _ELEVENLABS_OUTPUT_FORMATS cannot desync the
            # reported rate from the bytes.
            sample_rate=_ELEVENLABS_SAMPLE_RATES[produced_format],
            duration_seconds=duration,
        )

    async def generate_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from the ElevenLabs streaming API.

        Yields:
            AudioChunk objects, with is_final=True on the last one.
        """
        if not self._api_key:
            raise ValueError("ElevenLabs API key is not configured")

        url = ELEVENLABS_STREAM_URL_TEMPLATE.format(voice=quote(voice, safe=""))
        chunk_index = 0
        total_bytes = 0

        async with self._client.stream(
            "POST",
            url,
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": self._model_id,
                "voice_settings": {
                    "stability": self._stability,
                    "similarity_boost": self._similarity_boost,
                },
                "speed": speed,
            },
            params={"optimize_streaming_latency": 3},
        ) as response:
            response.raise_for_status()
            async for raw_chunk in response.aiter_bytes(
                chunk_size=self._streaming_chunk_size,
            ):
                total_bytes += len(raw_chunk)
                yield AudioChunk(
                    data=raw_chunk,
                    chunk_index=chunk_index,
                    format="mp3",
                    sample_rate=self.sample_rate,
                )
                chunk_index += 1

        # The SYNC path already rejects an empty body; the STREAM path did not,
        # so an upstream 200-with-no-body yielded only the ``is_final`` sentinel
        # and read downstream as a completed, billable synthesis.
        if total_bytes == 0:
            raise RuntimeError(
                "ElevenLabs stream returned no audio bytes (empty 200 body); "
                "refusing to report a zero-byte synthesis as complete"
            )

        yield AudioChunk(
            data=b"",
            chunk_index=chunk_index,
            is_final=True,
            format="mp3",
            sample_rate=self.sample_rate,
            total_bytes=total_bytes,
        )

    async def list_voices(self) -> list[VoiceMetadata]:
        """Return the static list of ElevenLabs voice metadata."""
        return list(ELEVENLABS_VOICES)

    def get_default_voice(self) -> str:
        """Return the configured default ElevenLabs voice ID."""
        return self._default_voice
