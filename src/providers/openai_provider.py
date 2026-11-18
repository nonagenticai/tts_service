"""OpenAI TTS provider (cloud, requires API key)."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 -- imported for TimeoutExpired only, no process is spawned here
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from src.providers.base import AudioChunk, GenerationResult, TTSProvider, VoiceMetadata
from src.utils.audio import probe_duration
from src.utils.retry import async_retry_with_backoff

logger = logging.getLogger(__name__)

OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

OPENAI_VOICES = [
    VoiceMetadata(id="alloy", display_name="Alloy", gender="neutral", language="en"),
    VoiceMetadata(id="echo", display_name="Echo", gender="male", language="en"),
    VoiceMetadata(id="fable", display_name="Fable", gender="neutral", language="en"),
    VoiceMetadata(id="onyx", display_name="Onyx", gender="male", language="en"),
    VoiceMetadata(id="nova", display_name="Nova", gender="female", language="en"),
    VoiceMetadata(id="shimmer", display_name="Shimmer", gender="female", language="en"),
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


class OpenAIProvider(TTSProvider):
    """OpenAI TTS API provider."""

    name = "openai"
    display_name = "OpenAI TTS"
    provider_type = "cloud"
    max_text_length = 4096
    sample_rate = 24000
    requires_api_key = True
    supports_streaming = True

    def __init__(
        self,
        api_key: str = "",
        default_voice: str = "alloy",
        model: str = "tts-1-hd",
        timeout: float = 60.0,
        streaming_chunk_size: int = 65536,
    ) -> None:
        self._api_key = api_key
        self._default_voice = default_voice
        self._model = model
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
        """Call the OpenAI TTS API with retry on transient errors.

        Args:
            text: Text to synthesize.
            voice: OpenAI voice ID (e.g. "alloy").
            speed: Playback speed multiplier.
            output_format: "mp3" or "wav".

        Returns:
            GenerationResult with audio bytes.

        Raises:
            ValueError: If API key is missing or text exceeds limit.
            RuntimeError: If the API returns an empty response.
        """
        if not self._api_key:
            raise ValueError("OpenAI TTS API key is not configured")
        if len(text) > self.max_text_length:
            raise ValueError(
                f"Text length ({len(text)} chars) exceeds OpenAI TTS limit "
                f"of {self.max_text_length} characters"
            )

        async def _call() -> httpx.Response:
            resp = await self._client.post(
                OPENAI_TTS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": text,
                    "voice": voice,
                    "response_format": output_format,
                    "speed": speed,
                },
            )
            _raise_for_status_retryable(resp)
            return resp

        logger.debug(
            "OpenAI TTS API call: voice=%s model=%s text_len=%d", voice, self._model, len(text)
        )
        start = time.monotonic()
        response = await async_retry_with_backoff(
            _call,
            max_retries=2,
            retryable_exceptions=(httpx.TransportError, _RetryableHTTPError),
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        if not response.content:
            logger.error("OpenAI TTS returned empty response after %.1fms", elapsed_ms)
            raise RuntimeError("OpenAI TTS returned empty response")

        logger.info(
            "OpenAI TTS API completed: voice=%s size=%d elapsed=%.1fms",
            voice,
            len(response.content),
            elapsed_ms,
        )

        content_type = response.headers.get("content-type", "")
        fmt = "mp3" if "mpeg" in content_type else output_format

        # Probe actual duration via ffprobe (OpenAI doesn't return it).
        # ``None`` means UNMEASURED and is carried as such all the way to the API
        # boundary and the cache — never collapsed to 0.0, which is a real
        # measurement a caller is entitled to believe.
        duration: float | None = None
        # ``tmp_path`` is bound BEFORE the write so an exception anywhere after
        # NamedTemporaryFile(delete=False) still reaches the unlink. Without the
        # try/finally a probe failure leaked one temp file per request.
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(response.content)
                tmp.flush()
            duration = probe_duration(tmp_path)
        except OSError, TimeoutError, ValueError, RuntimeError, subprocess.TimeoutExpired:
            # ffprobe runs with a timeout; TimeoutExpired subclasses
            # SubprocessError, so none of the other arms catch it. Without this
            # arm a hung ffprobe hard-failed the whole generate call instead of
            # leaving the duration UNMEASURED (same defect already fixed in the
            # ElevenLabs sibling).
            logger.debug("Failed to probe audio duration", exc_info=True)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        return GenerationResult(
            audio_data=response.content,
            format=fmt,
            sample_rate=self.sample_rate,
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
        """Stream audio chunks from the OpenAI TTS API via chunked transfer.

        Args:
            text: Text to synthesize.
            voice: OpenAI voice ID.
            speed: Playback speed multiplier.
            output_format: Audio format.

        Yields:
            AudioChunk objects, with is_final=True on the last one.
        """
        if not self._api_key:
            raise ValueError("OpenAI TTS API key is not configured")

        chunk_index = 0
        total_bytes = 0

        async with self._client.stream(
            "POST",
            OPENAI_TTS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "input": text,
                "voice": voice,
                "response_format": output_format,
                "speed": speed,
            },
        ) as response:
            response.raise_for_status()
            async for raw_chunk in response.aiter_bytes(
                chunk_size=self._streaming_chunk_size,
            ):
                total_bytes += len(raw_chunk)
                yield AudioChunk(
                    data=raw_chunk,
                    chunk_index=chunk_index,
                    format=output_format,
                    sample_rate=self.sample_rate,
                )
                chunk_index += 1

        # The SYNC path already rejects an empty body; the STREAM path did not,
        # so an upstream 200-with-no-body yielded only the ``is_final`` sentinel
        # and read downstream as a completed, billable synthesis.
        if total_bytes == 0:
            raise RuntimeError(
                "OpenAI TTS stream returned no audio bytes (empty 200 body); "
                "refusing to report a zero-byte synthesis as complete"
            )

        yield AudioChunk(
            data=b"",
            chunk_index=chunk_index,
            is_final=True,
            format=output_format,
            sample_rate=self.sample_rate,
            total_bytes=total_bytes,
        )

    async def list_voices(self) -> list[VoiceMetadata]:
        """Return the static list of OpenAI TTS voice metadata."""
        return list(OPENAI_VOICES)

    def get_default_voice(self) -> str:
        """Return the configured default OpenAI voice ID."""
        return self._default_voice
