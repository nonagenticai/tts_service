"""Kokoro-82M TTS provider (local, free, no API key required)."""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from src.providers.base import AudioChunk, GenerationResult, TTSProvider, VoiceMetadata
from src.utils.audio import is_ffmpeg_available, wav_to_mp3

if TYPE_CHECKING:
    import numpy as np


class _KokoroSegment(Protocol):
    """Protocol for a single Kokoro pipeline output segment."""

    @property
    def audio(self) -> np.ndarray | None: ...


class _KokoroPipeline(Protocol):
    """Protocol matching the kokoro.KPipeline callable interface."""

    def __call__(self, text: str, voice: str = ..., speed: float = ...) -> Any: ...


logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000

_KOKORO_LANG_CODES: dict[str, str] = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Brazilian Portuguese
    "z": "z",  # Chinese
}

KOKORO_VOICES = [
    VoiceMetadata(
        id="af_heart",
        display_name="Heart (American Female)",
        gender="female",
        language="en-US",
        languages=["en"],
    ),
    VoiceMetadata(
        id="af_bella",
        display_name="Bella (American Female)",
        gender="female",
        language="en-US",
        languages=["en"],
    ),
    VoiceMetadata(
        id="af_sarah",
        display_name="Sarah (American Female)",
        gender="female",
        language="en-US",
        languages=["en"],
    ),
    VoiceMetadata(
        id="am_adam",
        display_name="Adam (American Male)",
        gender="male",
        language="en-US",
        languages=["en"],
    ),
    VoiceMetadata(
        id="am_michael",
        display_name="Michael (American Male)",
        gender="male",
        language="en-US",
        languages=["en"],
    ),
    VoiceMetadata(
        id="bf_emma",
        display_name="Emma (British Female)",
        gender="female",
        language="en-GB",
        languages=["en"],
    ),
    VoiceMetadata(
        id="bm_george",
        display_name="George (British Male)",
        gender="male",
        language="en-GB",
        languages=["en"],
    ),
    VoiceMetadata(
        id="ef_dora",
        display_name="Dora (Spanish Female)",
        gender="female",
        language="es",
        languages=["es"],
    ),
    VoiceMetadata(
        id="ff_siwis",
        display_name="Siwis (French Female)",
        gender="female",
        language="fr",
        languages=["fr"],
    ),
    VoiceMetadata(
        id="hf_alpha",
        display_name="Alpha (Hindi Female)",
        gender="female",
        language="hi",
        languages=["hi"],
    ),
    VoiceMetadata(
        id="if_sara",
        display_name="Sara (Italian Female)",
        gender="female",
        language="it",
        languages=["it"],
    ),
    VoiceMetadata(
        id="jf_alpha",
        display_name="Alpha (Japanese Female)",
        gender="female",
        language="ja",
        languages=["ja"],
    ),
    VoiceMetadata(
        id="pf_dora",
        display_name="Dora (Portuguese Female)",
        gender="female",
        language="pt",
        languages=["pt"],
    ),
    VoiceMetadata(
        id="zf_xiaobei",
        display_name="Xiaobei (Chinese Female)",
        gender="female",
        language="zh",
        languages=["zh"],
    ),
]


class KokoroProvider(TTSProvider):
    """Local Kokoro-82M TTS provider.

    Requires kokoro, soundfile, torch, espeak-ng, and ffmpeg.
    """

    name = "kokoro"
    display_name = "Kokoro-82M"
    provider_type = "local"
    max_text_length = 10000
    sample_rate = KOKORO_SAMPLE_RATE
    requires_api_key = False
    supports_streaming = True

    _pipelines: ClassVar[dict[str, _KokoroPipeline]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, default_voice: str = "af_heart") -> None:
        self._default_voice = default_voice

    @classmethod
    def _get_pipeline(cls, lang_code: str) -> _KokoroPipeline:
        """Return a cached KPipeline for the given language code."""
        with cls._lock:
            if lang_code not in cls._pipelines:
                try:
                    from kokoro import KPipeline
                except ImportError as exc:
                    raise RuntimeError(
                        "Kokoro TTS requires extra dependencies. Install them with:\n"
                        "  pip install tts-service[kokoro]\n"
                        "or:\n"
                        "  pip install kokoro soundfile torch\n"
                        "You also need espeak-ng installed (brew install espeak-ng on macOS)."
                    ) from exc
                logger.info("Initializing Kokoro pipeline for lang_code='%s'...", lang_code)
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
                    warnings.filterwarnings(
                        "ignore",
                        message=".*dropout option adds dropout after all but last recurrent layer.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*torch.nn.utils.weight_norm.*is deprecated.*",
                        category=FutureWarning,
                    )
                    try:
                        import torch

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        logger.info("Kokoro using device=%s", device)
                        cls._pipelines[lang_code] = KPipeline(
                            lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=device
                        )
                    except SystemExit as exc:
                        raise RuntimeError(
                            "Kokoro pipeline initialization failed (SystemExit). "
                            "Ensure spaCy model 'en_core_web_sm' is installed: "
                            "python -m spacy download en_core_web_sm"
                        ) from exc
            return cls._pipelines[lang_code]

    @staticmethod
    def _voice_to_lang_code(voice: str) -> str:
        if not voice:
            return "a"
        prefix = voice[0].lower()
        code = _KOKORO_LANG_CODES.get(prefix)
        if code is None:
            logger.warning(
                "Unknown Kokoro voice prefix '%s' in voice '%s'; "
                "falling back to 'a' (American English)",
                prefix,
                voice,
            )
            return "a"
        return code

    def _generate_sync(
        self, text: str, voice: str, speed: float, output_format: str
    ) -> GenerationResult:
        """Synchronous generation — called via asyncio.to_thread."""
        import numpy as np

        lang_code = self._voice_to_lang_code(voice)
        pipeline = self._get_pipeline(lang_code)

        segments: list[np.ndarray] = []
        for result in pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                segments.append(result.audio)

        if not segments:
            raise RuntimeError(f"Kokoro produced no audio for text: {text[:80]!r}")

        audio = np.concatenate(segments)
        duration = len(audio) / KOKORO_SAMPLE_RATE

        if output_format == "wav":
            import io

            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio, KOKORO_SAMPLE_RATE, format="WAV")
            return GenerationResult(
                audio_data=buf.getvalue(),
                format="wav",
                sample_rate=KOKORO_SAMPLE_RATE,
                duration_seconds=duration,
            )

        # Default: WAV → MP3 via ffmpeg
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            sf.write(str(wav_path), audio, KOKORO_SAMPLE_RATE)
            mp3_path = wav_to_mp3(wav_path)
            audio_data = mp3_path.read_bytes()

        return GenerationResult(
            audio_data=audio_data,
            format="mp3",
            sample_rate=KOKORO_SAMPLE_RATE,
            duration_seconds=duration,
        )

    async def is_available(self) -> bool:
        """Check if Kokoro dependencies (kokoro, espeak-ng, ffmpeg) are installed."""
        import importlib.util

        if importlib.util.find_spec("kokoro") is None:
            return False
        if not shutil.which("espeak-ng"):
            return False
        return is_ffmpeg_available()

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> GenerationResult:
        """Generate audio via Kokoro pipeline in a background thread.

        Args:
            text: Text to synthesize.
            voice: Kokoro voice ID (e.g. "af_heart").
            speed: Playback speed multiplier.
            output_format: "mp3" or "wav".

        Returns:
            GenerationResult with audio bytes and metadata.
        """
        logger.debug(
            "Kokoro generate: voice=%s format=%s text_len=%d", voice, output_format, len(text)
        )
        import time as _time

        start = _time.monotonic()
        result = await asyncio.to_thread(self._generate_sync, text, voice, speed, output_format)
        elapsed_ms = (_time.monotonic() - start) * 1000
        logger.info(
            "Kokoro generation completed: voice=%s format=%s duration=%.2fs elapsed=%.1fms",
            voice,
            output_format,
            result.duration_seconds,
            elapsed_ms,
        )
        return result

    def _generate_segments_to_queue(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: str,
        q: queue.Queue[AudioChunk | None],
        stop_event: threading.Event,
        error_holder: list[BaseException],
    ) -> None:
        """Sync function that yields segments into a queue for async consumption.

        Any failure is recorded in ``error_holder`` so :meth:`generate_stream`
        can re-raise it on the consuming task. It is NOT swallowed: a pipeline
        that dies (espeak, OOM, a bad phoneme) used to be logged and then
        followed by the ``is_final=True`` sentinel anyway, which made a failed
        synthesis byte-for-byte indistinguishable from a successful one — the
        caller billed the full text and served an empty 200.

        ``soundfile`` and ``_get_pipeline`` are imported/resolved INSIDE the
        ``try`` on purpose: raising out of this thread body skipped the poison
        pill entirely and hung the consumer on ``q.get()`` forever.
        """
        chunk_index = 0
        total_bytes = 0

        try:
            import soundfile as sf

            lang_code = self._voice_to_lang_code(voice)
            pipeline = self._get_pipeline(lang_code)

            for result in pipeline(text, voice=voice, speed=speed):
                if stop_event.is_set():
                    break
                if result.audio is not None:
                    buf = io.BytesIO()
                    if output_format == "wav":
                        sf.write(buf, result.audio, KOKORO_SAMPLE_RATE, format="WAV")
                    else:
                        sf.write(buf, result.audio, KOKORO_SAMPLE_RATE, format="WAV")
                        # Convert WAV buffer to MP3 via temp file
                        with tempfile.TemporaryDirectory() as tmpdir:
                            wav_path = Path(tmpdir) / "seg.wav"
                            wav_path.write_bytes(buf.getvalue())
                            mp3_path = wav_to_mp3(wav_path)
                            buf = io.BytesIO(mp3_path.read_bytes())

                    data = buf.getvalue()
                    total_bytes += len(data)
                    q.put(
                        AudioChunk(
                            data=data,
                            chunk_index=chunk_index,
                            format=output_format,
                            sample_rate=KOKORO_SAMPLE_RATE,
                        )
                    )
                    chunk_index += 1
        except Exception as exc:
            error_holder.append(exc)
            logger.exception("Kokoro segment streaming error")
        finally:
            # The final sentinel is a SUCCESS marker — only emit it when the
            # pipeline actually ran to completion. On failure the consumer
            # re-raises ``error_holder[0]`` instead, so the stream can never be
            # reported as produced (and therefore can never be billed).
            if not error_holder:
                q.put(
                    AudioChunk(
                        data=b"",
                        chunk_index=chunk_index,
                        is_final=True,
                        format=output_format,
                        sample_rate=KOKORO_SAMPLE_RATE,
                        total_bytes=total_bytes,
                    )
                )
            q.put(None)  # poison pill

    async def generate_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> AsyncIterator[AudioChunk]:
        """Stream Kokoro audio chunks via a thread-safe queue.

        Runs the synchronous Kokoro pipeline in a background thread and
        yields AudioChunk objects as they become available.
        """
        logger.debug(
            "Kokoro streaming: voice=%s format=%s text_len=%d", voice, output_format, len(text)
        )
        q: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=16)
        stop_event = threading.Event()
        error_holder: list[BaseException] = []

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            self._generate_segments_to_queue,
            text,
            voice,
            speed,
            output_format,
            q,
            stop_event,
            error_holder,
        )
        future.add_done_callback(
            lambda f: (
                logger.error("Kokoro executor failed: %s", f.exception())
                if not f.cancelled() and f.exception()
                else None
            )
        )

        try:
            while True:
                chunk = await asyncio.to_thread(q.get)
                if chunk is None:
                    break
                yield chunk
            # Re-raise on the consuming task. Chunks already yielded stay
            # delivered (they are real audio the client keeps); the caller
            # decides whether that means "fall back" or "meter and re-raise".
            if error_holder:
                raise error_holder[0]
        finally:
            stop_event.set()
            # Drain remaining items so the thread can finish
            while not future.done():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    async def list_voices(self) -> list[VoiceMetadata]:
        """Return the static list of Kokoro voice metadata."""
        return list(KOKORO_VOICES)

    def get_default_voice(self) -> str:
        """Return the configured default Kokoro voice ID."""
        return self._default_voice
