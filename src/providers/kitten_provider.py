"""KittenTTS provider (local, free, multiple model sizes)."""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from src.providers.base import GenerationResult, TTSProvider, VoiceMetadata
from src.utils.audio import wav_to_mp3

if TYPE_CHECKING:
    from kittentts import KittenTTS

logger = logging.getLogger(__name__)

KITTEN_SAMPLE_RATE = 24000

KITTEN_VOICES = [
    VoiceMetadata(id="Bella", display_name="Bella", gender="female", language="en"),
    VoiceMetadata(id="Jasper", display_name="Jasper", gender="male", language="en"),
    VoiceMetadata(id="Luna", display_name="Luna", gender="female", language="en"),
    VoiceMetadata(id="Bruno", display_name="Bruno", gender="male", language="en"),
    VoiceMetadata(id="Rosie", display_name="Rosie", gender="female", language="en"),
    VoiceMetadata(id="Hugo", display_name="Hugo", gender="male", language="en"),
    VoiceMetadata(id="Kiki", display_name="Kiki", gender="female", language="en"),
    VoiceMetadata(id="Leo", display_name="Leo", gender="male", language="en"),
]

KITTEN_MODELS = {
    "mini": "KittenML/kitten-tts-mini-0.8",
    "micro": "KittenML/kitten-tts-micro-0.5",
    "nano": "KittenML/kitten-tts-nano-0.2",
    "nano-int8": "KittenML/kitten-tts-nano-int8-0.2",
}


class KittenProvider(TTSProvider):
    """KittenTTS provider supporting multiple model sizes.

    Requires the kittentts package.
    """

    name = "kitten"
    display_name = "KittenTTS"
    provider_type = "local"
    max_text_length = 10000
    sample_rate = KITTEN_SAMPLE_RATE
    requires_api_key = False

    _models: ClassVar[dict[str, KittenTTS]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        default_voice: str = "Bella",
        default_model_id: str = "KittenML/kitten-tts-mini-0.8",
    ) -> None:
        self._default_voice = default_voice
        self._default_model_id = default_model_id

    @classmethod
    def _get_model(cls, model_id: str) -> KittenTTS:
        """Return a cached KittenTTS model instance."""
        with cls._lock:
            if model_id not in cls._models:
                try:
                    from kittentts import KittenTTS
                except ImportError as exc:
                    raise RuntimeError(
                        "KittenTTS requires extra dependencies. Install with:\n"
                        "  pip install tts-service[kitten]\n"
                        "or:\n"
                        "  pip install kittentts"
                    ) from exc
                logger.info("Loading KittenTTS model: %s", model_id)
                cls._models[model_id] = KittenTTS(model_path=model_id)
            return cls._models[model_id]

    def _resolve_model_id(self, model: str | None) -> str:
        """Resolve a model name/alias to a full model ID."""
        if model is None:
            return self._default_model_id
        if model in KITTEN_MODELS:
            return KITTEN_MODELS[model]
        return model

    def _generate_sync(
        self, text: str, voice: str, speed: float, output_format: str, model_id: str
    ) -> GenerationResult:
        """Synchronous generation — called via asyncio.to_thread."""
        import numpy as np
        import soundfile as sf

        m = self._get_model(model_id)
        try:
            audio = m.generate(text, voice=voice, speed=speed)
        except TypeError:
            logger.warning(
                "KittenTTS model %s does not support speed parameter, ignoring", model_id
            )
            audio = m.generate(text, voice=voice)

        audio_array = audio if isinstance(audio, np.ndarray) else np.array(audio, dtype=np.float32)

        duration = len(audio_array) / KITTEN_SAMPLE_RATE

        if output_format == "wav":
            buf = io.BytesIO()
            sf.write(buf, audio_array, KITTEN_SAMPLE_RATE, format="WAV")
            return GenerationResult(
                audio_data=buf.getvalue(),
                format="wav",
                sample_rate=KITTEN_SAMPLE_RATE,
                duration_seconds=duration,
            )

        # WAV → MP3 via ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            sf.write(str(wav_path), audio_array, KITTEN_SAMPLE_RATE)
            mp3_path = wav_to_mp3(wav_path)
            audio_data = mp3_path.read_bytes()

        return GenerationResult(
            audio_data=audio_data,
            format="mp3",
            sample_rate=KITTEN_SAMPLE_RATE,
            duration_seconds=duration,
        )

    async def is_available(self) -> bool:
        """Check if the kittentts package is importable."""
        import importlib.util

        return importlib.util.find_spec("kittentts") is not None

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        **kwargs: object,
    ) -> GenerationResult:
        """Generate audio via KittenTTS in a background thread.

        Args:
            text: Text to synthesize.
            voice: KittenTTS voice name (e.g. "Bella").
            speed: Playback speed multiplier.
            output_format: "mp3" or "wav".
            **kwargs: Optional "model" key to select a specific model size.

        Returns:
            GenerationResult with audio bytes and metadata.
        """
        model = kwargs.get("model")
        model_id = self._resolve_model_id(str(model) if model else None)
        logger.debug("Kitten generate: voice=%s model=%s text_len=%d", voice, model_id, len(text))
        import time as _time

        start = _time.monotonic()
        result = await asyncio.to_thread(
            self._generate_sync, text, voice, speed, output_format, model_id
        )
        elapsed_ms = (_time.monotonic() - start) * 1000
        logger.info(
            "Kitten generation completed: voice=%s model=%s duration=%.2fs elapsed=%.1fms",
            voice,
            model_id,
            result.duration_seconds,
            elapsed_ms,
        )
        return result

    async def list_voices(self) -> list[VoiceMetadata]:
        """Return the static list of KittenTTS voice metadata."""
        return list(KITTEN_VOICES)

    def get_default_voice(self) -> str:
        """Return the configured default KittenTTS voice name."""
        return self._default_voice
