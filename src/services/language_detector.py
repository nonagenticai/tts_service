"""Language auto-detection for TTS text input."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingua import LanguageDetector

logger = logging.getLogger(__name__)

# Mapping from lingua language codes to BCP-47 language tags
_LINGUA_TO_BCP47: dict[str, str] = {
    "ENGLISH": "en",
    "SPANISH": "es",
    "FRENCH": "fr",
    "GERMAN": "de",
    "ITALIAN": "it",
    "PORTUGUESE": "pt",
    "DUTCH": "nl",
    "RUSSIAN": "ru",
    "CHINESE": "zh",
    "JAPANESE": "ja",
    "KOREAN": "ko",
    "ARABIC": "ar",
    "HINDI": "hi",
    "TURKISH": "tr",
    "POLISH": "pl",
    "SWEDISH": "sv",
    "DANISH": "da",
    "NORWEGIAN": "no",
    "FINNISH": "fi",
    "CZECH": "cs",
    "HUNGARIAN": "hu",
    "ROMANIAN": "ro",
    "GREEK": "el",
    "THAI": "th",
    "VIETNAMESE": "vi",
    "INDONESIAN": "id",
    "MALAY": "ms",
    "TAGALOG": "tl",
    "UKRAINIAN": "uk",
}

# Provider-to-language support mapping
PROVIDER_LANGUAGES: dict[str, set[str]] = {
    "kokoro": {"en", "es", "fr", "hi", "it", "ja", "pt", "zh"},
    "kitten": {"en"},
    "openai": {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "nl",
        "ru",
        "zh",
        "ja",
        "ko",
        "ar",
        "hi",
        "tr",
        "pl",
        "sv",
        "da",
        "no",
        "fi",
        "cs",
        "hu",
        "ro",
        "el",
        "th",
        "vi",
        "id",
    },
    "elevenlabs": {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "nl",
        "ru",
        "zh",
        "ja",
        "ko",
        "ar",
        "hi",
        "tr",
        "pl",
        "sv",
    },
}

# Kokoro language code mapping
KOKORO_LANG_MAP: dict[str, str] = {
    "en": "a",  # American English
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh": "z",
}


@lru_cache(maxsize=1)
def _get_detector() -> LanguageDetector | None:
    """Lazy-load the lingua language detector."""
    try:
        from lingua import LanguageDetectorBuilder

        detector: LanguageDetector = (
            LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        )
        logger.info("Lingua language detector initialized")
        return detector
    except ImportError:
        logger.warning(
            "lingua-language-detector not installed. "
            "Language auto-detection disabled. Install with: "
            "pip install lingua-language-detector"
        )
        return None


def detect_language(text: str, confidence_threshold: float = 0.8) -> tuple[str, float]:
    """Detect the language of the given text.

    Returns:
        Tuple of (language_code, confidence). Falls back to ("en", 0.0)
        if detection fails or confidence is below threshold.
    """
    detector = _get_detector()
    if detector is None:
        return "en", 0.0

    try:
        confidence_values = detector.compute_language_confidence_values(text)
        if not confidence_values:
            return "en", 0.0

        top = confidence_values[0]
        lang_name = top.language.name
        confidence = top.value

        bcp47 = _LINGUA_TO_BCP47.get(lang_name, "en")

        if confidence < confidence_threshold:
            logger.debug(
                "Language detection confidence %.2f below threshold %.2f for '%s'; "
                "falling back to 'en'",
                confidence,
                confidence_threshold,
                lang_name,
            )
            return "en", confidence

        logger.debug("Detected language: %s (%.2f confidence)", bcp47, confidence)
        return bcp47, confidence

    except (ValueError, RuntimeError) as exc:
        logger.warning("Language detection failed: %s", exc)
        return "en", 0.0


def provider_supports_language(provider: str, language: str) -> bool:
    """Check if a provider supports the given language."""
    supported = PROVIDER_LANGUAGES.get(provider, set())
    # Check exact match or prefix match
    return language in supported or language.split("-")[0] in supported


def get_best_voice_for_language(
    provider: str, language: str, voices: Sequence[object]
) -> str | None:
    """Find the best voice for a language from a provider's voice list.

    Returns the voice ID or None if no match found.
    """
    lang_prefix = language.split("-")[0]
    for voice in voices:
        voice_lang = getattr(voice, "language", "en")
        if voice_lang == language or voice_lang.startswith(lang_prefix):
            return getattr(voice, "id", None)
    return None
