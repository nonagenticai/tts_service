"""Unit tests for the language detector service."""

from unittest.mock import MagicMock, patch

from src.services.language_detector import (
    _LINGUA_TO_BCP47,
    KOKORO_LANG_MAP,
    PROVIDER_LANGUAGES,
    detect_language,
    get_best_voice_for_language,
    provider_supports_language,
)


class TestLinguaToBCP47Mapping:
    def test_english(self):
        assert _LINGUA_TO_BCP47["ENGLISH"] == "en"

    def test_spanish(self):
        assert _LINGUA_TO_BCP47["SPANISH"] == "es"

    def test_japanese(self):
        assert _LINGUA_TO_BCP47["JAPANESE"] == "ja"

    def test_chinese(self):
        assert _LINGUA_TO_BCP47["CHINESE"] == "zh"


class TestProviderLanguages:
    def test_kokoro_supports_en(self):
        assert "en" in PROVIDER_LANGUAGES["kokoro"]

    def test_kitten_english_only(self):
        assert PROVIDER_LANGUAGES["kitten"] == {"en"}

    def test_openai_supports_many_languages(self):
        assert len(PROVIDER_LANGUAGES["openai"]) > 10
        assert "en" in PROVIDER_LANGUAGES["openai"]
        assert "fr" in PROVIDER_LANGUAGES["openai"]

    def test_elevenlabs_supports_many_languages(self):
        assert len(PROVIDER_LANGUAGES["elevenlabs"]) > 5


class TestKokoroLangMap:
    def test_english_maps_to_a(self):
        assert KOKORO_LANG_MAP["en"] == "a"

    def test_japanese_maps_to_j(self):
        assert KOKORO_LANG_MAP["ja"] == "j"

    def test_chinese_maps_to_z(self):
        assert KOKORO_LANG_MAP["zh"] == "z"


class TestDetectLanguage:
    @patch("src.services.language_detector._get_detector")
    def test_detector_not_available(self, mock_get_detector):
        mock_get_detector.return_value = None
        lang, confidence = detect_language("Hello world")
        assert lang == "en"
        assert confidence == 0.0

    @patch("src.services.language_detector._get_detector")
    def test_detection_high_confidence(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "ENGLISH"
        mock_top.value = 0.95
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        lang, confidence = detect_language("Hello world")
        assert lang == "en"
        assert confidence == 0.95

    @patch("src.services.language_detector._get_detector")
    def test_detection_low_confidence(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "FRENCH"
        mock_top.value = 0.3
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        lang, confidence = detect_language("some text", confidence_threshold=0.8)
        # Below threshold, falls back to "en"
        assert lang == "en"
        assert confidence == 0.3

    @patch("src.services.language_detector._get_detector")
    def test_detection_empty_results(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_detector.compute_language_confidence_values.return_value = []
        mock_get_detector.return_value = mock_detector

        lang, confidence = detect_language("Hello")
        assert lang == "en"
        assert confidence == 0.0

    @patch("src.services.language_detector._get_detector")
    def test_detection_exception(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_detector.compute_language_confidence_values.side_effect = RuntimeError("error")
        mock_get_detector.return_value = mock_detector

        lang, confidence = detect_language("Hello")
        assert lang == "en"
        assert confidence == 0.0

    @patch("src.services.language_detector._get_detector")
    def test_detection_unknown_language_defaults_to_en(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_top = MagicMock()
        mock_top.language.name = "UNKNOWN_LANG"
        mock_top.value = 0.95
        mock_detector.compute_language_confidence_values.return_value = [mock_top]
        mock_get_detector.return_value = mock_detector

        lang, confidence = detect_language("text")
        assert lang == "en"  # fallback for unknown
        assert confidence == 0.95


class TestProviderSupportsLanguage:
    def test_exact_match(self):
        assert provider_supports_language("kokoro", "en") is True
        assert provider_supports_language("kokoro", "ja") is True

    def test_prefix_match(self):
        assert provider_supports_language("kokoro", "en-US") is True

    def test_unsupported_language(self):
        assert provider_supports_language("kitten", "fr") is False

    def test_unknown_provider(self):
        assert provider_supports_language("nonexistent", "en") is False


class TestGetBestVoiceForLanguage:
    def test_finds_matching_voice(self):
        voices = [
            MagicMock(id="v1", language="en"),
            MagicMock(id="v2", language="fr"),
        ]
        result = get_best_voice_for_language("openai", "fr", voices)
        assert result == "v2"

    def test_prefix_match(self):
        voices = [
            MagicMock(id="v1", language="en-US"),
        ]
        result = get_best_voice_for_language("openai", "en", voices)
        assert result == "v1"

    def test_no_matching_voice(self):
        voices = [
            MagicMock(id="v1", language="en"),
        ]
        result = get_best_voice_for_language("openai", "ja", voices)
        assert result is None

    def test_empty_voices(self):
        result = get_best_voice_for_language("openai", "en", [])
        assert result is None

    def test_voice_without_language_attr(self):
        voice = MagicMock(spec=[])  # No 'language' attribute
        result = get_best_voice_for_language("openai", "en", [voice])
        # getattr(voice, "language", "en") defaults to "en", which matches "en"
        # but getattr(voice, "id", None) also defaults to None since spec=[]
        # So function returns None
        assert result is None
