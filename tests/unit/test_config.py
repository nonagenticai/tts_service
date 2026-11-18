"""Unit tests for configuration."""

import pytest


class TestSettings:
    """Test Settings initialization and validation."""

    def test_default_settings_load(self):
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        settings = Settings()
        assert settings.app_name == "TTS Service"
        assert settings.api.port == 6194
        assert settings.kokoro.enabled is True
        assert settings.kitten.enabled is True
        clear_settings_cache()

    def test_port_8000_rejected(self):
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        with pytest.raises(ValueError, match="8000"):
            Settings(api={"port": 8000})
        clear_settings_cache()

    def test_provider_settings_defaults(self):
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        settings = Settings()
        assert settings.kokoro.default_voice == "af_heart"
        assert settings.kitten.default_voice == "Bella"
        assert settings.kitten.model_id == "KittenML/kitten-tts-mini-0.8"
        assert settings.openai_tts.default_voice == "alloy"
        assert settings.openai_tts.max_text_length == 4096
        assert settings.elevenlabs.default_voice == "rachel"
        assert settings.elevenlabs.max_text_length == 5000
        clear_settings_cache()

    def test_storage_defaults(self):
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        settings = Settings()
        assert settings.storage.backend == "local"
        clear_settings_cache()
