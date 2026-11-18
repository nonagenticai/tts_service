"""Unit tests for create_registry function.

Uses real :class:`src.config.Settings` (driven by env vars via ``monkeypatch``)
rather than ``MagicMock``, so any real change to provider settings — like
adding a new provider — surfaces here as a meaningful assertion failure
instead of a silent pass through MagicMock attribute auto-creation.
"""

from __future__ import annotations

import pytest

from src.config import Settings, clear_settings_cache
from src.providers.registry import create_registry

PROVIDER_ENABLE_ENVS = {
    "stub": "STUB_ENABLED",
    "kokoro": "KOKORO_ENABLED",
    "kitten": "KITTEN_ENABLED",
    "openai": "OPENAI_TTS_ENABLED",
    "elevenlabs": "ELEVENLABS_ENABLED",
}


def _build_settings(monkeypatch: pytest.MonkeyPatch, **enabled: bool) -> Settings:
    """Build real Settings with only the requested providers enabled."""
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    for name, env in PROVIDER_ENABLE_ENVS.items():
        monkeypatch.setenv(env, "true" if enabled.get(name, False) else "false")
    clear_settings_cache()
    return Settings()


class TestCreateRegistry:
    def test_create_with_local_providers(self, monkeypatch):
        settings = _build_settings(monkeypatch, kokoro=True, kitten=True)
        registry = create_registry(settings)
        names = registry.provider_names
        assert "kokoro" in names
        assert "kitten" in names
        assert "stub" not in names

    def test_create_with_no_providers(self, monkeypatch):
        settings = _build_settings(monkeypatch)
        registry = create_registry(settings)
        assert registry.provider_names == []

    def test_create_with_openai(self, monkeypatch):
        settings = _build_settings(monkeypatch, openai=True)
        registry = create_registry(settings)
        assert registry.provider_names == ["openai"]

    def test_create_with_elevenlabs(self, monkeypatch):
        settings = _build_settings(monkeypatch, elevenlabs=True)
        registry = create_registry(settings)
        assert registry.provider_names == ["elevenlabs"]

    def test_create_with_stub(self, monkeypatch):
        settings = _build_settings(monkeypatch, stub=True)
        registry = create_registry(settings)
        assert registry.provider_names == ["stub"]

    def test_create_with_all_providers(self, monkeypatch):
        settings = _build_settings(
            monkeypatch,
            stub=True,
            kokoro=True,
            kitten=True,
            openai=True,
            elevenlabs=True,
        )
        registry = create_registry(settings)
        # Stub registers first, then kokoro, kitten, openai, elevenlabs.
        assert registry.provider_names == [
            "stub",
            "kokoro",
            "kitten",
            "openai",
            "elevenlabs",
        ]

    def test_tts_provider_env_overrides_individual_flags(self, monkeypatch):
        # Even with all providers enabled via *_ENABLED env vars, TTS_PROVIDER=stub
        # must force only the stub to be registered.
        for env in PROVIDER_ENABLE_ENVS.values():
            monkeypatch.setenv(env, "true")
        monkeypatch.setenv("TTS_PROVIDER", "stub")
        clear_settings_cache()

        settings = Settings()
        registry = create_registry(settings)
        assert registry.provider_names == ["stub"]
