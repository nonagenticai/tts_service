"""Provider registry for discovering and managing TTS providers."""

from __future__ import annotations

import logging
import threading

from src.config import Settings
from src.providers.base import TTSProvider
from src.providers.elevenlabs_provider import ElevenLabsProvider
from src.providers.kitten_provider import KittenProvider
from src.providers.kokoro_provider import KokoroProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.stub_provider import StubProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Thread-safe registry for TTS providers."""

    def __init__(self) -> None:
        self._providers: dict[str, TTSProvider] = {}
        self._lock = threading.Lock()

    def register(self, provider: TTSProvider) -> None:
        """Register a TTS provider instance by its name.

        Args:
            provider: The TTSProvider instance to register.
        """
        with self._lock:
            self._providers[provider.name] = provider
            logger.info("Registered TTS provider: %s (%s)", provider.name, provider.display_name)

    def get_provider(self, name: str) -> TTSProvider:
        """Look up a registered provider by name.

        Args:
            name: Provider identifier (e.g. "kokoro", "openai").

        Returns:
            The registered TTSProvider instance.

        Raises:
            KeyError: If no provider is registered with the given name.
        """
        with self._lock:
            provider = self._providers.get(name)
        if provider is None:
            available = list(self._providers.keys())
            raise KeyError(f"TTS provider '{name}' is not registered. Available: {available}")
        return provider

    def list_providers(self) -> list[TTSProvider]:
        """Return all registered provider instances."""
        with self._lock:
            return list(self._providers.values())

    async def close_all(self) -> None:
        """Close all registered providers, releasing their resources."""
        with self._lock:
            providers = list(self._providers.values())
        for provider in providers:
            try:
                await provider.close()
            except ImportError, OSError, RuntimeError:
                logger.warning("Error closing provider %s", provider.name, exc_info=True)

    @property
    def provider_names(self) -> list[str]:
        with self._lock:
            return list(self._providers.keys())


def create_registry(settings: Settings) -> ProviderRegistry:
    """Create and populate a provider registry from settings."""
    registry = ProviderRegistry()

    if settings.stub.enabled:
        registry.register(StubProvider(default_voice=settings.stub.default_voice))

    if settings.kokoro.enabled:
        registry.register(KokoroProvider(default_voice=settings.kokoro.default_voice))

    if settings.kitten.enabled:
        registry.register(
            KittenProvider(
                default_voice=settings.kitten.default_voice,
                default_model_id=settings.kitten.model_id,
            )
        )

    if settings.openai_tts.enabled:
        registry.register(
            OpenAIProvider(
                api_key=settings.openai_tts.api_key,
                default_voice=settings.openai_tts.default_voice,
                model=settings.openai_tts.model,
                streaming_chunk_size=settings.openai_tts.streaming_chunk_size,
            )
        )

    if settings.elevenlabs.enabled:
        registry.register(
            ElevenLabsProvider(
                api_key=settings.elevenlabs.api_key,
                default_voice=settings.elevenlabs.default_voice,
                model_id=settings.elevenlabs.model_id,
                stability=settings.elevenlabs.stability,
                similarity_boost=settings.elevenlabs.similarity_boost,
                streaming_chunk_size=settings.elevenlabs.streaming_chunk_size,
            )
        )

    logger.info("Provider registry initialized with %d providers", len(registry.provider_names))
    return registry
