"""
Configuration management for tts_service using Pydantic Settings.

Hierarchical settings with provider-specific sub-configs.
Follows skill_builder config pattern (minus MCP components).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import APISettings as _BaseAPISettings
from w2v_common.config import AuthSettings, DatabaseSettings, LoggingSettings, RedisSettings
from w2v_common.config import CacheSettings as _BaseCacheSettings
from w2v_common.config import StorageSettings as _BaseStorageSettings

__all__ = [
    "AuthSettings",
    "DatabaseSettings",
    "LoggingSettings",
    "RedisSettings",
]


class APISettings(_BaseAPISettings):
    """TTS service API settings (port 6194)."""

    port: int = Field(default=6194, description="API server port")


class StorageSettings(_BaseStorageSettings):
    """Audio file storage configuration."""

    local_path: str = Field(default="/app/data/audio", description="Local storage path for audio")
    minio_bucket: str = Field(default="tts-audio", description="MinIO bucket name")
    minio_external_endpoint: str | None = Field(
        default=None,
        description="External MinIO endpoint for presigned URLs (replaces internal host when set)",
    )


class CacheSettings(_BaseCacheSettings):
    """Audio generation caching configuration."""

    default_ttl: int = Field(default=86400, description="Cache TTL in seconds (24h)")
    # v2: the pre-fix writer stored a FABRICATED ``duration: 0.0`` whenever
    # ffprobe failed. Omitting it on write does nothing about the entries
    # already in Redis, which the (correct) new reader would serve back as a
    # measurement for up to ``default_ttl``. Moving the namespace makes those
    # entries unreachable; they expire on their own TTL. Do not revert without
    # a migration.
    key_prefix: str = Field(default="tts:gen:v2:", description="Redis cache key prefix")


class RoutingSettings(BaseSettings):
    """Multi-provider routing configuration."""

    enabled: bool = Field(default=True, description="Enable auto-routing")
    default_strategy: str = Field(default="balanced", description="Default routing strategy")
    fallback_enabled: bool = Field(default=True, description="Enable provider fallback")
    max_fallback_attempts: int = Field(default=2, description="Max fallback attempts")
    quality_scores: dict[str, float] = Field(
        default={"kokoro": 0.80, "kitten": 0.75, "openai": 0.90, "elevenlabs": 0.95},
    )
    cost_per_1k_chars: dict[str, float] = Field(
        default={"kokoro": 0.0, "kitten": 0.0, "openai": 0.015, "elevenlabs": 0.030},
    )

    model_config = SettingsConfigDict(
        env_prefix="ROUTING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class WebSocketSettings(BaseSettings):
    """WebSocket configuration settings."""

    max_connections_per_tenant: int = Field(default=10)
    heartbeat_interval: int = Field(default=30, description="Server ping interval (seconds)")
    pubsub_channel: str = Field(default="tts:job_events")
    event_queue_size: int = Field(default=100, description="Per-connection event queue size")

    model_config = SettingsConfigDict(
        env_prefix="WS_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class QuotaSettings(BaseSettings):
    """Quota enforcement settings."""

    enforcement_enabled: bool = Field(default=False, description="Enable quota enforcement")
    default_monthly_budget_usd: float | None = Field(default=None)
    default_monthly_character_limit: int | None = Field(default=None)
    redis_quota_cache_ttl: int = Field(default=60, description="Quota cache TTL (seconds)")

    model_config = SettingsConfigDict(
        env_prefix="QUOTA_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class _ProviderToggle(Protocol):
    """Structural type for provider sub-settings that have an ``enabled`` flag."""

    enabled: bool


class StubSettings(BaseSettings):
    """Stub TTS provider settings (silent offline output for tests/CI)."""

    enabled: bool = Field(
        default=False,
        description=(
            "Enable the stub provider (silent audio, no external deps). "
            "Set STUB_ENABLED=true to register the stub alongside other providers."
        ),
    )
    default_voice: str = Field(default="stub_default", description="Default stub voice ID")
    sample_rate: int = Field(default=24000, description="Audio sample rate (matches Kokoro)")
    cost_per_1k_chars: float = Field(default=0.0, description="Cost per 1K chars (free/local)")

    model_config = SettingsConfigDict(
        env_prefix="STUB_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class KokoroSettings(BaseSettings):
    """Kokoro TTS provider settings."""

    enabled: bool = Field(default=True, description="Enable Kokoro provider")
    default_voice: str = Field(default="af_heart", description="Default Kokoro voice")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    cost_per_1k_chars: float = Field(default=0.0, description="Cost per 1K chars (free/local)")

    model_config = SettingsConfigDict(
        env_prefix="KOKORO_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class KittenSettings(BaseSettings):
    """KittenTTS provider settings."""

    enabled: bool = Field(default=True, description="Enable KittenTTS provider")
    model_id: str = Field(
        default="KittenML/kitten-tts-mini-0.8",
        description="Default KittenTTS model ID",
    )
    default_voice: str = Field(default="Bella", description="Default KittenTTS voice")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    cost_per_1k_chars: float = Field(default=0.0, description="Cost per 1K chars (free/local)")

    model_config = SettingsConfigDict(
        env_prefix="KITTEN_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class OpenAITTSSettings(BaseSettings):
    """OpenAI TTS provider settings."""

    enabled: bool = Field(default=True, description="Enable OpenAI TTS provider")
    api_key: str = Field(default="", description="OpenAI API key for TTS")
    default_voice: str = Field(default="alloy", description="Default OpenAI TTS voice")
    model: str = Field(default="tts-1-hd", description="OpenAI TTS model")
    max_text_length: int = Field(default=4096, description="Maximum text length")
    cost_per_1k_chars: float = Field(default=0.030, description="Cost per 1K chars (tts-1-hd)")
    streaming_chunk_size: int = Field(
        default=65536, description="Streaming response chunk size in bytes"
    )

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_TTS_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ElevenLabsSettings(BaseSettings):
    """ElevenLabs TTS provider settings."""

    enabled: bool = Field(default=True, description="Enable ElevenLabs provider")
    api_key: str = Field(default="", description="ElevenLabs API key")
    default_voice: str = Field(default="rachel", description="Default ElevenLabs voice")
    model_id: str = Field(default="eleven_multilingual_v2", description="ElevenLabs model ID")
    max_text_length: int = Field(default=5000, description="Maximum text length")
    stability: float = Field(default=0.5, description="Voice stability setting")
    similarity_boost: float = Field(default=0.75, description="Voice similarity boost")
    cost_per_1k_chars: float = Field(default=0.030, description="Cost per 1K chars")
    streaming_chunk_size: int = Field(
        default=65536, description="Streaming response chunk size in bytes"
    )

    model_config = SettingsConfigDict(
        env_prefix="ELEVENLABS_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ConcurrencySettings(BaseSettings):
    """Concurrency control settings."""

    max_concurrent_generations: int = Field(
        default=10, description="Global semaphore for generation calls"
    )
    max_concurrent_batch_jobs: int = Field(default=20, description="Semaphore for batch processing")
    openai_max_concurrent: int = Field(default=5, description="Per-provider limit for OpenAI")
    elevenlabs_max_concurrent: int = Field(
        default=5, description="Per-provider limit for ElevenLabs"
    )
    kokoro_max_concurrent: int = Field(
        default=3, description="Per-provider limit for local Kokoro (CPU/GPU bound)"
    )
    kitten_max_concurrent: int = Field(
        default=3, description="Per-provider limit for local KittenTTS"
    )

    model_config = SettingsConfigDict(
        env_prefix="CONCURRENCY_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class MeteringSettings(BaseSettings):
    """Cost-event metering configuration.

    Emits per-synthesis cost events to a Redis stream consumed by
    metering_service. The Redis at ``redis_url`` is SEPARATE from the
    service's regular cache/notification Redis — see EMITTER_GUIDE.md.
    Leave ``redis_url`` empty to disable emission silently (the service
    boots and serves normally; events are just dropped).

    The cost-bus stream name is resolved centrally by
    ``w2v_common.metering.resolve_stream`` (``METERING_STREAM`` env var,
    falling back to the canonical ``billing:costs``) — there is
    no per-service ``stream`` field here any more.
    """

    redis_url: str = Field(default="", description="Metering Redis URL (separate from cache Redis)")

    model_config = SettingsConfigDict(
        env_prefix="METERING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    app_name: str = Field(default="TTS Service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development", description="Environment (development, staging, production)"
    )
    tts_provider: str | None = Field(
        default=None,
        description=(
            "Optional single-provider override. When set (e.g. TTS_PROVIDER=stub), "
            "only the named provider is registered and all other providers are "
            "force-disabled. Used by the offline-pipeline / CI smoke path."
        ),
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
    stub: StubSettings = Field(default_factory=StubSettings)
    kokoro: KokoroSettings = Field(default_factory=KokoroSettings)
    kitten: KittenSettings = Field(default_factory=KittenSettings)
    openai_tts: OpenAITTSSettings = Field(default_factory=OpenAITTSSettings)
    elevenlabs: ElevenLabsSettings = Field(default_factory=ElevenLabsSettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metering: MeteringSettings = Field(default_factory=MeteringSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._apply_tts_provider_override()
        self._validate_configuration()

    def _apply_tts_provider_override(self) -> None:
        """If ``TTS_PROVIDER`` is set, enable only that provider."""
        if self.tts_provider is None:
            return
        target = self.tts_provider.strip().lower()
        if not target:
            return
        known: dict[str, _ProviderToggle] = {
            "stub": self.stub,
            "kokoro": self.kokoro,
            "kitten": self.kitten,
            "openai": self.openai_tts,
            "elevenlabs": self.elevenlabs,
        }
        if target not in known:
            raise ValueError(
                f"TTS_PROVIDER={self.tts_provider!r} is not a known provider. "
                f"Valid: {sorted(known)}"
            )
        for name, sub in known.items():
            sub.enabled = name == target

    def _validate_configuration(self) -> None:
        if self.api.port == 8000:
            raise ValueError(
                "API port is set to 8000, which is not allowed. "
                "Please use a different port (default: 6194)."
            )

    def get_database_url(self) -> str:
        """Return the configured database connection URL."""
        return self.database.url

    def get_redis_url(self) -> str:
        """Return the configured Redis connection URL."""
        return self.redis.url

    def is_production(self) -> bool:
        """Return True if the environment is set to 'production'."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return True if the environment is set to 'development'."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing)."""
    get_settings.cache_clear()
