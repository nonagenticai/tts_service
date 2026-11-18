"""Request/response Pydantic models for the TTS API."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Enums
# ============================================================================


class RoutingStrategy(StrEnum):
    """Provider routing strategies."""

    COST = "cost"
    QUALITY = "quality"
    LATENCY = "latency"
    BALANCED = "balanced"


class QualityProfile(StrEnum):
    """Pre-defined quality profiles for audio generation."""

    DEFAULT = "default"
    NARRATION = "narration"


# ============================================================================
# Post-Processing Config (Feature 6)
# ============================================================================


class NormalizationConfig(BaseModel):
    """Loudness normalization settings."""

    enabled: bool = Field(default=True, description="Enable loudness normalization")
    target_lufs: float = Field(
        default=-16.0,
        ge=-70.0,
        le=0.0,
        description="Target loudness in LUFS",
    )
    true_peak: float = Field(
        default=-1.5,
        ge=-10.0,
        le=0.0,
        description="Maximum true peak in dBTP",
    )


class SilenceConfig(BaseModel):
    """Silence insertion settings."""

    leading_ms: int = Field(default=0, ge=0, le=10000, description="Silence to prepend (ms)")
    trailing_ms: int = Field(default=0, ge=0, le=10000, description="Silence to append (ms)")


class FadeConfig(BaseModel):
    """Fade-in/fade-out settings."""

    fade_in_ms: int = Field(default=0, ge=0, le=10000, description="Fade-in duration (ms)")
    fade_out_ms: int = Field(default=0, ge=0, le=10000, description="Fade-out duration (ms)")


class TrimConfig(BaseModel):
    """Silence trimming settings."""

    enabled: bool = Field(default=False, description="Trim leading/trailing silence")
    threshold_db: float = Field(
        default=-40.0,
        ge=-80.0,
        le=-10.0,
        description="Silence threshold in dB",
    )


class SampleRateConfig(BaseModel):
    """Sample rate conversion settings."""

    target_rate: int = Field(
        default=0,
        description="Target sample rate in Hz (0 = keep original)",
    )

    @field_validator("target_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Ensure the target sample rate is one of the supported values."""
        valid_rates = {0, 8000, 16000, 22050, 24000, 44100, 48000, 96000}
        if v not in valid_rates:
            raise ValueError(f"Invalid sample rate: {v}. Must be one of {sorted(valid_rates)}")
        return v


class BitrateConfig(BaseModel):
    """MP3 bitrate control."""

    bitrate: str = Field(
        default="",
        pattern=r"^(|64k|96k|128k|160k|192k|256k|320k)$",
        description="MP3 bitrate (empty = default VBR quality 2)",
    )


class PostProcessingConfig(BaseModel):
    """Post-processing pipeline configuration."""

    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)
    fade: FadeConfig = Field(default_factory=FadeConfig)
    trim: TrimConfig = Field(default_factory=TrimConfig)
    sample_rate: SampleRateConfig = Field(default_factory=SampleRateConfig)
    bitrate: BitrateConfig = Field(default_factory=BitrateConfig)

    def has_any_processing(self) -> bool:
        """Return True if any post-processing step is enabled."""
        return (
            self.normalization.enabled
            or self.silence.leading_ms > 0
            or self.silence.trailing_ms > 0
            or self.fade.fade_in_ms > 0
            or self.fade.fade_out_ms > 0
            or self.trim.enabled
            or self.sample_rate.target_rate > 0
            or bool(self.bitrate.bitrate)
        )

    @classmethod
    def from_profile(cls, profile: QualityProfile) -> PostProcessingConfig:
        """Create a PostProcessingConfig from a named quality profile."""
        if profile == QualityProfile.NARRATION:
            return cls(
                normalization=NormalizationConfig(enabled=True, target_lufs=-16.0, true_peak=-1.5),
                trim=TrimConfig(enabled=True, threshold_db=-40.0),
                fade=FadeConfig(fade_in_ms=30, fade_out_ms=50),
                sample_rate=SampleRateConfig(target_rate=48000),
            )
        return cls()


# ============================================================================
# TTS Request/Response Schemas
# ============================================================================


class TTSGenerateRequest(BaseModel):
    """Request body for TTS generation."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    provider: str = Field(default="auto", description="TTS provider name or 'auto' for routing")
    voice: str | None = Field(default=None, description="Provider-specific voice ID")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier")
    output_format: str = Field(default="mp3", pattern="^(mp3|wav)$", description="Output format")
    model: str | None = Field(default=None, description="Provider-specific model")
    async_mode: bool = Field(default=False, description="If True, return job_id instead of audio")
    language: str | None = Field(
        default=None, description="Language code for auto-detection/routing"
    )
    ssml: bool | None = Field(default=None, description="Input is SSML (None = auto-detect)")
    strategy: RoutingStrategy | None = Field(
        default=None,
        description="Routing strategy when provider='auto'",
    )
    post_processing: PostProcessingConfig | None = Field(
        default=None, description="Audio post-processing pipeline"
    )
    quality_profile: QualityProfile | None = Field(
        default=None, description="Named quality profile (narration, default)"
    )
    provider_params: dict[str, float] | None = Field(
        default=None,
        description=(
            "Provider-specific generation parameters. "
            "Each value is clamped to [-1000, 1000]; up to 32 keys are accepted."
        ),
    )

    @field_validator("provider_params")
    @classmethod
    def _validate_provider_params(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Bound provider_params to prevent DoS via unbounded numeric payloads.

        Limits: at most 32 keys, each key at most 64 chars, each value finite
        and in the closed interval [-1000, 1000].
        """
        if v is None:
            return v
        if len(v) > 32:
            raise ValueError(f"provider_params accepts at most 32 keys (got {len(v)})")
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 64:
                raise ValueError(f"provider_params keys must be strings of ≤64 chars (got {key!r})")
            if not math.isfinite(value):
                raise ValueError(f"provider_params[{key!r}] must be finite (got {value!r})")
            if value < -1000 or value > 1000:
                raise ValueError(
                    f"provider_params[{key!r}] must be in [-1000, 1000] (got {value!r})"
                )
        return v


class TTSStreamRequest(BaseModel):
    """Request body for SSE streaming TTS generation."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    provider: str = Field(default="auto", description="TTS provider name or 'auto' for routing")
    voice: str | None = Field(default=None, description="Provider-specific voice ID")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier")
    output_format: str = Field(
        default="mp3", pattern="^(mp3|wav|pcm)$", description="Output format"
    )
    model: str | None = Field(default=None, description="Provider-specific model")
    language: str | None = Field(default=None, description="Language code")


class TTSBatchItem(BaseModel):
    """Single item in a batch TTS request."""

    id: str = Field(..., description="Caller-assigned item ID")
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")


class TTSBatchRequest(BaseModel):
    """Request body for batch TTS generation (always async)."""

    items: list[TTSBatchItem] = Field(
        ..., min_length=1, max_length=100, description="Items to generate"
    )
    provider: str = Field(default="auto", description="TTS provider name or 'auto' for routing")
    voice: str | None = Field(default=None, description="Provider-specific voice ID")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier")
    output_format: str = Field(default="mp3", pattern="^(mp3|wav)$", description="Output format")
    concatenate: bool = Field(default=False, description="Produce concatenated full audio")
    gap_ms: int = Field(default=0, ge=0, le=5000, description="Silence gap between segments (ms)")
    strategy: str | None = Field(default=None, description="Routing strategy when provider='auto'")
    language: str | None = Field(default=None, description="Language code")
    post_processing: PostProcessingConfig | None = Field(
        default=None, description="Post-processing applied to each audio"
    )


# ============================================================================
# Job Response Schemas
# ============================================================================


class AudioMetricsResponse(BaseModel):
    """Audio quality metrics measured via ffprobe/ffmpeg."""

    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    true_peak_dbtp: float | None = None
    silence_ratio: float | None = None
    duration_seconds: float | None = None


class JobStatusResponse(BaseModel):
    """Response for async job status."""

    job_id: str
    status: str
    provider: str | None = None
    voice: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    audio_url: str | None = None
    presigned_url: str | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    cached: bool = False
    detected_language: str | None = None
    routing_strategy: str | None = None
    estimated_cost_usd: float | None = None
    audio_metrics: AudioMetricsResponse | None = None


class AsyncJobResponse(BaseModel):
    """Response when a job is created in async mode."""

    job_id: str
    status: str = "pending"
    poll_url: str
    ws_url: str = "/api/tts/ws"


class BatchJobResponse(BaseModel):
    """Response for batch job creation."""

    batch_id: str
    job_ids: dict[str, str]
    status: str = "pending"
    poll_url: str
    ws_url: str = "/api/tts/ws"
    concatenated_audio_url: str | None = None
    concatenated_duration_seconds: float | None = None


# ============================================================================
# Provider/Voice Schemas
# ============================================================================


class ProviderInfo(BaseModel):
    """Information about a TTS provider."""

    name: str
    display_name: str
    provider_type: str
    available: bool
    voices_count: int
    default_voice: str
    max_text_length: int
    requires_api_key: bool
    supports_streaming: bool = False
    supports_ssml: bool = False
    cost_per_1k_chars: float = 0.0


class VoiceInfo(BaseModel):
    """Information about a TTS voice."""

    id: str
    display_name: str
    gender: str = "unknown"
    language: str = "en"
    languages: list[str] = Field(default_factory=lambda: ["en"])
    preview_url: str | None = None


# ============================================================================
# Routing Schemas (Feature 5)
# ============================================================================


class RoutingInfo(BaseModel):
    """Information about how a provider was selected."""

    strategy: str
    selected_provider: str
    voice: str
    reason: str
    fallback_chain: list[str]
    estimated_cost_usd: float | None = None
    actual_provider: str | None = None


class CostEstimateItem(BaseModel):
    """Cost estimate for a single provider."""

    provider: str
    estimated_cost_usd: float
    cost_per_1k_chars: float
    character_count: int
    is_free: bool
    available: bool


class CostEstimateResponse(BaseModel):
    """Response for cost estimation endpoint."""

    estimates: list[CostEstimateItem]
    recommended_provider: str
    recommended_strategy: str


# ============================================================================
# Usage & Analytics Schemas (Feature 10)
# ============================================================================


class ProviderUsageSummary(BaseModel):
    """Usage summary for a single provider."""

    provider: str
    request_count: int
    total_characters: int
    total_audio_seconds: float
    cost_usd: float
    error_count: int
    avg_latency_ms: float


class UsageSummaryResponse(BaseModel):
    """Usage summary response."""

    tenant_id: str | None = None
    period_start: str
    period_end: str
    total_requests: int
    total_characters: int
    total_audio_seconds: float
    total_cost_usd: float
    error_count: int
    cache_hit_rate: float = 0.0
    by_provider: list[ProviderUsageSummary]


class CostBreakdownItem(BaseModel):
    """Cost breakdown for a group."""

    group_key: str
    cost_usd: float
    request_count: int
    character_count: int


class CostBreakdownResponse(BaseModel):
    """Cost breakdown response."""

    period_start: str
    period_end: str
    total_cost_usd: float
    breakdown: list[CostBreakdownItem]


class DailyUsagePoint(BaseModel):
    """Single day usage data."""

    date: str
    request_count: int
    total_characters: int
    total_audio_seconds: float
    cost_usd: float


class DashboardDataResponse(BaseModel):
    """Dashboard time-series data."""

    tenant_id: str
    days: int
    usage_over_time: list[DailyUsagePoint]
    provider_distribution: list[ProviderUsageSummary]


class QuotaStatusResponse(BaseModel):
    """Current tenant quota status."""

    tenant_id: str
    monthly_budget_usd: float | None = None
    monthly_budget_used_usd: float = 0.0
    monthly_character_limit: int | None = None
    monthly_characters_used: int = 0
    monthly_request_limit: int | None = None
    monthly_requests_used: int = 0
    daily_request_limit: int | None = None
    daily_requests_used: int = 0
    is_active: bool = True


class QuotaUpdateRequest(BaseModel):
    """Request to update tenant quotas."""

    monthly_budget_usd: float | None = None
    monthly_character_limit: int | None = None
    monthly_request_limit: int | None = None
    daily_request_limit: int | None = None


class UsageAlertResponse(BaseModel):
    """Usage alert record."""

    id: str
    tenant_id: str
    alert_type: str
    threshold_pct: int
    current_value: float
    limit_value: float
    period: str
    message: str
    acknowledged: bool
    created_at: str


class ProviderCostRateResponse(BaseModel):
    """Provider cost rate info."""

    provider: str
    cost_per_1k_chars: float
    provider_type: str
