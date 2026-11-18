"""SQLAlchemy models for TTS service."""

from __future__ import annotations

import datetime as _dt
from uuid import uuid7

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""


# ── tenant_id is a real postgres ``uuid`` column ─────────────────────────────
# ``as_uuid=False`` IS LOAD-BEARING. It keeps the Python-side value a ``str``
# (SQLAlchemy's result processor renders the driver's ``uuid.UUID`` back to the
# canonical lowercase dashed text), so every ``tenant_id: str`` annotation,
# every ``TTSJob.tenant_id == tenant_id`` comparison against a request-scoped
# string, and every Redis channel key built as ``f"tenant:{tenant_id}"`` stay
# true without a single call-site change.
#
# ⛔ Do NOT "tidy" this to ``as_uuid=True``. That flips every read to a
# ``uuid.UUID`` object, which compares FALSE against the request's ``str`` and
# is not JSON-serialisable — the shape that has frozen outbox/notification
# paths elsewhere in the fleet.
TenantId = PG_UUID(as_uuid=False)


class TTSJob(Base):
    """TTS generation job tracking."""

    __tablename__ = "tts_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    tenant_id: Mapped[str] = mapped_column(TenantId, nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_length: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    voice: Mapped[str] = mapped_column(String(100), nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    output_format: Mapped[str] = mapped_column(String(10), nullable=False, default="mp3")
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extra_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    caller_item_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Feature 3: Caching
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cache_key: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Feature 5: Routing
    routing_strategy: Mapped[str | None] = mapped_column(String(20), nullable=True)
    selected_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Feature 8: Language detection
    detected_language: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Feature 10: Cost tracking
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Feature 2: Batch concatenation
    concatenated_audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    __table_args__ = (Index("ix_tts_jobs_status_created", "status", "created_at"),)


class TTSUsage(Base):
    """Daily TTS usage aggregation per tenant and provider."""

    __tablename__ = "tts_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(TenantId, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    date: Mapped[_dt.date] = mapped_column(Date, nullable=False, index=True)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_characters: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_audio_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Feature 10: Enhanced usage tracking
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_misses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("tenant_id", "provider", "date", name="uq_usage_tenant_provider_date"),
    )


class TenantQuota(Base):
    """Per-tenant quota/budget configuration with monthly and daily limits."""

    __tablename__ = "tenant_quotas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(TenantId, unique=True, nullable=False, index=True)
    monthly_budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    monthly_character_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_request_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_request_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )


class UsageAlert(Base):
    """Usage alert records triggered when quota thresholds are crossed."""

    __tablename__ = "usage_alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid7()))
    tenant_id: Mapped[str] = mapped_column(TenantId, nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    threshold_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    current_value: Mapped[float] = mapped_column(Float, nullable=False)
    limit_value: Mapped[float] = mapped_column(Float, nullable=False)
    period: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
