"""TTS job lifecycle management."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TTSJob, TTSUsage

logger = logging.getLogger(__name__)

# The states a job never leaves. The service has no 'cancelled' status (verified
# by grep over src/); the full vocabulary is pending -> processing -> one of these.
# Kept as an ordered tuple, not a set: a set has no stable iteration order, which
# would churn the rendered SQL between processes.
TERMINAL_JOB_STATUSES: tuple[str, ...] = ("completed", "failed")


class JobManager:
    """Manages TTS job lifecycle."""

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        tenant_id: str,
        text: str,
        provider: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "mp3",
        model: str | None = None,
        batch_id: str | None = None,
        caller_item_id: str | None = None,
        extra_params: dict | None = None,
        routing_strategy: str | None = None,
    ) -> TTSJob:
        """Create a new TTS job record in the database.

        Args:
            db: Async database session.
            tenant_id: Tenant identifier.
            text: Text to synthesize.
            provider: Resolved provider name.
            voice: Resolved voice ID.
            speed: Playback speed multiplier.
            output_format: Audio format.
            model: Optional model override.
            batch_id: Optional batch group identifier.
            caller_item_id: Caller-assigned item ID.
            extra_params: Additional provider-specific parameters.
            routing_strategy: Strategy used for provider selection.

        Returns:
            The newly created TTSJob instance.
        """
        job = TTSJob(
            job_id=str(uuid.uuid7()),
            batch_id=batch_id,
            tenant_id=tenant_id,
            text=text,
            text_length=len(text),
            provider=provider,
            voice=voice,
            speed=speed,
            output_format=output_format,
            model=model,
            caller_item_id=caller_item_id,
            extra_params=extra_params,
            routing_strategy=routing_strategy,
            status="pending",
        )
        db.add(job)
        await db.flush()
        logger.info("Created TTS job %s (provider=%s, voice=%s)", job.job_id, provider, voice)
        return job

    @staticmethod
    async def get_job(db: AsyncSession, job_id: str, *, tenant_id: str) -> TTSJob | None:
        """Retrieve a job owned by ``tenant_id``, or ``None`` if there is no such job.

        ``tenant_id`` is keyword-only and REQUIRED — deliberately not defaulted
        and with no unscoped sibling to fall back on. This service has **no
        RLS** (unlike most of the fleet), so this predicate is the ONLY thing
        standing between a guessed/leaked ``job_id`` and another tenant's row.
        Every caller therefore has to produce a tenant, and a caller that has
        none has no business reading jobs.

        Returning ``None`` for a job that exists but belongs to someone else is
        intentional and indistinguishable from "no such job": the HTTP layer
        turns both into 404, so a job id is never confirmed to exist. A 403
        would be an existence oracle.
        """
        result = await db.execute(
            select(TTSJob).where(TTSJob.job_id == job_id, TTSJob.tenant_id == tenant_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_batch_jobs(db: AsyncSession, batch_id: str, *, tenant_id: str) -> list[TTSJob]:
        """Retrieve the jobs of ``batch_id`` owned by ``tenant_id``, ordered by ID.

        Keyword-only, required ``tenant_id`` for the same reason as
        :meth:`get_job` — a foreign batch reads as an empty list, which the HTTP
        layer renders as 404.
        """
        result = await db.execute(
            select(TTSJob)
            .where(TTSJob.batch_id == batch_id, TTSJob.tenant_id == tenant_id)
            .order_by(TTSJob.id)
        )
        return list(result.scalars().all())

    @staticmethod
    async def mark_processing(db: AsyncSession, job_id: str) -> None:
        """Transition a job to 'processing' status with a started_at timestamp."""
        await db.execute(
            update(TTSJob)
            .where(TTSJob.job_id == job_id)
            .values(status="processing", started_at=datetime.now(UTC))
        )

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        audio_path: str,
        audio_size_bytes: int,
        audio_duration_seconds: float | None,
        cached: bool = False,
        estimated_cost_usd: float | None = None,
        latency_ms: float | None = None,
        detected_language: str | None = None,
    ) -> None:
        """Transition a job to 'completed' with audio metadata."""
        values: dict = {
            "status": "completed",
            "audio_path": audio_path,
            "audio_size_bytes": audio_size_bytes,
            "audio_duration_seconds": audio_duration_seconds,
            "completed_at": datetime.now(UTC),
            "cached": cached,
        }
        if estimated_cost_usd is not None:
            values["estimated_cost_usd"] = estimated_cost_usd
        if latency_ms is not None:
            values["latency_ms"] = latency_ms
        if detected_language is not None:
            values["detected_language"] = detected_language
        await db.execute(update(TTSJob).where(TTSJob.job_id == job_id).values(**values))

    @staticmethod
    async def mark_failed(db: AsyncSession, job_id: str, error_message: str) -> bool:
        """Transition a NON-TERMINAL job to 'failed'. Returns True if it moved the row.

        The ``status NOT IN (...)`` predicate is load-bearing, not defensive
        decoration. This used to be an unconditional UPDATE, which made two
        things possible:

        * **Demoting a completed, billed job.** ``process_job`` commits the
          completion and only then emits the cost event; anything raising after
          that commit lands in the caller's ``except`` arm in ``api/tts.py``,
          which calls straight through to here and overwrote a durably-completed
          row as ``failed``. Widening that arm to catch ``RedisError`` and
          ``SQLAlchemyError`` (the point of this change) widened that window too,
          so the guard has to come with it.
        * **Clobbering the real error.** A second, generic ``mark_failed`` would
          overwrite the first, specific one — the message an operator actually
          needs.

        It is ONE conditional statement rather than a read-then-write so Postgres
        re-evaluates the predicate against the row it locks; a SELECT-then-UPDATE
        pair is a TOCTOU that a concurrent ``mark_completed`` slips through.
        """
        result = cast(
            "CursorResult[Any]",
            await db.execute(
                update(TTSJob)
                .where(
                    TTSJob.job_id == job_id,
                    TTSJob.status.not_in(TERMINAL_JOB_STATUSES),
                )
                .values(
                    status="failed",
                    error_message=error_message,
                    completed_at=datetime.now(UTC),
                )
            ),
        )
        return bool(result.rowcount)

    @staticmethod
    async def set_batch_concat_path(
        db: AsyncSession, batch_id: str, concat_path: str, *, tenant_id: str
    ) -> None:
        """Store the concatenated audio path on the tenant's jobs in a batch.

        Scoped for the same no-RLS reason as the reads: this is a blind
        ``UPDATE ... WHERE batch_id = ...``, so without the tenant predicate a
        caller-influenced ``batch_id`` would stamp a storage key onto rows it
        does not own — and ``download_batch_audio`` serves whatever key it finds
        there.
        """
        await db.execute(
            update(TTSJob)
            .where(TTSJob.batch_id == batch_id, TTSJob.tenant_id == tenant_id)
            .values(concatenated_audio_path=concat_path)
        )

    @staticmethod
    async def record_usage(
        db: AsyncSession,
        *,
        tenant_id: str,
        provider: str,
        characters: int,
        audio_seconds: float,
        is_error: bool = False,
        estimated_cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        """Record usage for daily aggregation (upsert).

        A FAILED synthesis (``is_error=True``) is recorded as an error ONLY: it
        does not accrue billable ``request_count`` / ``total_characters`` /
        ``total_audio_seconds`` / ``estimated_cost_usd``, nor cache/latency
        stats. ``check_quota`` enforces the monthly character/request/budget caps
        by summing exactly those billable columns, so a provider 5xx storm must
        not consume characters the tenant never received (that would 429-lock a
        tenant for failed work). ``tts_usage`` is a daily aggregate keyed on
        (tenant, provider, date) with no per-request ``is_error`` row to filter,
        so the exclusion is made here at write time rather than in the query.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        # On error, only ``error_count`` moves; every billable/analytic delta is 0.
        ok = not is_error
        req_delta = 1 if ok else 0
        char_delta = characters if ok else 0
        audio_delta = audio_seconds if ok else 0.0
        cost_delta = estimated_cost_usd if ok else 0.0
        hit_delta = 1 if (ok and cache_hit) else 0
        miss_delta = 1 if (ok and not cache_hit) else 0

        today = datetime.now(UTC).date()
        stmt = pg_insert(TTSUsage).values(
            tenant_id=tenant_id,
            provider=provider,
            date=today,
            request_count=req_delta,
            total_characters=char_delta,
            total_audio_seconds=audio_delta,
            error_count=0 if ok else 1,
            estimated_cost_usd=cost_delta,
            avg_latency_ms=latency_ms if ok else 0.0,
            cache_hits=hit_delta,
            cache_misses=miss_delta,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_usage_tenant_provider_date",
            set_={
                "request_count": TTSUsage.request_count + req_delta,
                "total_characters": TTSUsage.total_characters + char_delta,
                "total_audio_seconds": TTSUsage.total_audio_seconds + audio_delta,
                "error_count": TTSUsage.error_count + (0 if ok else 1),
                "estimated_cost_usd": TTSUsage.estimated_cost_usd + cost_delta,
                # Latency is a running mean over BILLED requests only; an error
                # contributes no sample, so leave the mean untouched on failure.
                "avg_latency_ms": (
                    (TTSUsage.avg_latency_ms * TTSUsage.request_count + latency_ms)
                    / (TTSUsage.request_count + 1)
                    if ok
                    else TTSUsage.avg_latency_ms
                ),
                "cache_hits": TTSUsage.cache_hits + hit_delta,
                "cache_misses": TTSUsage.cache_misses + miss_delta,
            },
        )
        await db.execute(stmt)
