"""TTS generation API endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess  # nosec B404 -- imported for TimeoutExpired only, no process is spawned here
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import PurePosixPath

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import LICENSE_GATE

from src.api.schemas import (
    AsyncJobResponse,
    BatchJobResponse,
    JobStatusResponse,
    ProviderInfo,
    TTSBatchRequest,
    TTSGenerateRequest,
    TTSStreamRequest,
    VoiceInfo,
)
from src.database.session import get_db, get_db_session
from src.providers.base import GenerationResult
from src.providers.registry import ProviderRegistry
from src.services.job_manager import JobManager
from src.services.metering import InvalidTenantIdError, MissingTenantError, require_tenant
from src.services.quota_service import QuotaExceededError, check_quota
from src.services.storage_service import StorageService
from src.services.tts_service import TTSService, _estimate_cost

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()

# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted router-wide.
# Until now this service authenticated the licence token for tenant IDENTITY
# (AuthMiddleware answers "who are you") and never asked "may you" — a tenant
# whose licence is expired or revoked still carries a syntactically valid token
# and was served. LICENSE_GATE closes that: none/expired/revoked -> 402
# license_<state>, an unestablishable verdict -> 503 license_unavailable +
# Retry-After, active/expiring_soon -> pass.
#
# Every route under /api/tts here is tenant-scoped generation/job/streaming
# work, so gating the whole router is correct.
#
# DELIBERATELY NOT GATED, each for a stated reason:
#   * src.api.health (/health, /health/live, /health/ready, /healthz, /readyz),
#     mounted separately in asgi.py. A 402/503 on a probe CrashLoops the pod
#     and an unlicensed tenant cannot be billed by killing our own deployment.
#     The gate also refuses these by path (is_license_exempt_path) — belt and
#     braces, not a single point of failure.
#   * /metrics and /docs|/openapi.json, added by w2v_common's create_app: a
#     Prometheus scrape and the schema carry no tenant and do no tenant work.
#
# GATED ELSEWHERE, not here: src.api.ws (the /api/tts/ws WebSocket router).
# ``LICENSE_GATE``'s ``require_license`` is typed for a starlette ``Request``
# and cannot be mounted on a websocket route, which is why that surface was
# left ungated when this comment was first written. w2v_common 0.76.0 added the
# WS twin (``require_license_ws``), and ``websocket_job_notifications`` now
# calls it directly — it has to be called from INSIDE the handler, after the
# handshake resolves and stamps the tenant, because no middleware stamps
# identity on a websocket scope. See the comment at that call site.
#
# DORMANT ON MERGE: require_license is a no-op unless W2V_LICENSE_GATE_ENABLED
# is true, read per-call from os.environ. It defaults to False, so this change
# alters no behaviour until gitops arms the flag. Preserve that default.
router = APIRouter(prefix="/api/tts", tags=["tts"], dependencies=list(LICENSE_GATE))


def _get_tenant_id(request: Request) -> str:
    """Return the caller's REAL forwarded tenant, or 401 — never a sentinel.

    PRODUCT-OWNER RULING: a ``tenant_id`` may never hold a sentinel. All traffic
    carries a real registered tenant forwarded from the requestor; an absent one
    is an ERROR, not a substituted default. This helper used to return the
    nil-UUID ``DEFAULT_TENANT_ID`` — the most dangerous spelling, because it is a
    VALID uuid that passes every cast and dirty-row check, and billing then
    buckets it as unattributed and NEVER CHARGES it. Real usage, silently
    unbilled. Do not re-add a fallback here.

    Two outcomes, two status codes — they are NOT the same failure:

    * **401 on ABSENT.** ``request.state.tenant_id`` is populated exclusively by
      ``AuthMiddleware`` from the ``X-Tenant-Id: Bearer <license_token>``
      handshake, so its absence means authentication resolved no identity, not
      that the payload was malformed. Same choice as ``slide_service``, and
      consistent with this repo's own WS path, which already closes 1008 on the
      identical condition.
    * **400 on PRESENT-BUT-NOT-A-UUID.** Since revision ``005_tenant_id_uuid``
      the tenant is a postgres ``uuid`` COLUMN here too, so an unparseable value
      is a bad argument — exactly ``video_compiler_service``'s reasoning, which
      reached the uuid column first. Catching it here is what keeps a legacy
      slug from becoming a psycopg ``DataError`` 500 midway through a job write.

    The returned value is canonical lowercase-dashed uuid text, so it compares
    equal to what postgres renders back out of the column.
    """
    try:
        return require_tenant(getattr(request.state, "tenant_id", None))
    except MissingTenantError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except InvalidTenantIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_tts_service(request: Request) -> TTSService:
    """Get TTSService from app state."""
    return request.app.state.tts_service


def _get_registry(request: Request) -> ProviderRegistry:
    """Get ProviderRegistry from app state."""
    return request.app.state.provider_registry


def _get_storage(request: Request) -> StorageService:
    """Get StorageService from app state."""
    return request.app.state.storage_service


async def _enforce_stream_quota(db: AsyncSession, tenant_id: str, text: str, provider: str) -> None:
    """Enforce tenant quota for a streaming request, translating to HTTP 429.

    Must run in the route body BEFORE the StreamingResponse is returned: a
    generator raising mid-stream cannot un-send a 200 to deliver a 429.

    ``provider`` is used to project the request's cost so the monthly budget cap
    cannot be overshot by a single large streamed request (``auto`` estimates to
    $0 pre-routing — the budget bug it fixes is for explicitly-priced providers).
    """
    try:
        await check_quota(
            db, tenant_id, len(text), projected_cost=_estimate_cost(provider, len(text))
        )
    except QuotaExceededError as e:
        logger.warning("Quota exceeded for stream tenant=%s: %s", tenant_id, e)
        raise HTTPException(status_code=429, detail=str(e)) from e


# ============================================================================
# Generation Endpoints
# ============================================================================


def _validate_provider_or_404(registry: ProviderRegistry, provider: str) -> None:
    """Raise HTTP 404 if a non-auto provider is not registered."""
    if provider == "auto":
        return
    try:
        registry.get_provider(provider)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def _resolve_quality_profile(
    body: TTSGenerateRequest,
) -> tuple[object | None, dict[str, float] | None]:
    """Translate a quality_profile preset into post-processing + provider params."""
    post_processing = body.post_processing
    provider_params = body.provider_params
    if not (body.quality_profile and post_processing is None):
        return post_processing, provider_params

    from src.api.schemas import PostProcessingConfig, QualityProfile

    post_processing = PostProcessingConfig.from_profile(body.quality_profile)
    if body.quality_profile == QualityProfile.NARRATION and body.provider == "elevenlabs":
        if provider_params is None:
            provider_params = {}
        provider_params.setdefault("stability", 0.7)
        provider_params.setdefault("similarity_boost", 0.85)
    return post_processing, provider_params


async def _dispatch_async_job(
    tts: TTSService,
    db: AsyncSession,
    body: TTSGenerateRequest,
    tenant_id: str,
    post_processing: object | None,
) -> AsyncJobResponse:
    """Create an async job, commit, and schedule the background processor."""
    job_id = await tts.generate_async(
        db,
        tenant_id=tenant_id,
        text=body.text,
        provider=body.provider,
        voice=body.voice,
        speed=body.speed,
        output_format=body.output_format,
        model=body.model,
        strategy=body.strategy,
    )
    await db.commit()
    task = asyncio.create_task(_process_job_background(tts, job_id, tenant_id, post_processing))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    logger.info("Async TTS job created: job_id=%s tenant=%s", job_id, tenant_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/tts/jobs/{job_id}",
    )


async def _run_sync_generate(
    tts: TTSService,
    db: AsyncSession,
    body: TTSGenerateRequest,
    tenant_id: str,
    post_processing: object | None,
    provider_params: dict[str, float] | None,
    no_cache: bool,
) -> tuple[GenerationResult, dict]:
    """Invoke generate_sync and translate provider/quota/value errors to HTTPExceptions."""
    try:
        return await tts.generate_sync(
            text=body.text,
            provider=body.provider,
            voice=body.voice,
            speed=body.speed,
            output_format=body.output_format,
            model=body.model,
            tenant_id=tenant_id,
            language=body.language,
            ssml=body.ssml,
            strategy=body.strategy,
            post_processing=post_processing,
            provider_params=provider_params,
            no_cache=no_cache,
            db=db,
        )
    except QuotaExceededError as e:
        logger.warning("Quota exceeded for tenant=%s: %s", tenant_id, e)
        raise HTTPException(status_code=429, detail=str(e)) from e
    except RuntimeError as e:
        logger.error("Provider error during generate: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        logger.warning("Validation error during generate: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e


_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "pcm": "audio/L16",
}


def _content_type_for(audio_format: str) -> str:
    """Content-Type for the format of the bytes we are about to send.

    Always call this with the format that was actually PRODUCED, never with the
    format the caller requested — a provider may not honour the request (e.g.
    ElevenLabs only emits mp3), and serving mp3 bytes labelled ``audio/wav``
    breaks every downstream decoder.
    """
    return _AUDIO_CONTENT_TYPES.get(audio_format.lower(), f"audio/{audio_format.lower()}")


def _stored_format(audio_path: str | None, requested_format: str) -> str:
    """Format of the bytes actually in storage, read off the storage key.

    ``StorageService.store`` builds the key as ``audio/{job_id}.{result.format}``
    from the *produced* format, so the extension is the one authoritative record
    of what the stored bytes really are — ``job.output_format`` only records what
    the caller asked for. Falls back to the request when the path carries no
    extension (legacy rows).
    """
    if audio_path:
        suffix = PurePosixPath(audio_path).suffix.lstrip(".")
        if suffix:
            return suffix.lower()
    return requested_format


def _audio_duration_headers(duration_seconds: float | None) -> dict[str, str]:
    """``X-Audio-Duration`` for a MEASURED duration, nothing for an unmeasured one.

    ``None`` means ffprobe could not measure the audio. Emitting ``0.0`` there is
    a claim — on a 200 response carrying real audio bytes it tells the client it
    holds a zero-length clip. An absent header says only "we do not know", which
    is the truth, and leaves the audio itself untouched. Mirrors how
    ``X-Provider`` / ``X-Detected-Language`` are omitted when unknown.
    """
    if duration_seconds is None:
        return {}
    return {"X-Audio-Duration": str(duration_seconds)}


def _build_audio_response(result: GenerationResult, meta: dict, fallback_provider: str) -> Response:
    """Wrap a GenerationResult + meta into the audio Response with X-* headers."""
    content_type = _content_type_for(result.format)
    headers = {
        **_audio_duration_headers(result.duration_seconds),
        "X-Audio-Format": result.format,
        "X-Audio-Size": str(len(result.audio_data)),
        "X-Cache": "HIT" if meta.get("cached") else "MISS",
    }
    if meta.get("resolved_provider"):
        headers["X-Provider"] = meta["resolved_provider"]
    if meta.get("detected_language"):
        headers["X-Detected-Language"] = meta["detected_language"]
    if meta.get("estimated_cost") is not None:
        headers["X-Estimated-Cost-USD"] = str(meta["estimated_cost"])
    return Response(
        content=result.audio_data,
        media_type=content_type,
        headers=headers,
    )


@router.post(
    "/generate",
    response_model=None,
    operation_id="generate_speech",
    responses={
        400: {"description": "Validation error in generation parameters"},
        404: {"description": "Provider not found"},
        422: {"description": "Request body validation error"},
        429: {"description": "Quota exceeded"},
        503: {"description": "Provider error"},
    },
)
async def generate(
    body: TTSGenerateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response | AsyncJobResponse:
    """Generate TTS audio.

    - Sync mode (default): Returns audio bytes directly.
    - Async mode: Returns a job_id for polling.
    """
    tenant_id = _get_tenant_id(request)
    request_id = uuid.uuid7().hex[-8:]
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, request_id=request_id)
    tts = _get_tts_service(request)
    registry = _get_registry(request)

    logger.info(
        "TTS generate request: tenant=%s provider=%s async=%s text_len=%d",
        tenant_id,
        body.provider,
        body.async_mode,
        len(body.text),
    )

    _validate_provider_or_404(registry, body.provider)
    post_processing, provider_params = _resolve_quality_profile(body)

    if body.async_mode:
        return await _dispatch_async_job(tts, db, body, tenant_id, post_processing)

    no_cache = request.headers.get("X-Cache-Control", "").lower() == "no-cache"
    start_time = time.monotonic()
    result, meta = await _run_sync_generate(
        tts, db, body, tenant_id, post_processing, provider_params, no_cache
    )

    elapsed_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "TTS generate completed: tenant=%s provider=%s cached=%s size=%d elapsed=%.1fms",
        tenant_id,
        meta.get("resolved_provider", body.provider),
        meta.get("cached", False),
        len(result.audio_data),
        elapsed_ms,
    )
    return _build_audio_response(result, meta, body.provider)


async def _process_job_background(
    tts: TTSService,
    job_id: str,
    tenant_id: str,
    post_processing: object | None = None,
) -> None:
    """Background task to process a TTS job.

    Creates its own database session instead of reusing the request-scoped
    session, which will already be committed/closed by the time this runs.
    """
    try:
        async with get_db_session() as db:
            await tts.process_job(db, job_id, tenant_id, post_processing=post_processing)
    except QuotaExceededError as e:
        logger.warning("Background job %s quota exceeded: %s", job_id, e)
        await _mark_job_failed(job_id, f"Quota exceeded: {e}")
    except ValueError as e:
        logger.error("Background job %s validation error: %s", job_id, e)
        await _mark_job_failed(job_id, f"Validation error: {e}")
    except RuntimeError as e:
        logger.error("Background job %s provider error: %s", job_id, e)
        await _mark_job_failed(job_id, f"Provider error: {e}")
    # ``subprocess.TimeoutExpired`` subclasses SubprocessError — NOT OSError and
    # NOT the builtin TimeoutError — so a hung ffmpeg/ffprobe escaped every arm
    # above and the job never reached a terminal status. The same is true of
    # ``sqlalchemy.exc.SQLAlchemyError`` (psycopg3 faults are wrapped into it, so
    # a pgbouncer/Postgres blip left the job stuck in ``processing``) and of
    # ``redis.exceptions.RedisError`` (redis-py's ConnectionError/TimeoutError are
    # NOT the builtins of the same name).
    except (
        SQLAlchemyError,
        RedisError,
        OSError,
        TimeoutError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.exception("Background job %s failed with unexpected error", job_id)
        await _mark_job_failed(job_id, f"Unexpected error: {e}")


async def _mark_job_failed(job_id: str, error_message: str) -> None:
    """Mark a job as failed in a fresh DB session."""
    try:
        async with get_db_session() as db:
            await JobManager.mark_failed(db, job_id, error_message)
    except (
        SQLAlchemyError,
        RedisError,
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.TimeoutExpired,
    ):
        logger.exception("Failed to mark job %s as failed", job_id)


# ============================================================================
# SSE Streaming Endpoint (Feature 1)
# ============================================================================


@router.post(
    "/stream",
    operation_id="stream_speech_sse",
    responses={
        404: {"description": "Provider not found"},
        422: {"description": "Request body validation error"},
    },
)
async def stream_generate(
    body: TTSStreamRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream TTS audio via Server-Sent Events (SSE).

    Events:
        - metadata: {requested_provider, voice, format} — emitted BEFORE
          generation, so it names the provider ASKED FOR (often the literal
          "auto"), never the one that serves.
        - audio: {chunk_index, data (base64)}
        - done: {total_chunks, total_bytes, provider?} — ``provider`` is the
          provider that actually SERVED the stream, matching ``X-Provider`` on
          /generate. It is OMITTED when the serving provider is unknown rather
          than defaulted back to the requested one.
        - error: {message}
    """
    tts = _get_tts_service(request)
    registry = _get_registry(request)
    tenant_id = _get_tenant_id(request)

    logger.info(
        "TTS stream request: tenant=%s provider=%s text_len=%d format=%s",
        tenant_id,
        body.provider,
        len(body.text),
        body.output_format,
    )

    if body.provider != "auto":
        try:
            registry.get_provider(body.provider)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # Enforce quota BEFORE the first byte: once the StreamingResponse starts,
    # the 200 status is committed and a mid-stream raise cannot become a 429.
    # Without this, a tenant 429'd on /generate could stream unlimited audio.
    await _enforce_stream_quota(db, tenant_id, body.text, body.provider)

    async def event_stream() -> AsyncIterator[str]:
        try:
            # Send metadata event
            # ``requested_provider``, not ``provider``: this event is emitted
            # BEFORE generation, so it cannot know who will serve. It used to be
            # keyed ``provider`` while carrying ``body.provider`` — normally the
            # literal "auto" — and ``generate_stream``'s own fallback loop meant
            # the serving provider was frequently a different one, with nothing
            # downstream correcting the record. The resolved provider is reported
            # on the ``done`` event below, matching ``X-Provider`` on /generate.
            meta = {
                "requested_provider": body.provider,
                "voice": body.voice or "default",
                "format": body.output_format,
            }
            yield f"event: metadata\ndata: {json.dumps(meta)}\n\n"

            total_chunks = 0
            total_bytes = 0

            async for chunk in tts.generate_stream(
                text=body.text,
                provider=body.provider,
                voice=body.voice,
                speed=body.speed,
                output_format=body.output_format,
                model=body.model,
                language=body.language,
                tenant_id=tenant_id,
                db=db,
            ):
                if chunk.is_final:
                    done: dict[str, object] = {
                        "total_chunks": total_chunks,
                        "total_bytes": chunk.total_bytes or total_bytes,
                    }
                    # Omitted when unknown — never defaulted back to
                    # ``body.provider``, which is the claim this fixes.
                    if chunk.provider:
                        done["provider"] = chunk.provider
                    yield f"event: done\ndata: {json.dumps(done)}\n\n"
                else:
                    total_chunks += 1
                    total_bytes += len(chunk.data)
                    audio_event = {
                        "chunk_index": chunk.chunk_index,
                        "data": chunk.to_base64(),
                    }
                    yield f"event: audio\ndata: {json.dumps(audio_event)}\n\n"

        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            error = {"message": str(exc)}
            yield f"event: error\ndata: {json.dumps(error)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/stream/raw",
    operation_id="stream_speech_raw",
    responses={
        404: {"description": "Provider not found"},
        422: {"description": "Request body validation error"},
    },
)
async def stream_generate_raw(
    body: TTSStreamRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream raw binary audio chunks (chunked transfer encoding).

    Useful for direct audio piping without SSE overhead.
    """
    tts = _get_tts_service(request)
    registry = _get_registry(request)
    tenant_id = _get_tenant_id(request)

    logger.info(
        "TTS raw stream request: tenant=%s provider=%s text_len=%d format=%s",
        tenant_id,
        body.provider,
        len(body.text),
        body.output_format,
    )

    if body.provider != "auto":
        try:
            registry.get_provider(body.provider)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # Same pre-stream quota gate as the SSE route (see stream_generate).
    await _enforce_stream_quota(db, tenant_id, body.text, body.provider)

    content_type = "audio/mpeg" if body.output_format == "mp3" else f"audio/{body.output_format}"

    async def audio_stream() -> AsyncIterator[bytes]:
        async for chunk in tts.generate_stream(
            text=body.text,
            provider=body.provider,
            voice=body.voice,
            speed=body.speed,
            output_format=body.output_format,
            model=body.model,
            language=body.language,
            tenant_id=tenant_id,
            db=db,
        ):
            if not chunk.is_final and chunk.data:
                yield chunk.data

    return StreamingResponse(
        audio_stream(),
        media_type=content_type,
        headers={
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Batch Endpoint (Feature 2 enhanced)
# ============================================================================


@router.post(
    "/generate/batch",
    operation_id="generate_speech_batch",
    responses={
        400: {"description": "Validation error in batch parameters"},
        404: {"description": "Provider not found"},
        422: {"description": "Request body validation error"},
        429: {"description": "Quota exceeded"},
        503: {"description": "Provider error"},
    },
)
async def generate_batch(
    body: TTSBatchRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchJobResponse:
    """Generate TTS audio in batch (always async)."""
    tenant_id = _get_tenant_id(request)
    request_id = uuid.uuid7().hex[-8:]
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, request_id=request_id)
    tts = _get_tts_service(request)
    registry = _get_registry(request)

    logger.info(
        "TTS batch request: tenant=%s provider=%s items=%d concatenate=%s",
        tenant_id,
        body.provider,
        len(body.items),
        body.concatenate,
    )

    if body.provider != "auto":
        try:
            registry.get_provider(body.provider)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    batch_id = str(uuid.uuid7())
    job_ids: dict[str, str] = {}

    try:
        for item in body.items:
            job_id = await tts.generate_async(
                db,
                tenant_id=tenant_id,
                text=item.text,
                provider=body.provider,
                voice=body.voice,
                speed=body.speed,
                output_format=body.output_format,
                batch_id=batch_id,
                caller_item_id=item.id,
                strategy=body.strategy,
            )
            job_ids[item.id] = job_id
    except QuotaExceededError as e:
        logger.warning("Quota exceeded during batch for tenant=%s: %s", tenant_id, e)
        raise HTTPException(status_code=429, detail=str(e)) from e
    except RuntimeError as e:
        logger.error("Provider error during batch: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        logger.warning("Validation error during batch: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Commit so job records are visible to background tasks' sessions
    await db.commit()

    # Process all jobs in background with concurrency limiting
    async def _guarded_process(jid: str) -> None:
        async with tts.batch_semaphore:
            await _process_job_background(tts, jid, tenant_id, body.post_processing)

    for job_id in job_ids.values():
        task = asyncio.create_task(_guarded_process(job_id))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    logger.info("Batch %s created with %d jobs for tenant=%s", batch_id, len(job_ids), tenant_id)

    # If concatenation requested, schedule it after all jobs complete
    if body.concatenate:
        task = asyncio.create_task(
            _concatenate_batch_background(
                tts, batch_id, body.output_format, body.gap_ms, tenant_id=tenant_id
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return BatchJobResponse(
        batch_id=batch_id,
        job_ids=job_ids,
        status="pending",
        poll_url=f"/api/tts/jobs?batch_id={batch_id}",
    )


async def _concatenate_batch_background(
    tts: TTSService,
    batch_id: str,
    output_format: str,
    gap_ms: int,
    *,
    tenant_id: str,
) -> None:
    """Wait for all batch jobs to complete, then concatenate.

    Creates its own database session instead of reusing the request-scoped
    session, which will already be committed/closed by the time this runs.

    ``tenant_id`` is carried in from the creating request rather than looked up:
    ``JobManager`` has no unscoped read, so a detached background task has to
    remember whose batch it is working on. It is also a second safety net — this
    task can only ever touch rows of the tenant that asked for the batch.
    """
    try:
        async with get_db_session() as db:
            # Poll until all jobs are done (max 10 minutes)
            for _ in range(600):
                await asyncio.sleep(1)
                jobs = await JobManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)
                if not jobs:
                    return
                if all(j.status in ("completed", "failed") for j in jobs):
                    break
            else:
                logger.error("Batch %s timed out after 600s waiting for jobs to complete", batch_id)
                # Mark all still-pending jobs as failed so client gets feedback
                jobs = await JobManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)
                for job in jobs:
                    if job.status not in ("completed", "failed"):
                        await JobManager.mark_failed(
                            db, job.job_id, "Batch concatenation timed out"
                        )
                await db.commit()
                return

            await tts.concatenate_batch(db, batch_id, output_format, gap_ms, tenant_id=tenant_id)
            logger.info("Batch %s concatenation complete", batch_id)
    except OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired:
        # concatenate_batch shells out to ffmpeg with a timeout; TimeoutExpired
        # is a SubprocessError and is caught by none of the other arms.
        logger.exception("Batch %s concatenation failed", batch_id)


@router.get(
    "/batch/{batch_id}/audio",
    operation_id="download_speech_batch_audio",
    responses={
        404: {"description": "Batch or concatenated audio not found"},
        409: {"description": "Concatenated audio not yet available"},
    },
)
async def download_batch_audio(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download concatenated audio for a completed batch."""
    logger.debug("Batch audio download request: batch_id=%s", batch_id)
    jobs = await JobManager.get_batch_jobs(db, batch_id, tenant_id=_get_tenant_id(request))
    if not jobs:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    # Find concatenated audio path
    concat_path = None
    output_format = "mp3"
    for job in jobs:
        if job.concatenated_audio_path:
            concat_path = job.concatenated_audio_path
            output_format = job.output_format
            break

    if not concat_path:
        raise HTTPException(
            status_code=409,
            detail="Concatenated audio not yet available. Ensure concatenate=true and all jobs are completed.",
            headers={"Retry-After": "30"},
        )

    storage = _get_storage(request)
    audio_data = await storage.retrieve(concat_path)
    if audio_data is None:
        raise HTTPException(status_code=404, detail="Concatenated audio file not found")

    # As with the single-job download: the concatenated object's storage key
    # carries the format that was actually written.
    actual_format = _stored_format(concat_path, output_format)
    return Response(
        content=audio_data,
        media_type=_content_type_for(actual_format),
        headers={
            "Content-Disposition": f'attachment; filename="{batch_id}_concatenated.{actual_format}"',
            "X-Audio-Format": actual_format,
        },
    )


# ============================================================================
# Provider/Voice Endpoints
# ============================================================================


@router.get("/providers", operation_id="list_tts_providers")
async def list_providers(request: Request) -> list[ProviderInfo]:
    """List all registered TTS providers."""
    logger.debug("Listing TTS providers")
    registry = _get_registry(request)
    providers = registry.list_providers()
    settings = request.app.state.settings if hasattr(request.app.state, "settings") else None
    result = []
    for p in providers:
        available = await p.is_available()
        voices = await p.list_voices()

        cost = 0.0
        if settings:
            cost = settings.routing.cost_per_1k_chars.get(p.name, 0.0)

        result.append(
            ProviderInfo(
                name=p.name,
                display_name=p.display_name,
                provider_type=p.provider_type,
                available=available,
                voices_count=len(voices),
                default_voice=p.get_default_voice(),
                max_text_length=p.max_text_length,
                requires_api_key=p.requires_api_key,
                supports_streaming=p.supports_streaming,
                supports_ssml=p.supports_ssml,
                cost_per_1k_chars=cost,
            )
        )
    return result


@router.get(
    "/providers/{provider}/voices",
    operation_id="list_tts_voices",
    responses={
        404: {"description": "Provider not found"},
    },
)
async def list_voices(provider: str, request: Request) -> list[VoiceInfo]:
    """List voices for a specific provider."""
    logger.debug("Listing voices for provider=%s", provider)
    registry = _get_registry(request)
    try:
        p = registry.get_provider(provider)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    voices = await p.list_voices()
    return [
        VoiceInfo(
            id=v.id,
            display_name=v.display_name,
            gender=v.gender,
            language=v.language,
            languages=v.languages,
            preview_url=v.preview_url,
        )
        for v in voices
    ]


# ============================================================================
# Job Endpoints
# ============================================================================


@router.get(
    "/jobs/{job_id}",
    operation_id="get_tts_job_status",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_status(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Get status of a TTS generation job."""
    logger.debug("Job status request: job_id=%s", job_id)
    # Tenant-scoped: this response carries a PRESIGNED MinIO URL, which needs no
    # further auth once handed out. An unscoped lookup here turned a guessed job
    # id into another tenant's audio.
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    audio_url = f"/api/tts/jobs/{job_id}/audio" if job.status == "completed" else None

    # Presigned URL for MinIO backend
    presigned_url = None
    if job.status == "completed" and job.audio_path:
        storage = _get_storage(request)
        presigned_url = await storage.get_presigned_url(job.audio_path)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        provider=job.provider,
        voice=job.voice,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        audio_url=audio_url,
        presigned_url=presigned_url,
        duration_seconds=job.audio_duration_seconds,
        size_bytes=job.audio_size_bytes,
        cached=job.cached or False,
        detected_language=job.detected_language,
        routing_strategy=job.routing_strategy,
        estimated_cost_usd=job.estimated_cost_usd,
    )


@router.get(
    "/jobs/{job_id}/audio",
    operation_id="download_tts_audio",
    responses={
        404: {"description": "Job or audio file not found"},
        409: {"description": "Job is not completed yet"},
    },
)
async def download_audio(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download audio for a completed TTS job."""
    logger.debug("Audio download request: job_id=%s", job_id)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )
    if not job.audio_path:
        raise HTTPException(status_code=404, detail="Audio file path not found")

    storage = _get_storage(request)
    audio_data = await storage.retrieve(job.audio_path)
    if audio_data is None:
        raise HTTPException(status_code=404, detail="Audio file not found on storage")

    # Content-Type + extension follow the PRODUCED bytes (storage key), not the
    # requested job.output_format — a provider may not have honoured the request.
    actual_format = _stored_format(job.audio_path, job.output_format)
    return Response(
        content=audio_data,
        media_type=_content_type_for(actual_format),
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.{actual_format}"',
            # ``or 0`` used to make an unmeasured duration byte-identical to a
            # measured zero. The column is nullable precisely because the value
            # can be unknown; say so by omission.
            **_audio_duration_headers(job.audio_duration_seconds),
            "X-Audio-Format": actual_format,
        },
    )


@router.get(
    "/jobs/{job_id}/audio/url",
    operation_id="get_tts_audio_url",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Audio not available yet"},
    },
)
async def get_audio_presigned_url(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a presigned URL for downloading audio (MinIO backend)."""
    logger.debug("Presigned URL request: job_id=%s", job_id)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed" or not job.audio_path:
        raise HTTPException(
            status_code=409,
            detail="Audio not available",
            headers={"Retry-After": "30"},
        )

    storage = _get_storage(request)
    url = await storage.get_presigned_url(job.audio_path)
    if url is not None:
        return {"url": url, "expires_in": 3600}

    # Local storage fallback: return the direct download endpoint URL
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/api/tts/jobs/{job_id}/audio"
    return {"url": download_url, "expires_in": None, "backend": "local"}
