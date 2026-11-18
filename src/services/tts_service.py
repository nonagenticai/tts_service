"""Core TTS orchestration service with all feature integrations."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess  # nosec B404 -- imported for TimeoutExpired only, no process is spawned here
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import W2V_APP_KEY
from w2v_common.auth.identity_context import (
    get_attribution_subject_id_from_context,
    get_license_id_from_context,
)
from w2v_common.observability.request_context import get_request_id

from src.cache.redis_client import RedisClient
from src.config import ConcurrencySettings, get_settings
from src.providers.base import (
    PROVIDER_FAILURE_ERRORS,
    AudioChunk,
    GenerationResult,
    TTSProvider,
)
from src.providers.registry import ProviderRegistry
from src.services.job_manager import JobManager
from src.services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    funding_for_provider,
    schedule_cost_event,
    synthesis_event_key,
)
from src.services.protocols import CacheChecker
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

_PROVIDER_SEMAPHORE_MAP = {
    "openai": "openai_max_concurrent",
    "elevenlabs": "elevenlabs_max_concurrent",
    "kokoro": "kokoro_max_concurrent",
    "kitten": "kitten_max_concurrent",
}


def _compute_cache_key(
    tenant_id: str,
    text: str,
    provider: str,
    voice: str,
    speed: float,
    output_format: str,
) -> str:
    """SHA256-based cache key including tenant_id for isolation."""
    raw = f"{tenant_id}:{provider}:{voice}:{speed}:{output_format}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


#: Provider values that name no concrete provider yet. ``"auto"`` is the DEFAULT
#: request value — the router has not run, so no rate applies.
_UNRESOLVED_PROVIDER_VALUES = frozenset({"auto", ""})


def _estimate_cost(provider_name: str, character_count: int) -> float:
    """Estimate generation cost in USD.

    ``"auto"`` has no entry in ``cost_per_1k_chars``, so the old
    ``rates.get(name, 0.0)`` priced every un-pinned request at $0.00 — and
    ``"auto"`` is the DEFAULT. The pre-flight monthly-budget ceiling in
    ``quota_service.check_quota`` therefore never fired on the default path: a
    single arbitrarily large request could overshoot the cap outright.

    A pre-flight ceiling must never UNDER-estimate, so an unresolved provider is
    priced at the worst case the router could pick. Callers that have already
    resolved a concrete provider (see :meth:`TTSService.generate_sync`) pass
    that name instead and get the exact rate.
    """
    settings = get_settings()
    rates = settings.routing.cost_per_1k_chars
    if provider_name in _UNRESOLVED_PROVIDER_VALUES:
        rate = max(rates.values(), default=0.0)
    else:
        rate = rates.get(provider_name, 0.0)
    return (character_count / 1000.0) * rate


@dataclass(frozen=True)
class _StreamMeterContext:
    """The per-request facts every metering arm of one stream needs.

    A value object rather than seven repeated keyword arguments: the three arms
    (success, produced-then-failed, client teardown) MUST bill identically, and
    three separate argument lists is how they would drift apart.
    """

    tenant_id: str
    provider: str
    text: str
    model: str | None
    output_format: str
    started_at: float
    record_usage: bool


@dataclass
class _StreamAttempt:
    """Mutable per-candidate state for one provider attempt in a stream.

    ``produced`` means AUDIO BYTES reached the client — not merely "a chunk was
    seen". It used to be set on ANY chunk including the provider's empty
    ``is_final=True`` sentinel, which meant a synthesis that died before its
    first segment (and therefore emitted only that sentinel) was treated as a
    completed stream: no fallback was tried, the tenant was billed for the full
    text, and ``/api/tts/stream/raw`` returned 200 ``audio/mpeg`` with zero
    bytes. A first-segment failure was indistinguishable from success.

    ``pending_final`` buffers that sentinel so it is forwarded only once real
    audio has been produced — a ``done`` event cannot be un-sent, so emitting it
    eagerly would foreclose falling back to the next candidate.
    """

    produced: bool = False
    pending_final: AudioChunk | None = None
    #: Bound up-front so the produced-then-failed metering path can reference it
    #: unconditionally.
    processed_text: str = ""
    #: Set the instant metering is ATTEMPTED, not when it succeeds. There are now
    #: three ways out of one attempt (success, provider failure after bytes, and
    #: client teardown) and the teardown arm is a ``finally``, so it fires on the
    #: other two as well. Without this latch the success path would meter, return
    #: through the ``finally``, and meter a SECOND time — turning a fix for a
    #: billing bypass into a double-bill, which is strictly worse than the bug.
    metered: bool = False


class TTSService:
    """Orchestrates TTS generation across providers."""

    def __init__(
        self,
        registry: ProviderRegistry,
        storage: StorageService,
        redis_client: RedisClient | None = None,
        router: object | None = None,
        notification_service: object | None = None,
        concurrency_settings: ConcurrencySettings | None = None,
        cache_checker: CacheChecker | None = None,
        metering_redis: object | None = None,
    ) -> None:
        self._registry = registry
        self._storage = storage
        self._redis = redis_client
        self._router = router
        self._notifications = notification_service
        self._settings = get_settings()
        # DI seam for the cache-check phase. Defaults to the bound
        # `_check_cache` method; tests can pass a fake matching CacheChecker.
        self._cache_checker: CacheChecker = cache_checker or self._check_cache
        # Metering: None = disabled (METERING_REDIS_URL unset or unreachable);
        # any truthy object exposes async .xadd(stream, fields, maxlen=..., approximate=...).
        # The stream name + the GC-safe background-task lifecycle are owned
        # centrally by ``w2v_common.metering.schedule_cost_event``.
        self._metering_redis = metering_redis

        cs = concurrency_settings or ConcurrencySettings()
        self._generation_semaphore = asyncio.Semaphore(cs.max_concurrent_generations)
        self._batch_semaphore = asyncio.Semaphore(cs.max_concurrent_batch_jobs)
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        for provider_name, attr_name in _PROVIDER_SEMAPHORE_MAP.items():
            self._provider_semaphores[provider_name] = asyncio.Semaphore(getattr(cs, attr_name))

    @property
    def batch_semaphore(self) -> asyncio.Semaphore:
        """Semaphore for limiting concurrent batch job processing."""
        return self._batch_semaphore

    def set_cache_checker(self, checker: CacheChecker) -> None:
        """Replace the cache-check seam.

        Public DI hook used by composition roots (and tests) to swap in a
        different cache-check implementation without subclassing.
        """
        self._cache_checker = checker

    # ========================================================================
    # Provider Resolution
    # ========================================================================

    async def _resolve_provider_chain(
        self,
        provider: str,
        text: str,
        strategy: str | None = None,
        language: str | None = None,
    ) -> tuple[TTSProvider, str, list[str]]:
        """Resolve a provider request into (instance, name, fallback_chain).

        If provider == 'auto' and a ``ProviderRouter`` is available, delegates to
        the router and returns both the winning provider AND the ranked fallback
        chain the router computed. For an explicit provider the chain is empty:
        the caller pinned that provider, so we honour it without falling back.
        """
        if provider == "auto" and self._router is not None:
            from src.api.schemas import RoutingStrategy
            from src.services.router import ProviderRouter

            if isinstance(self._router, ProviderRouter):
                strat_value = strategy or self._settings.routing.default_strategy
                try:
                    routing_strategy = RoutingStrategy(strat_value)
                except ValueError:
                    routing_strategy = RoutingStrategy.BALANCED
                decision = await self._router.select_provider(
                    text=text,
                    strategy=routing_strategy,
                    language=language,
                )
                p = self._registry.get_provider(decision.provider_name)
                return p, decision.provider_name, list(decision.fallback_chain)
        p = self._registry.get_provider(provider)
        return p, provider, []

    async def _resolve_provider(
        self,
        provider: str,
        text: str,
        strategy: str | None = None,
        language: str | None = None,
    ) -> tuple[TTSProvider, str]:
        """Resolve provider name to a TTSProvider instance (no fallback chain).

        Thin wrapper over :meth:`_resolve_provider_chain` kept for callers that
        only need the winning provider.
        """
        p, name, _chain = await self._resolve_provider_chain(provider, text, strategy, language)
        return p, name

    def _resolve_voice(self, provider: TTSProvider, voice: str | None) -> str:
        """Resolve voice: use provided voice or provider default."""
        if voice and voice != "default":
            return voice
        return provider.get_default_voice()

    # ========================================================================
    # SSML Processing
    # ========================================================================

    def _process_text(
        self,
        text: str,
        provider_name: str,
        ssml: bool | None = None,
    ) -> str:
        """Process text through SSML if applicable."""
        try:
            from src.services.ssml_processor import is_ssml, process_ssml_for_provider
        except ImportError:
            return text

        if ssml is None:
            ssml = is_ssml(text)
        if ssml:
            return process_ssml_for_provider(text, provider_name)
        return text

    # ========================================================================
    # Language Detection
    # ========================================================================

    def _detect_language(
        self,
        text: str,
        provider_name: str,
        voice: str | None,
        voices: Sequence[object] | None = None,
    ) -> tuple[str | None, str | None]:
        """Detect language and optionally select best voice.

        Returns (detected_language, best_voice_or_None).
        """
        try:
            from src.services.language_detector import (
                detect_language,
                get_best_voice_for_language,
            )
        except ImportError:
            return None, None

        lang, confidence = detect_language(text)
        if lang and confidence >= 0.8:
            # Only auto-select voice if no explicit voice was provided
            if voice is None:
                best = get_best_voice_for_language(provider_name, lang, voices or [])
                return lang, best
            return lang, None
        return None, None

    # ========================================================================
    # Caching
    # ========================================================================

    async def _check_cache(
        self,
        tenant_id: str,
        text: str,
        provider: str,
        voice: str,
        speed: float,
        output_format: str,
    ) -> tuple[bytes | None, str, float | None]:
        """Check Redis cache for existing generation.

        Returns (audio_data, cache_key, cached_duration_seconds). The duration is
        ``None`` when the cached entry carries no measurement — see
        :meth:`_store_cache`.
        """
        if not self._redis or not self._settings.cache.enabled:
            return None, "", None

        cache_key = _compute_cache_key(tenant_id, text, provider, voice, speed, output_format)
        prefix = self._settings.cache.key_prefix

        # Check metadata cache (stores audio path + duration)
        meta = await self._redis.get(f"{prefix}{cache_key}")
        if meta and isinstance(meta, dict):
            audio_path = meta.get("audio_path")
            if audio_path:
                audio_data = await self._storage.retrieve(audio_path)
                if audio_data:
                    # ABSENT means unmeasured, not zero. Never re-default here:
                    # this is exactly where a poisoned entry would be laundered
                    # back into a confident number on every hit for the whole TTL.
                    raw_duration = meta.get("duration")
                    cached_duration = None if raw_duration is None else float(raw_duration)
                    logger.info("Cache HIT for key %s", cache_key[:16])
                    return audio_data, cache_key, cached_duration

        return None, cache_key, None

    async def _store_cache(
        self,
        cache_key: str,
        audio_path: str,
        duration: float | None,
        fmt: str,
    ) -> None:
        """Store generation metadata in Redis cache.

        ⚠️ An UNMEASURED duration (``None``) is OMITTED from the payload rather
        than written as ``0.0``. The cache outlives the ffprobe outage that
        produced the gap, so a substituted number here is not one bad response —
        it is every hit on this key for the whole TTL, with nothing to say the
        value was never measured. Same shape as ``_measure_audio_metrics``, which
        omits ``meta["audio_metrics"]`` rather than substituting.
        """
        if not self._redis or not cache_key or not self._settings.cache.enabled:
            return

        prefix = self._settings.cache.key_prefix
        ttl = self._settings.cache.default_ttl
        payload: dict = {"audio_path": audio_path, "format": fmt}
        if duration is not None:
            payload["duration"] = duration
        await self._redis.set(f"{prefix}{cache_key}", payload, ttl=ttl)

    # ========================================================================
    # Post-Processing
    # ========================================================================

    async def _apply_post_processing(
        self,
        audio_data: bytes,
        fmt: str,
        sample_rate: int,
        post_processing: object | None,
    ) -> tuple[bytes, float]:
        """Apply audio post-processing pipeline if configured.

        Returns (processed_audio_data, processed_duration_seconds).
        """
        if post_processing is None:
            return audio_data, 0.0

        from src.api.schemas import PostProcessingConfig

        if not isinstance(post_processing, PostProcessingConfig):
            return audio_data, 0.0
        if not post_processing.has_any_processing():
            return audio_data, 0.0

        from src.utils.post_processor import AudioPostProcessor

        processor = AudioPostProcessor()
        pp_result = await asyncio.to_thread(
            processor.process, audio_data, fmt, sample_rate, post_processing
        )
        return pp_result.audio_data, pp_result.duration_seconds

    # ========================================================================
    # Notifications
    # ========================================================================

    async def _notify(self, event_type: str, job_id: str, tenant_id: str, **extra: object) -> None:
        """Publish a job event notification."""
        if self._notifications is None:
            return
        try:
            from src.services.notification_service import JobEvent, NotificationService

            if isinstance(self._notifications, NotificationService):
                event = JobEvent(
                    event=event_type,
                    job_id=job_id,
                    tenant_id=tenant_id,
                    payload=dict(extra),
                )
                await self._notifications.publish_event(event)
        # ``RedisError`` is mandatory here: redis-py's ConnectionError/TimeoutError
        # do NOT subclass the builtins, so a Redis outage used to escape this
        # handler and propagate into the caller. A notification is best-effort
        # and must never fail the work it is reporting on.
        except RedisError, OSError, TimeoutError, ValueError, RuntimeError:
            logger.warning("Failed to publish notification for job=%s", job_id, exc_info=True)

    # ========================================================================
    # Sync Generation
    # ========================================================================

    async def _detect_language_and_voice(
        self,
        text: str,
        provider_obj: TTSProvider,
        resolved_provider: str,
        voice: str | None,
        resolved_voice: str,
        language: str | None,
        meta: dict,
    ) -> str:
        """Run language detection (when no override) and auto-select a voice if needed."""
        if language is not None:
            meta["detected_language"] = language
            return resolved_voice
        voices = await provider_obj.list_voices() if voice is None else []
        detected_lang, auto_voice = self._detect_language(text, resolved_provider, voice, voices)
        if detected_lang:
            meta["detected_language"] = detected_lang
            if auto_voice and voice is None:
                return auto_voice
        return resolved_voice

    async def _try_cache_hit(
        self,
        *,
        no_cache: bool,
        tenant_id: str,
        processed_text: str,
        resolved_provider: str,
        resolved_voice: str,
        speed: float,
        output_format: str,
        provider_obj: TTSProvider,
        meta: dict,
        start_time: float,
    ) -> tuple[GenerationResult | None, str]:
        """Check cache. Returns (result, "") on hit, else (None, cache_key) on miss.

        On hit, also fills `meta` with cached/cache_hit/latency_ms/estimated_cost.
        """
        if no_cache:
            return None, ""
        cached_audio, cache_key, cached_duration = await self._cache_checker(
            tenant_id,
            processed_text,
            resolved_provider,
            resolved_voice,
            speed,
            output_format,
        )
        if cached_audio:
            meta["cached"] = True
            meta["cache_hit"] = True
            result = GenerationResult(
                audio_data=cached_audio,
                format=output_format,
                sample_rate=provider_obj.sample_rate,
                duration_seconds=cached_duration,
            )
            meta["latency_ms"] = (time.monotonic() - start_time) * 1000
            meta["estimated_cost"] = 0.0
            return result, ""
        return None, cache_key

    async def _invoke_provider(
        self,
        *,
        provider_obj: TTSProvider,
        resolved_provider: str,
        processed_text: str,
        resolved_voice: str,
        speed: float,
        output_format: str,
        model: str | None,
        provider_params: dict[str, float] | None,
    ) -> GenerationResult:
        """Call provider.generate() under the configured concurrency semaphores."""
        kwargs: dict[str, object] = {}
        if model:
            kwargs["model"] = model
        if provider_params:
            kwargs.update(provider_params)

        provider_sem = self._provider_semaphores.get(resolved_provider)
        async with self._generation_semaphore:
            if provider_sem:
                async with provider_sem:
                    return await provider_obj.generate(
                        text=processed_text,
                        voice=resolved_voice,
                        speed=speed,
                        output_format=output_format,
                        **kwargs,
                    )
            return await provider_obj.generate(
                text=processed_text,
                voice=resolved_voice,
                speed=speed,
                output_format=output_format,
                **kwargs,
            )

    def _record_health(self, provider: str, start: float, *, ok: bool) -> None:
        """Best-effort feed the router's health tracker with a call outcome."""
        from src.services.router import ProviderRouter

        if not isinstance(self._router, ProviderRouter):
            return
        self._router.record_result(provider, (time.monotonic() - start) * 1000, ok=ok)

    async def _generate_with_fallback(
        self,
        *,
        primary: TTSProvider,
        primary_name: str,
        primary_voice: str,
        primary_text: str,
        fallback_chain: list[str],
        original_text: str,
        voice: str | None,
        ssml: bool | None,
        speed: float,
        output_format: str,
        model: str | None,
        provider_params: dict[str, float] | None,
        meta: dict,
    ) -> tuple[GenerationResult, str, str, str]:
        """Try the primary provider, then each fallback, returning the first success.

        The single buffered-generation fallback loop, shared by the sync path and
        (via ``process_job`` -> ``generate_sync``) the async job path. It invokes
        each provider through :meth:`_invoke_provider` (so concurrency semaphores
        apply) and, on a provider/transport failure, degrades to the next entry
        of the router-computed chain. Only a caller-pinned provider (empty chain)
        cannot fall back. The streaming path cannot buffer chunks, so it runs its
        own analogous loop in :meth:`generate_stream` sharing the same error set.

        Returns ``(result, used_provider, used_voice, used_processed_text)`` and
        sets ``meta["resolved_provider"]`` to the provider that actually produced
        audio (plus ``meta["fallback_used"] = True`` when it was not the primary).
        Raises ``RuntimeError`` only when every candidate is unavailable or fails.
        """
        attempts: list[tuple[TTSProvider, str, str, str]] = [
            (primary, primary_name, primary_voice, primary_text)
        ]
        for name in fallback_chain:
            if name == primary_name:
                continue
            try:
                fb = self._registry.get_provider(name)
            except KeyError:
                logger.warning("Fallback provider '%s' is not registered; skipping", name)
                continue
            fb_voice = self._resolve_voice(fb, voice)
            fb_text = self._process_text(original_text, name, ssml)
            attempts.append((fb, name, fb_voice, fb_text))

        last_error: Exception | None = None
        for obj, name, rvoice, ptext in attempts:
            start = time.monotonic()
            try:
                if not await obj.is_available():
                    logger.warning("Provider '%s' unavailable; trying next candidate", name)
                    continue
                result = await self._invoke_provider(
                    provider_obj=obj,
                    resolved_provider=name,
                    processed_text=ptext,
                    resolved_voice=rvoice,
                    speed=speed,
                    output_format=output_format,
                    model=model,
                    provider_params=provider_params,
                )
            except PROVIDER_FAILURE_ERRORS as exc:
                self._record_health(name, start, ok=False)
                last_error = exc
                logger.warning("Provider '%s' failed; trying fallback: %s", name, exc)
                continue
            self._record_health(name, start, ok=True)
            meta["resolved_provider"] = name
            if name != primary_name:
                meta["fallback_used"] = True
            return result, name, rvoice, ptext

        if last_error is not None:
            raise RuntimeError(
                f"All TTS providers failed. Last error: {last_error}"
            ) from last_error
        raise RuntimeError(f"Provider '{primary_name}' is not available")

    async def _measure_audio_metrics(self, result: GenerationResult, meta: dict) -> None:
        """Best-effort attach loudness/peak/silence metrics to meta. Never raises.

        ``subprocess.TimeoutExpired`` is caught explicitly — it subclasses
        ``SubprocessError``, not ``OSError`` and not the builtin ``TimeoutError``,
        so the rest of the tuple does not cover it. A single hung ffprobe used to
        escape this "never raises" method all the way out of ``process_job`` and
        strand the job row in ``processing`` until the service was restarted."""
        try:
            from src.utils.post_processor import AudioPostProcessor

            processor = AudioPostProcessor()
            metrics = await asyncio.to_thread(
                processor.measure_audio, result.audio_data, result.format
            )
            meta["audio_metrics"] = {
                "integrated_lufs": metrics.integrated_lufs,
                "loudness_range_lu": metrics.loudness_range_lu,
                "true_peak_dbtp": metrics.true_peak_dbtp,
                "silence_ratio": metrics.silence_ratio,
                "duration_seconds": metrics.duration_seconds,
            }
        except OSError, TimeoutError, ValueError, RuntimeError, subprocess.TimeoutExpired:
            logger.debug("Audio metrics measurement failed", exc_info=True)

    async def _persist_cache(self, cache_key: str, result: GenerationResult) -> None:
        """Best-effort store the generated audio in the cache. Never raises."""
        if not (cache_key and self._redis):
            return
        try:
            path = await self._storage.store(
                f"cache_{cache_key[:32]}", result.audio_data, result.format
            )
            await self._store_cache(cache_key, path, result.duration_seconds, result.format)
        except OSError, TimeoutError, ValueError, RuntimeError:
            logger.debug("Cache store failed", exc_info=True)

    async def _record_sync_usage(
        self,
        db: AsyncSession | None,
        *,
        tenant_id: str,
        provider: str,
        characters: int,
        audio_seconds: float,
        meta: dict,
    ) -> None:
        """Record a ``TTSUsage`` row for a synchronous generation.

        Quota enforcement (:func:`quota_service.check_quota`) is computed by
        aggregating ``TTSUsage`` rows, but usage used to be recorded ONLY inside
        :meth:`process_job` — the async path. A tenant posting
        ``async_mode=false`` therefore passed quota forever while
        :meth:`_emit_cost_event` still billed it, so billing and quota disagreed.

        This is the same ``JobManager.record_usage`` call shape the async path
        uses — one accounting path with two entry points, not a second one.
        Skipped when ``db`` is None (no session ⇒ nothing to aggregate against).
        """
        if db is None:
            return
        await JobManager.record_usage(
            db,
            tenant_id=tenant_id,
            provider=provider,
            characters=characters,
            audio_seconds=audio_seconds,
            estimated_cost_usd=meta.get("estimated_cost") or 0.0,
            latency_ms=meta.get("latency_ms") or 0.0,
            cache_hit=meta.get("cache_hit", False),
        )

    async def _record_stream_usage(
        self,
        db: AsyncSession | None,
        *,
        tenant_id: str,
        provider: str,
        characters: int,
        latency_ms: float,
    ) -> None:
        """Record a ``TTSUsage`` row for a completed streamed generation.

        The streaming path bills the same per-character unit as the sync/async
        paths — without this a tenant 429'd on ``/generate`` could stream
        unlimited, unattributed, paid-provider audio via ``/stream`` because the
        stream routes wrote no usage row and quota aggregates those rows.
        ``audio_seconds`` is 0.0: a live stream has no known final duration
        (metering is on completion, keyed on characters). Skipped when ``db`` is
        None (no session ⇒ nothing to aggregate against).
        """
        if db is None:
            return
        await JobManager.record_usage(
            db,
            tenant_id=tenant_id,
            provider=provider,
            characters=characters,
            audio_seconds=0.0,
            estimated_cost_usd=_estimate_cost(provider, characters),
            latency_ms=latency_ms,
        )

    def _emit_cost_event(
        self,
        *,
        tenant_id: str,
        provider: str,
        model: str,
        text: str,
        duration_s: float,
        audio_format: str,
        latency_ms: int,
        call_kind: str,
        call_id: str | None,
    ) -> None:
        """Fire-and-forget cost event to billing:costs, fully attributed (G5).

        No-ops when METERING_REDIS_URL is unset / Redis was unreachable at boot.
        license_id + subject_id come from the per-request ContextVars set by
        w2v_common.auth.middleware.AuthMiddleware; ``app_id`` is the w2v app
        key; ``funding`` is declared per provider (an undeclared provider
        raises); ``event_key`` is derived from ``call_kind``/``call_id`` (job id
        or inbound request id) so a re-emit of the same synthesis dedupes.
        project/actor come from w2v_common's ambient chain. Under
        ``METERING_REQUIRE_ATTRIBUTION`` a missing field raises instead of
        shipping a partial event.
        """
        if self._metering_redis is None:
            return
        funding = funding_for_provider(provider)
        schedule_cost_event(
            cast(Any, self._metering_redis),
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=tenant_id,
            license_id=get_license_id_from_context(),
            subject_id=get_attribution_subject_id_from_context(),
            app_id=W2V_APP_KEY,
            funding=funding,
            event_key=synthesis_event_key(call_kind, call_id),
            provider=provider,
            model=model,
            input_tokens=len(text),
            output_tokens=0,
            latency_ms=latency_ms,
            # Bill the estimated provider spend, not $0. characters == len(text)
            # (see usage_extra unit below); mirrors _record_stream_usage's use of
            # the same helper so streaming and one-shot report costs consistently.
            cost_usd=_estimate_cost(provider, len(text)),
            usage_extra={
                "unit": "characters",
                "duration_s": duration_s,
                "audio_format": audio_format,
            },
        )

    async def generate_sync(
        self,
        *,
        text: str,
        provider: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: str = "mp3",
        model: str | None = None,
        tenant_id: str,
        language: str | None = None,
        ssml: bool | None = None,
        strategy: str | None = None,
        post_processing: object | None = None,
        provider_params: dict[str, float] | None = None,
        no_cache: bool = False,
        db: AsyncSession | None = None,
        record_usage: bool = True,
        emit_cost: bool = True,
    ) -> tuple[GenerationResult, dict]:
        """Generate audio synchronously (direct return).

        Returns (result, metadata_dict) where metadata contains:
        cached, cache_hit, detected_language, resolved_provider, estimated_cost, latency_ms

        If db is provided and quota enforcement is enabled, checks tenant quota
        before generation and raises QuotaExceededError if limits are exceeded.

        ``record_usage`` (default True) writes the ``TTSUsage`` row that
        ``check_quota`` later aggregates. :meth:`process_job` passes False
        because it records the same usage itself with job-level attribution —
        without that flag the async path would double-count.

        ``emit_cost`` (default True) fires the billing cost event once audio is
        produced. :meth:`process_job` passes False and emits the event ITSELF —
        only after the job is durably marked ``completed`` — so a storage/DB
        failure *after* a paid synthesis cannot bill the tenant while the job
        row reads ``failed``. The synchronous HTTP path returns the bytes
        directly (no later commit that can fail), so it emits here as before.
        """
        meta: dict = {"cached": False, "cache_hit": False}
        start_time = time.monotonic()

        # Resolve BEFORE the quota check: pricing the RAW request value meant
        # the default ``provider="auto"`` always projected $0.00 (it has no
        # ``cost_per_1k_chars`` entry), so the monthly-budget ceiling could not
        # fire on the default path. With the routed name the projection is the
        # provider's exact rate.
        p, resolved_provider, fallback_chain = await self._resolve_provider_chain(
            provider, text, strategy, language
        )
        meta["resolved_provider"] = resolved_provider

        if db is not None:
            from src.services.quota_service import check_quota

            await check_quota(
                db,
                tenant_id,
                len(text),
                projected_cost=_estimate_cost(resolved_provider, len(text)),
            )

        resolved_voice = self._resolve_voice(p, voice)
        resolved_voice = await self._detect_language_and_voice(
            text, p, resolved_provider, voice, resolved_voice, language, meta
        )
        processed_text = self._process_text(text, resolved_provider, ssml)

        cached_result, cache_key = await self._try_cache_hit(
            no_cache=no_cache,
            tenant_id=tenant_id,
            processed_text=processed_text,
            resolved_provider=resolved_provider,
            resolved_voice=resolved_voice,
            speed=speed,
            output_format=output_format,
            provider_obj=p,
            meta=meta,
            start_time=start_time,
        )
        if cached_result is not None:
            # A cache hit is still a billable request and still consumes the
            # tenant's request/character quota — the async path records it too.
            # ``billed_text`` is the single source of truth for the character
            # count of this request: the text actually synthesized (SSML tags
            # stripped for non-SSML providers). Every downstream counter —
            # quota row, cost bus, async-job attribution — reads it, so they
            # can never disagree. See the cache-MISS branch below.
            meta["billed_text"] = processed_text
            if record_usage:
                await self._record_sync_usage(
                    db,
                    tenant_id=tenant_id,
                    provider=resolved_provider,
                    characters=len(processed_text),
                    # Metering keeps its existing numeric contract: an
                    # unmeasured duration meters as it always has (0.0). Making
                    # the gap visible to BILLING is a separate, deliberate
                    # decision and is out of scope here.
                    audio_seconds=cached_result.duration_seconds or 0.0,
                    meta=meta,
                )
            return cached_result, meta

        # Generate through the shared fallback loop: the primary provider first,
        # then the router-computed chain. ``used_*`` reflect whichever provider
        # actually produced audio, so cost/cache/metering below attribute it.
        result, used_provider, used_voice, used_text = await self._generate_with_fallback(
            primary=p,
            primary_name=resolved_provider,
            primary_voice=resolved_voice,
            primary_text=processed_text,
            fallback_chain=fallback_chain,
            original_text=text,
            voice=voice,
            ssml=ssml,
            speed=speed,
            output_format=output_format,
            model=model,
            provider_params=provider_params,
            meta=meta,
        )

        if post_processing:
            processed_data, processed_duration = await self._apply_post_processing(
                result.audio_data, result.format, result.sample_rate, post_processing
            )
            result = GenerationResult(
                audio_data=processed_data,
                format=result.format,
                sample_rate=result.sample_rate,
                duration_seconds=processed_duration
                if processed_duration > 0
                else result.duration_seconds,
                metadata=result.metadata,
            )

        await self._measure_audio_metrics(result, meta)

        # Single source of truth for this request's billable character count:
        # the text the provider actually synthesized. ``used_text`` is the
        # SSML-processed text (tags stripped for non-SSML providers) belonging
        # to whichever provider served after fallback. Quota, cost estimate,
        # cost bus and the async-job path all count THIS string.
        meta["billed_text"] = used_text
        meta["estimated_cost"] = _estimate_cost(used_provider, len(used_text))
        meta["latency_ms"] = (time.monotonic() - start_time) * 1000

        # ``cache_key`` was keyed on the primary provider/voice; if a fallback
        # served the request, re-key so the audio is cached under the provider
        # that actually produced it (cache keys include provider for isolation).
        persist_key = cache_key
        if cache_key and used_provider != resolved_provider:
            persist_key = _compute_cache_key(
                tenant_id, used_text, used_provider, used_voice, speed, output_format
            )
        await self._persist_cache(persist_key, result)
        # Record usage BEFORE emitting the cost event so quota and billing can
        # never disagree about a request that was served.
        if record_usage:
            await self._record_sync_usage(
                db,
                tenant_id=tenant_id,
                provider=used_provider,
                # Bill quota on the SAME text the cost bus bills (``used_text``,
                # SSML tags stripped for non-SSML providers) so quota and
                # billing can never disagree on an SSML request.
                characters=len(used_text),
                audio_seconds=result.duration_seconds or 0.0,
                meta=meta,
            )
        if emit_cost:
            self._emit_cost_event(
                tenant_id=tenant_id,
                provider=used_provider,
                model=model or used_provider,
                text=used_text,
                duration_s=result.duration_seconds or 0.0,
                audio_format=result.format,
                latency_ms=int(meta["latency_ms"]),
                call_kind="sync",
                call_id=get_request_id(),
            )
        return result, meta

    # ========================================================================
    # Streaming Generation
    # ========================================================================

    async def generate_stream(
        self,
        *,
        text: str,
        provider: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: str = "mp3",
        model: str | None = None,
        language: str | None = None,
        tenant_id: str,
        db: AsyncSession | None = None,
        record_usage: bool = True,
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from a provider.

        Args:
            text: Text to synthesize.
            provider: Provider name or "auto" for routing.
            voice: Optional voice ID override.
            speed: Playback speed multiplier.
            output_format: Audio format.
            model: Optional provider-specific model.
            language: Optional language hint.
            tenant_id: Tenant to attribute usage + billing to.
            db: Session for the on-completion ``TTSUsage`` write (None ⇒ skip).
            record_usage: Set False to suppress the usage-row write.

        Yields:
            AudioChunk objects from the selected provider.

        Fallback for streaming is best-effort: a candidate that fails *before*
        emitting its first chunk is skipped in favour of the next provider in
        the chain. Once any chunk has been yielded the stream is committed to
        that provider (we cannot un-send bytes), so a later error propagates.

        Billing/quota: quota is enforced by the route BEFORE the first byte (a
        generator cannot raise a clean 429 once streaming has begun). Usage +
        the cost event are recorded HERE, once a provider streams to completion,
        attributed to the provider that actually served — mirroring the sync
        path so a streamed request is neither free nor unattributed.
        """
        _p, resolved_provider, fallback_chain = await self._resolve_provider_chain(
            provider, text, language=language
        )
        logger.info("Streaming from provider=%s text_len=%d", resolved_provider, len(text))

        overall_start = time.monotonic()
        attempts = self._stream_attempt_order(resolved_provider, fallback_chain)
        last_error: Exception | None = None
        for name in attempts:
            # Registration + availability + voice/text resolution in ONE
            # skip-or-proceed decision. They were three separate branches inline
            # and they all mean the same thing here — "this candidate never
            # started, move to the next" — so they read better, and cost the
            # complexity budget less, as one.
            opened = await self._open_stream_candidate(
                name=name, text=text, voice=voice, language=language
            )
            if opened is None:
                continue
            obj, resolved_voice, processed_text = opened

            attempt = _StreamAttempt()
            attempt.processed_text = processed_text
            start = time.monotonic()

            meter_ctx = _StreamMeterContext(
                tenant_id=tenant_id,
                provider=name,
                text=text,
                model=model,
                output_format=output_format,
                started_at=overall_start,
                record_usage=record_usage,
            )

            try:
                async for chunk in self._classify_stream_chunks(
                    self._stream_provider(
                        provider_obj=obj,
                        resolved_provider=name,
                        processed_text=attempt.processed_text,
                        resolved_voice=resolved_voice,
                        speed=speed,
                        output_format=output_format,
                        model=model,
                    ),
                    attempt,
                ):
                    yield chunk
            except PROVIDER_FAILURE_ERRORS as exc:
                self._record_health(name, start, ok=False)
                if attempt.produced:
                    # Bytes already went out on the wire; cannot fall back now.
                    # The delivered audio is real, paid product the client keeps
                    # even though the stream ends in an error — meter it before
                    # re-raising so a mid-stream provider failure is not free (and
                    # trivially repeatable) audio, i.e. a billing bypass. Precise
                    # per-chunk delivered-character tracking is infeasible: chunks
                    # are opaque encoded-audio bytes with no source-character
                    # mapping. So we bill what was COMMITTED to this provider —
                    # the full ``processed_text`` (the same unit the success path
                    # meters). ``processed_text`` is always bound here because
                    # ``produced`` only becomes True inside the chunk loop, after
                    # it is assigned.
                    await self._meter_attempt_once(db, attempt, meter_ctx)
                    raise
                last_error = exc
                logger.warning(
                    "Stream provider '%s' failed before first chunk; trying fallback: %s",
                    name,
                    exc,
                )
                continue
            else:
                # Reached only when ``name`` streamed to completion. A "completion"
                # that carried zero audio bytes is a FAILURE, not a free success:
                # nothing was sent on the wire, so fall back to the next candidate
                # and — critically — do not meter it. Otherwise a provider whose
                # pipeline died on its first segment returned 200/zero bytes and
                # charged the tenant for the whole text.
                #
                # ⚠️ MOVED INTO AN ``else:`` and NOT otherwise changed. It has to
                # sit inside the ``try`` whose ``finally`` is below, because that
                # ``finally`` is what closes the client-abort bypass — if this
                # block stayed after the ``try`` statement, the ``finally`` would
                # run BEFORE it on the success path and meter an attempt the
                # success path was about to meter itself.
                if not attempt.produced:
                    self._record_health(name, start, ok=False)
                    last_error = RuntimeError(
                        f"Stream provider '{name}' completed without producing any audio"
                    )
                    logger.warning(
                        "Stream provider '%s' produced zero audio bytes; trying fallback", name
                    )
                    continue

                # Meter on completion (per-character), attributed to the provider
                # that actually served, then bill.
                if attempt.pending_final is not None:
                    # Stamp the provider that actually served. ``name`` is the
                    # winner of THIS loop's fallback, which may not be the
                    # requested provider (``auto``, or an earlier candidate that
                    # died before its first chunk) — so it is the only place the
                    # truth exists. Mirrors ``meta["resolved_provider"]`` on the
                    # buffered path.
                    attempt.pending_final.provider = name
                    yield attempt.pending_final
                self._record_health(name, start, ok=True)
                await self._meter_attempt_once(db, attempt, meter_ctx)
                return
            finally:
                # ⛔ THE CLIENT-ABORT ARM. Every other exit from this attempt is
                # an EXCEPTION SUBCLASS, and ``PROVIDER_FAILURE_ERRORS`` can only
                # ever list those. A client that hangs up mid-stream does not
                # raise one: starlette cancels the response task, so what arrives
                # at the ``yield`` above is ``asyncio.CancelledError`` or
                # ``GeneratorExit`` — both ``BaseException``, both invisible to
                # the ``except`` clause, and there was no ``finally`` at all.
                #
                # So the delivered audio was FREE. Worse than free: quota is
                # aggregated from the ``TTSUsage`` rows this writes
                # (``api/tts._enforce_stream_quota``), so an aborted stream did
                # not accumulate against the tenant's limit either. `curl
                # --max-time 2 -X POST /api/tts/stream/raw` in a loop is
                # unmetered, unlimited audio — the exact bypass the comment on
                # the ``except`` arm above says it exists to close, only
                # client-controlled rather than provider-driven.
                #
                # Bills the same unit as both other arms (``processed_text``, the
                # text COMMITTED to this provider). Per-chunk delivered-character
                # billing is not available: chunks are opaque encoded-audio bytes
                # with no source-character mapping.
                #
                # ⚠️ ``asyncio.shield`` IS LOAD-BEARING, not caution. On the
                # cancellation path this coroutine is already cancelled, so a
                # bare ``await`` would re-raise ``CancelledError`` at its first
                # suspension point — i.e. the write would be abandoned by the
                # very condition that makes it necessary, and the fix would be
                # inert on its own trigger. Shielded, the metering task runs to
                # completion even as the outer await unwinds.
                #
                # ``except Exception`` and NOT ``BaseException``: a metering
                # failure must never replace the teardown that is already in
                # flight, but a ``CancelledError`` raised by the shielded await
                # must still propagate.
                await self._meter_abandoned_stream(db, attempt, meter_ctx)

        if last_error is not None:
            raise RuntimeError(
                f"All TTS providers failed. Last error: {last_error}"
            ) from last_error
        raise RuntimeError(f"Provider '{resolved_provider}' is not available")

    async def _open_stream_candidate(
        self,
        *,
        name: str,
        text: str,
        voice: str | None,
        language: str | None,
    ) -> tuple[TTSProvider, str, str] | None:
        """Resolve one streaming candidate, or ``None`` to skip to the next.

        ``None`` covers both ways a candidate can fail to START: it is not
        registered, or ``_prepare_stream_attempt`` found it unavailable. Neither
        has produced a byte, so neither is metered and neither is a health event
        — they are simply not this request's provider.
        """
        try:
            provider_obj = self._registry.get_provider(name)
        except KeyError:
            logger.warning("Stream provider '%s' is not registered; skipping", name)
            return None
        prepared = await self._prepare_stream_attempt(
            provider_obj=provider_obj, name=name, text=text, voice=voice, language=language
        )
        if prepared is None:
            return None
        resolved_voice, processed_text = prepared
        return provider_obj, resolved_voice, processed_text

    @staticmethod
    def _stream_attempt_order(resolved_provider: str, fallback_chain: list[str]) -> list[str]:
        """The candidate order for one stream: the resolved provider, then the
        fallback chain with that provider removed so it is never tried twice."""
        return [resolved_provider, *(n for n in fallback_chain if n != resolved_provider)]

    async def _meter_attempt_once(
        self,
        db: AsyncSession | None,
        attempt: _StreamAttempt,
        ctx: _StreamMeterContext,
    ) -> None:
        """Meter ONE stream attempt AT MOST ONCE, whichever way it ends.

        There are three exits from a produced attempt — success, provider
        failure after bytes, and client teardown — and the teardown arm is a
        ``finally``, so it also fires on the other two. Without this latch the
        success path would meter, return through the ``finally``, and meter a
        SECOND time, turning a fix for a billing bypass into a double-bill.
        ``tests/unit/test_billing_fixes.py`` pins the count, not just the call.
        """
        if attempt.metered:
            return
        # Set BEFORE the await, not after: the write can be interrupted, and a
        # latch that only closes on success would let the teardown arm retry a
        # partially-applied charge.
        attempt.metered = True
        await self._meter_completed_stream(
            db,
            tenant_id=ctx.tenant_id,
            provider=ctx.provider,
            text=ctx.text,
            processed_text=attempt.processed_text,
            model=ctx.model,
            output_format=ctx.output_format,
            started_at=ctx.started_at,
            record_usage=ctx.record_usage,
        )

    async def _meter_abandoned_stream(
        self,
        db: AsyncSession | None,
        attempt: _StreamAttempt,
        ctx: _StreamMeterContext,
    ) -> None:
        """Meter a stream torn down from OUTSIDE — the client-abort arm.

        Every other exit from a stream attempt is an ``Exception`` subclass, and
        ``PROVIDER_FAILURE_ERRORS`` can only ever list those. A client that hangs
        up mid-stream raises none of them: starlette cancels the response task,
        so what arrives at the ``yield`` is ``asyncio.CancelledError`` or
        ``GeneratorExit`` — both ``BaseException``, both invisible to that
        ``except``, and there was no ``finally`` at all.

        So delivered audio was FREE. Worse than free: quota is aggregated from
        the ``TTSUsage`` rows this writes (``api/tts._enforce_stream_quota``), so
        an aborted stream did not count against the tenant's limit either.
        ``curl --max-time 2 -X POST /api/tts/stream/raw`` in a loop was
        unmetered, unlimited audio — the exact bypass the produced-then-failed
        arm says it exists to close, only client-controlled rather than
        provider-driven.

        Bills the same unit as both other arms (``processed_text``, the text
        COMMITTED to this provider). Per-chunk delivered-character billing is
        not available: chunks are opaque encoded-audio bytes with no
        source-character mapping.

        ⚠️ ``asyncio.shield`` — AND AN HONEST NOTE ON WHAT IS AND IS NOT PROVEN.
        Under the anyio cancel scopes starlette runs responses in, a cancelled
        scope re-delivers ``CancelledError`` at EVERY subsequent checkpoint, so
        an unshielded ``await`` here would be abandoned by the very condition
        that makes it necessary. Shielded, the write completes.

        ⛔ THE UNIT TESTS BELOW DO NOT PIN THAT. Removing the ``shield`` leaves
        ``test_cancelled_stream_task_still_meters_delivered_audio`` GREEN — bare
        asyncio delivers a cancellation ONCE, so the following ``await``
        proceeds, and the test's ``record_usage`` is an ``AsyncMock`` that never
        truly suspends anyway. That is a fake that cannot exercise the property,
        and it is recorded here rather than dressed up: the shield is retained
        on the anyio reasoning, not on a passing assertion. Do not delete it
        because "no test fails" — no test CAN, at this level.

        ``except Exception`` and NOT ``BaseException``: a metering failure must
        never replace the teardown already in flight, but a ``CancelledError``
        from the shielded await must still propagate.
        """
        if not attempt.produced or attempt.metered:
            return
        try:
            await asyncio.shield(self._meter_attempt_once(db, attempt, ctx))
        except Exception:
            logger.exception(
                "Stream metering failed during teardown (provider=%s chars=%d)",
                ctx.provider,
                len(attempt.processed_text),
            )

    async def _meter_completed_stream(
        self,
        db: AsyncSession | None,
        *,
        tenant_id: str,
        provider: str,
        text: str,
        processed_text: str,
        model: str | None,
        output_format: str,
        started_at: float,
        record_usage: bool,
    ) -> None:
        """Record usage + emit the cost event for a stream that ran to completion.

        Kept off ``generate_stream`` so the success path is a single call: a
        streamed request is metered exactly like the sync path (per-character,
        attributed to the provider that actually served), closing the free +
        unattributed streaming bypass.
        """
        latency_ms = (time.monotonic() - started_at) * 1000
        if record_usage:
            await self._record_stream_usage(
                db,
                tenant_id=tenant_id,
                provider=provider,
                # Bill quota on the SAME text the cost bus bills
                # (``processed_text``, SSML tags stripped for non-SSML
                # providers) so quota and billing agree on an SSML request.
                characters=len(processed_text),
                latency_ms=latency_ms,
            )
        self._emit_cost_event(
            tenant_id=tenant_id,
            provider=provider,
            model=model or provider,
            text=processed_text,
            duration_s=0.0,
            audio_format=output_format,
            latency_ms=int(latency_ms),
            call_kind="stream",
            call_id=get_request_id(),
        )

    @staticmethod
    async def _classify_stream_chunks(
        source: AsyncIterator[AudioChunk],
        attempt: _StreamAttempt,
    ) -> AsyncIterator[AudioChunk]:
        """Forward real audio chunks; record what the attempt actually produced.

        The empty ``is_final=True`` sentinel is a SUCCESS marker, not audio, so
        it is buffered on ``attempt`` rather than forwarded. See
        :class:`_StreamAttempt`.
        """
        async for chunk in source:
            if chunk.is_final and not chunk.data:
                attempt.pending_final = chunk
                continue
            if chunk.data:
                attempt.produced = True
            yield chunk

    async def _prepare_stream_attempt(
        self,
        *,
        provider_obj: TTSProvider,
        name: str,
        text: str,
        voice: str | None,
        language: str | None,
    ) -> tuple[str, str] | None:
        """Resolve (voice, processed_text) for one streaming candidate.

        Returns ``None`` when the provider is unavailable and the caller should
        move on to the next candidate in the fallback chain.
        """
        if not await provider_obj.is_available():
            logger.warning("Stream provider '%s' unavailable; trying next", name)
            return None
        resolved_voice = self._resolve_voice(provider_obj, voice)
        if language is None and voice is None:
            voices = await provider_obj.list_voices()
            _detected_lang, auto_voice = self._detect_language(text, name, voice, voices)
            if auto_voice:
                resolved_voice = auto_voice
        return resolved_voice, self._process_text(text, name)

    async def _stream_provider(
        self,
        *,
        provider_obj: TTSProvider,
        resolved_provider: str,
        processed_text: str,
        resolved_voice: str,
        speed: float,
        output_format: str,
        model: str | None,
    ) -> AsyncIterator[AudioChunk]:
        """Yield chunks from a provider under the configured concurrency semaphores."""
        kwargs: dict[str, object] = {}
        if model:
            kwargs["model"] = model

        provider_sem = self._provider_semaphores.get(resolved_provider)
        async with self._generation_semaphore:
            if provider_sem:
                async with provider_sem:
                    async for chunk in provider_obj.generate_stream(
                        text=processed_text,
                        voice=resolved_voice,
                        speed=speed,
                        output_format=output_format,
                        **kwargs,
                    ):
                        yield chunk
            else:
                async for chunk in provider_obj.generate_stream(
                    text=processed_text,
                    voice=resolved_voice,
                    speed=speed,
                    output_format=output_format,
                    **kwargs,
                ):
                    yield chunk

    # ========================================================================
    # Async Job Generation
    # ========================================================================

    async def generate_async(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        text: str,
        provider: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: str = "mp3",
        model: str | None = None,
        batch_id: str | None = None,
        caller_item_id: str | None = None,
        strategy: str | None = None,
    ) -> str:
        """Create an async generation job and return the job_id.

        Args:
            db: Database session for persisting the job record.
            tenant_id: Tenant identifier for quota/usage isolation.
            text: Text to synthesize.
            provider: Provider name or "auto" for routing.
            voice: Optional voice override.
            speed: Playback speed multiplier.
            output_format: Output audio format.
            model: Optional provider-specific model.
            batch_id: Optional batch identifier for grouping.
            caller_item_id: Caller-assigned item ID within a batch.
            strategy: Optional routing strategy override.

        Returns:
            The unique job_id string.
        """
        # Defer routing to process time so the fallback chain is applied when the
        # job actually runs: store the request's provider verbatim (including the
        # literal "auto") rather than freezing it to one concrete provider here.
        # An explicit provider is still validated up-front so a bad name 404s now.
        if provider != "auto":
            self._registry.get_provider(provider)  # raises KeyError if unregistered

        job = await JobManager.create_job(
            db,
            tenant_id=tenant_id,
            text=text,
            provider=provider,
            voice=voice or "default",
            speed=speed,
            output_format=output_format,
            model=model,
            batch_id=batch_id,
            caller_item_id=caller_item_id,
            routing_strategy=strategy,
        )
        return str(job.job_id)

    async def process_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        post_processing: object | None = None,
    ) -> None:
        """Process a pending TTS job through generation, storage, and usage recording.

        Args:
            db: Database session.
            job_id: The job to process.
            tenant_id: Tenant identifier for quota/usage.
            post_processing: Optional post-processing configuration.

        Raises:
            ValueError: If the job is not found.
        """
        logger.info("Processing job %s for tenant=%s", job_id, tenant_id)
        # Scoped by the tenant this job is being processed FOR: a job id that
        # does not belong to that tenant must not be synthesised, stored, or
        # billed under it.
        job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        await JobManager.mark_processing(db, job_id)
        await db.commit()

        await self._notify("job_started", job_id, tenant_id)

        try:
            result, meta = await self.generate_sync(
                text=job.text,
                provider=job.provider,
                voice=job.voice,
                speed=job.speed,
                output_format=job.output_format,
                model=job.model,
                tenant_id=tenant_id,
                strategy=job.routing_strategy,
                post_processing=post_processing,
                db=db,
                # This method records the same usage below, with job-level
                # attribution (job.text_length / resolved provider). Letting
                # generate_sync record too would double-count every async job.
                record_usage=False,
                # Defer the billing cost event: emit it only after the job is
                # durably marked ``completed`` (below). If ``_storage.store`` or
                # ``mark_completed`` raises after a paid synthesis, we must NOT
                # have already put a charge on the cost bus for a job the tenant
                # sees as failed.
                emit_cost=False,
            )
            # ``job.provider`` may be the literal "auto"; attribute usage to the
            # provider the fallback loop actually used (falls back to the stored
            # value when generation reported none).
            used_provider = meta.get("resolved_provider") or job.provider
            # Bill the async job on the SAME text the sync/stream paths bill:
            # the synthesized (SSML-stripped) text, not the raw ``job.text``.
            # ``job.text_length`` is len(raw) captured at job-creation time,
            # before any provider was resolved, so on an SSML job it overbilled
            # both the usage row and the cost bus — and disagreed with what the
            # identical synchronous request would have charged.
            billed_text = meta.get("billed_text", job.text)

            audio_path = await self._storage.store(job_id, result.audio_data, result.format)

            await JobManager.mark_completed(
                db,
                job_id,
                audio_path=audio_path,
                audio_size_bytes=len(result.audio_data),
                audio_duration_seconds=result.duration_seconds,
                cached=meta.get("cached", False),
                estimated_cost_usd=meta.get("estimated_cost"),
                latency_ms=meta.get("latency_ms"),
                detected_language=meta.get("detected_language"),
            )

            await JobManager.record_usage(
                db,
                tenant_id=tenant_id,
                provider=used_provider,
                characters=len(billed_text),
                audio_seconds=result.duration_seconds or 0.0,
                estimated_cost_usd=meta.get("estimated_cost", 0.0),
                latency_ms=meta.get("latency_ms", 0.0),
                cache_hit=meta.get("cache_hit", False),
            )

            # Check usage thresholds and create alerts if needed
            try:
                from src.services.quota_service import check_and_create_alerts

                await check_and_create_alerts(db, tenant_id)
            except RedisError, SQLAlchemyError, OSError, TimeoutError, ValueError, RuntimeError:
                logger.debug("Alert generation failed", exc_info=True)

            # NOTE: the ``job_completed`` notification is deliberately NOT sent
            # here — see the commit + emit below.

        # ``subprocess.TimeoutExpired`` (SubprocessError, not OSError/TimeoutError)
        # comes out of the ffmpeg post-processing / ffprobe calls; without it a
        # hung binary left this job non-terminal forever. ``RedisError`` and
        # ``SQLAlchemyError`` sit outside every builtin tree too (redis-py's
        # ConnectionError is not the builtin one; psycopg3 errors are wrapped
        # into ``sqlalchemy.exc.DBAPIError``), so a Redis blip or a
        # pgbouncer/Postgres fault used to escape this handler as well.
        except (
            RedisError,
            SQLAlchemyError,
            OSError,
            TimeoutError,
            ValueError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            try:
                await JobManager.mark_failed(db, job_id, str(exc))
                await JobManager.record_usage(
                    db,
                    tenant_id=tenant_id,
                    provider=job.provider,
                    characters=job.text_length,
                    audio_seconds=0.0,
                    is_error=True,
                )
                await db.commit()
            except SQLAlchemyError:
                # This session is poisoned — very likely by the very fault we are
                # handling. Roll it back and let the caller terminalise the job in
                # a FRESH session (``src.api.tts._mark_job_failed``); the ``raise``
                # below still re-raises the ORIGINAL failure.
                logger.exception("Could not record failure for job %s in this session", job_id)
                await db.rollback()
            await self._notify("job_failed", job_id, tenant_id, error=str(exc))
            raise

        await db.commit()

        # Bill ONLY now — the job row is durably ``completed`` and committed.
        # Emitting inside generate_sync (before store/mark_completed) would have
        # charged the tenant on the cost bus even when a late storage/DB failure
        # dropped the job into ``failed`` with $0 recorded usage: a silent
        # overbill. ``used_provider`` attributes the charge to the provider that
        # actually produced audio (never the literal "auto").
        #
        # Skip the cost bus on a cache hit: no provider work occurred, the usage
        # row above already recorded ``estimated_cost=0.0``/``cache_hit=True``,
        # and ``_emit_cost_event`` would otherwise recompute a nonzero charge
        # from ``len(text)`` — a silent overbill for paid providers. The
        # synchronous path returns before its own emit on a cache hit, so this
        # keeps the async job path symmetric with it.
        if not meta.get("cache_hit"):
            self._emit_cost_event(
                tenant_id=tenant_id,
                provider=used_provider,
                model=job.model or used_provider,
                text=billed_text,
                duration_s=result.duration_seconds or 0.0,
                audio_format=result.format,
                latency_ms=int(meta.get("latency_ms") or 0),
                call_kind="job",
                call_id=str(job_id),
            )

        # Announce completion LAST, and best-effort.
        #
        # This notification used to run INSIDE the try above, BEFORE
        # ``await db.commit()``. Because ``redis.exceptions.ConnectionError`` is
        # not the builtin ``ConnectionError``, it escaped ``_notify``,
        # ``publish_event`` and the wrapper's ``publish`` alike, aborted
        # ``process_job`` before the commit, and so rolled back BOTH
        # ``mark_completed`` and ``record_usage`` — then skipped the cost emit —
        # for a job whose audio had already been synthesized (and paid for) and
        # stored. A dead Redis silently un-billed and stranded completed work.
        #
        # A notification must never be able to undo committed work, so it is
        # fired after the durable write and the billing emit, and its failures
        # are swallowed here as well as inside ``_notify``.
        try:
            await self._notify(
                "job_completed",
                job_id,
                tenant_id,
                audio_url=f"/api/tts/jobs/{job_id}/audio",
                duration=result.duration_seconds,
            )
        except RedisError, SQLAlchemyError, OSError, TimeoutError, ValueError, RuntimeError:
            logger.warning(
                "job_completed notification failed for job=%s (job is committed)",
                job_id,
                exc_info=True,
            )

    # ========================================================================
    # Batch Concatenation
    # ========================================================================

    async def concatenate_batch(
        self,
        db: AsyncSession,
        batch_id: str,
        output_format: str = "mp3",
        gap_ms: int = 0,
        strict: bool = True,
        *,
        tenant_id: str,
    ) -> tuple[str, float | None]:
        """Concatenate all completed batch job audio into a single file.

        Args:
            strict: If True, raise ValueError when any completed job's audio
                    cannot be retrieved (default). If False, skip missing segments
                    with warnings.
            tenant_id: Owner of the batch. Required and keyword-only — the batch
                    read is tenant-scoped, so a foreign ``batch_id`` yields no
                    jobs rather than concatenating someone else's audio into a
                    file this tenant can then download.

        Returns (storage_path, total_duration); the duration is ``None`` when
        ffprobe could not measure the concatenated file.
        """
        from src.utils.audio import concatenate_audio, probe_duration

        jobs = await JobManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)

        failed_jobs = [j for j in jobs if j.status == "failed"]
        for j in failed_jobs:
            logger.warning(
                "Batch %s: job %s failed (error: %s), excluding from concatenation",
                batch_id,
                j.job_id,
                j.error_message,
            )

        completed_jobs = [j for j in jobs if j.status == "completed" and j.audio_path]

        if not completed_jobs:
            raise ValueError(f"No completed jobs in batch {batch_id}")

        # Retrieve audio in order, tracking each segment's REAL format. The
        # storage key is ``audio/{job_id}.{produced_format}``, so its extension —
        # not ``job.output_format`` — says what the bytes actually are. A
        # provider that cannot honour the requested format (ElevenLabs only emits
        # mp3) otherwise had its MP3 written into a ``.wav`` temp file and
        # stream-copied, producing a corrupt concatenation.
        from pathlib import PurePosixPath

        audio_segments: list[bytes] = []
        segment_formats: list[str] = []
        for job in completed_jobs:
            audio_path = job.audio_path
            if not audio_path:
                continue
            data = await self._storage.retrieve(audio_path)
            if data:
                suffix = PurePosixPath(audio_path).suffix.lstrip(".").lower()
                audio_segments.append(data)
                segment_formats.append(suffix or job.output_format)
            else:
                logger.warning(
                    "Batch %s: could not retrieve audio for completed job %s (path: %s)",
                    batch_id,
                    job.job_id,
                    job.audio_path,
                )
                if strict:
                    raise ValueError(
                        f"Batch {batch_id}: audio retrieval failed for completed job {job.job_id}"
                    )

        if len(audio_segments) < len(completed_jobs):
            logger.warning(
                "Batch %s: %d of %d completed segments could not be retrieved",
                batch_id,
                len(completed_jobs) - len(audio_segments),
                len(completed_jobs),
            )

        if not audio_segments:
            raise ValueError("No audio data found for batch concatenation")

        # Concatenate (runs ffmpeg in thread)
        concatenated = await asyncio.to_thread(
            concatenate_audio, audio_segments, output_format, gap_ms, segment_formats
        )

        # Store concatenated audio
        concat_key = f"{batch_id}_concatenated"
        concat_path = await self._storage.store(concat_key, concatenated, output_format)

        # Get duration via ffprobe
        import tempfile
        from pathlib import Path

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(concatenated)
                tmp.flush()
            duration = await asyncio.to_thread(probe_duration, tmp_path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        # Update batch jobs with concatenated path
        await JobManager.set_batch_concat_path(db, batch_id, concat_path, tenant_id=tenant_id)
        await db.commit()

        return concat_path, duration
