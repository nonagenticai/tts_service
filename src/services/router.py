"""Intelligent multi-provider routing with fallback support."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from src.api.schemas import RoutingStrategy
from src.providers.base import PROVIDER_FAILURE_ERRORS, GenerationResult, TTSProvider
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

# Default strategy weights: {cost, quality, latency, health}
STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "cost": {"cost": 0.50, "quality": 0.15, "latency": 0.15, "health": 0.20},
    "quality": {"cost": 0.05, "quality": 0.50, "latency": 0.15, "health": 0.30},
    "latency": {"cost": 0.10, "quality": 0.10, "latency": 0.50, "health": 0.30},
    "balanced": {"cost": 0.25, "quality": 0.25, "latency": 0.25, "health": 0.25},
}

DEFAULT_QUALITY_SCORES: dict[str, float] = {
    "kokoro": 0.80,
    "kitten": 0.75,
    "openai": 0.90,
    "elevenlabs": 0.95,
}

DEFAULT_COST_PER_1K: dict[str, float] = {
    "kokoro": 0.0,
    "kitten": 0.0,
    "openai": 0.015,
    "elevenlabs": 0.030,
}


@dataclass
class HealthRecord:
    """Single health observation."""

    ok: bool
    latency_ms: float
    ts: float


@dataclass
class ProviderScore:
    """Composite score for ranking a provider."""

    provider_name: str
    cost_score: float
    quality_score: float
    latency_score: float
    health_score: float
    composite: float = 0.0


@dataclass
class RoutingDecision:
    """Result of provider selection."""

    provider_name: str
    voice: str
    strategy_used: RoutingStrategy
    reason: str
    fallback_chain: list[str]
    estimated_cost_usd: float | None = None


class ProviderHealthTracker:
    """Tracks provider health using in-memory sliding window."""

    WINDOW_SECONDS = 600  # 10-minute window

    def __init__(self) -> None:
        self._records: dict[str, deque[HealthRecord]] = {}

    def record_success(self, provider: str, latency_ms: float) -> None:
        """Record a successful generation for the provider."""
        self._ensure_provider(provider)
        self._records[provider].append(
            HealthRecord(ok=True, latency_ms=latency_ms, ts=time.monotonic())
        )
        self._evict_old(provider)

    def record_failure(self, provider: str, latency_ms: float) -> None:
        """Record a failed generation for the provider."""
        self._ensure_provider(provider)
        self._records[provider].append(
            HealthRecord(ok=False, latency_ms=latency_ms, ts=time.monotonic())
        )
        self._evict_old(provider)

    def get_success_rate(self, provider: str) -> float:
        """Return the success rate (0.0-1.0) for a provider within the sliding window."""
        self._evict_old(provider)
        records = self._records.get(provider)
        if not records:
            return 1.0  # No data = assume healthy
        total = len(records)
        successes = sum(1 for r in records if r.ok)
        return successes / total

    def get_p50_latency_ms(self, provider: str) -> float:
        """Return the median (p50) latency for successful calls within the window."""
        self._evict_old(provider)
        records = self._records.get(provider)
        if not records:
            return 500.0  # Default assumption
        latencies = sorted(r.latency_ms for r in records if r.ok)
        if not latencies:
            return 5000.0
        idx = len(latencies) // 2
        return latencies[idx]

    def _ensure_provider(self, provider: str) -> None:
        if provider not in self._records:
            self._records[provider] = deque(maxlen=1000)

    def _evict_old(self, provider: str) -> None:
        records = self._records.get(provider)
        if not records:
            return
        cutoff = time.monotonic() - self.WINDOW_SECONDS
        while records and records[0].ts < cutoff:
            records.popleft()


class CostEstimator:
    """Estimates generation cost per provider."""

    def __init__(self, pricing: dict[str, float] | None = None) -> None:
        self._pricing = pricing or dict(DEFAULT_COST_PER_1K)

    def estimate(self, provider: str, text_length: int) -> float:
        """Estimate generation cost in USD for the given text length.

        Args:
            provider: Provider name.
            text_length: Character count of input text.

        Returns:
            Estimated cost in USD.
        """
        rate = self._pricing.get(provider, 0.0)
        return (text_length / 1000.0) * rate

    def get_rate(self, provider: str) -> float:
        """Return the cost-per-1K-chars rate for a provider."""
        return self._pricing.get(provider, 0.0)

    def get_all_rates(self) -> dict[str, float]:
        """Return a copy of all provider pricing rates."""
        return dict(self._pricing)


class ProviderRouter:
    """Selects best TTS provider based on strategy, health, and cost."""

    def __init__(
        self,
        registry: ProviderRegistry,
        health_tracker: ProviderHealthTracker | None = None,
        cost_estimator: CostEstimator | None = None,
        quality_scores: dict[str, float] | None = None,
        default_strategy: str = "balanced",
        max_fallback_attempts: int = 2,
    ) -> None:
        self._registry = registry
        self._health = health_tracker or ProviderHealthTracker()
        self._cost = cost_estimator or CostEstimator()
        self._quality_scores = quality_scores or dict(DEFAULT_QUALITY_SCORES)
        self.default_strategy = default_strategy
        self._max_fallback = max_fallback_attempts

    @staticmethod
    def _routable_providers(providers: list[TTSProvider]) -> list[TTSProvider]:
        """Drop providers that must not be picked implicitly (the silent stub).

        A provider with ``implicit_selection = False`` does not produce real
        speech, so it may never win an ``auto`` decision nor sit in a fallback
        chain — that path returns HTTP 200 with silence and nothing downstream
        can tell it apart from real narration.

        The exemption is deliberately keyed on what is REGISTERED, not on what is
        currently available: a deployment that registered only the stub
        (``TTS_PROVIDER=stub`` / ``STUB_ENABLED`` alone) explicitly asked for it,
        while "the real provider happens to be down right now" must surface as a
        routing failure, not a silent degrade to silence.
        """
        real = [p for p in providers if p.implicit_selection]
        if real:
            for p in providers:
                if not p.implicit_selection:
                    logger.debug(
                        "Excluding provider '%s' from implicit routing (not a real synthesizer)",
                        p.name,
                    )
            return real
        if providers:
            logger.warning(
                "Only non-synthesizing providers (%s) are registered; routing to one of them. "
                "Output will NOT contain real speech.",
                ", ".join(p.name for p in providers),
            )
        return providers

    async def select_provider(
        self,
        text: str,
        language: str | None = None,
        strategy: RoutingStrategy | None = None,
        voice: str | None = None,
    ) -> RoutingDecision:
        """Select the best provider based on strategy."""
        strat = strategy or RoutingStrategy(self.default_strategy)
        weights = STRATEGY_WEIGHTS.get(strat.value, STRATEGY_WEIGHTS["balanced"])

        # Get candidates. Providers that may not be picked implicitly (the silent
        # stub) are removed up-front so they can neither win nor land in the
        # fallback chain.
        providers = self._routable_providers(self._registry.list_providers())
        if not providers:
            raise RuntimeError("No available providers match the request requirements")
        candidates: list[tuple[TTSProvider, ProviderScore]] = []

        max_cost = max(self._cost.get_rate(p.name) for p in providers) or 0.001

        for p in providers:
            if not await p.is_available():
                continue
            if len(text) > p.max_text_length:
                continue

            # Language filter
            if language:
                voices = await p.list_voices()
                langs = {v.language for v in voices}
                # Also check language prefix match (e.g., "en" matches "en-US")
                if not any(
                    language == lang
                    or language.startswith(lang.split("-")[0])
                    or lang.startswith(language)
                    for lang in langs
                ):
                    continue

            # Score
            cost_rate = self._cost.get_rate(p.name)
            cost_score = 1.0 - (cost_rate / max_cost) if max_cost > 0 else 1.0
            quality_score = self._quality_scores.get(p.name, 0.7)
            success_rate = self._health.get_success_rate(p.name)
            p50 = self._health.get_p50_latency_ms(p.name)
            latency_score = max(0.0, 1.0 - min(p50 / 5000.0, 1.0))
            health_score = success_rate

            # Local providers get latency bonus
            if p.provider_type == "local":
                latency_score = min(1.0, latency_score + 0.2)

            composite = (
                weights["cost"] * cost_score
                + weights["quality"] * quality_score
                + weights["latency"] * latency_score
                + weights["health"] * health_score
            )

            score = ProviderScore(
                provider_name=p.name,
                cost_score=cost_score,
                quality_score=quality_score,
                latency_score=latency_score,
                health_score=health_score,
                composite=composite,
            )
            candidates.append((p, score))

        if not candidates:
            raise RuntimeError("No available providers match the request requirements")

        # Sort by composite score descending
        candidates.sort(key=lambda c: c[1].composite, reverse=True)

        best_provider, best_score = candidates[0]
        fallback_chain = [c[0].name for c in candidates[1:]]

        # Resolve voice
        effective_voice = voice or best_provider.get_default_voice()
        est_cost = self._cost.estimate(best_provider.name, len(text))

        reason = (
            f"{strat.value} strategy: {best_provider.name} scored "
            f"{best_score.composite:.3f} (cost={best_score.cost_score:.2f}, "
            f"quality={best_score.quality_score:.2f}, "
            f"latency={best_score.latency_score:.2f}, "
            f"health={best_score.health_score:.2f})"
        )

        logger.info(
            "Routing decision: strategy=%s provider=%s score=%.3f fallbacks=%s",
            strat.value,
            best_provider.name,
            best_score.composite,
            fallback_chain[: self._max_fallback],
        )

        return RoutingDecision(
            provider_name=best_provider.name,
            voice=effective_voice,
            strategy_used=strat,
            reason=reason,
            fallback_chain=fallback_chain[: self._max_fallback],
            estimated_cost_usd=est_cost,
        )

    async def execute_with_fallback(
        self,
        decision: RoutingDecision,
        text: str,
        voice: str | None,
        speed: float,
        output_format: str,
        model: str | None = None,
        **kwargs: object,
    ) -> tuple[GenerationResult, str]:
        """Execute generation with automatic fallback on failure."""
        providers_to_try = [decision.provider_name, *decision.fallback_chain]

        last_error: Exception | None = None
        for provider_name in providers_to_try:
            start = time.monotonic()
            try:
                p = self._registry.get_provider(provider_name)
                if not await p.is_available():
                    continue
                effective_voice = (
                    voice if provider_name == decision.provider_name else p.get_default_voice()
                )
                gen_kwargs: dict[str, object] = {}
                if model:
                    gen_kwargs["model"] = model
                result = await p.generate(
                    text=text,
                    voice=effective_voice or p.get_default_voice(),
                    speed=speed,
                    output_format=output_format,
                    **gen_kwargs,
                )
                elapsed = (time.monotonic() - start) * 1000
                self._health.record_success(provider_name, elapsed)
                return result, provider_name
            except PROVIDER_FAILURE_ERRORS as exc:
                elapsed = (time.monotonic() - start) * 1000
                self._health.record_failure(provider_name, elapsed)
                last_error = exc
                logger.warning("Provider '%s' failed, trying fallback: %s", provider_name, exc)

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def record_result(self, provider: str, latency_ms: float, *, ok: bool) -> None:
        """Feed a generation outcome into the health tracker.

        Lets a caller that runs its own generation loop (e.g. ``TTSService``,
        which must apply concurrency semaphores + SSML per provider) keep the
        router's health-based scoring informed, so a provider that just failed
        is de-prioritised on the next ``select_provider`` call.
        """
        if ok:
            self._health.record_success(provider, latency_ms)
        else:
            self._health.record_failure(provider, latency_ms)
