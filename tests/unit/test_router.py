"""Unit tests for the provider router service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas import RoutingStrategy
from src.providers.base import GenerationResult, VoiceMetadata
from src.services.router import (
    STRATEGY_WEIGHTS,
    CostEstimator,
    ProviderHealthTracker,
    ProviderRouter,
    ProviderScore,
    RoutingDecision,
)

# ============================================================================
# ProviderHealthTracker
# ============================================================================


class TestProviderHealthTracker:
    def test_no_data_returns_healthy(self):
        tracker = ProviderHealthTracker()
        assert tracker.get_success_rate("unknown") == 1.0

    def test_no_data_default_latency(self):
        tracker = ProviderHealthTracker()
        assert tracker.get_p50_latency_ms("unknown") == 500.0

    def test_record_success(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("kokoro", 100.0)
        assert tracker.get_success_rate("kokoro") == 1.0

    def test_record_failure(self):
        tracker = ProviderHealthTracker()
        tracker.record_failure("kokoro", 200.0)
        assert tracker.get_success_rate("kokoro") == 0.0

    def test_mixed_success_rate(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("kokoro", 100.0)
        tracker.record_success("kokoro", 120.0)
        tracker.record_failure("kokoro", 300.0)
        # 2 success out of 3
        rate = tracker.get_success_rate("kokoro")
        assert abs(rate - 2.0 / 3.0) < 0.01

    def test_p50_latency_with_successes(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("kokoro", 100.0)
        tracker.record_success("kokoro", 200.0)
        tracker.record_success("kokoro", 300.0)
        p50 = tracker.get_p50_latency_ms("kokoro")
        assert p50 == 200.0

    def test_p50_latency_no_successes(self):
        tracker = ProviderHealthTracker()
        tracker.record_failure("kokoro", 100.0)
        tracker.record_failure("kokoro", 200.0)
        p50 = tracker.get_p50_latency_ms("kokoro")
        # No successful latencies, returns high default
        assert p50 == 5000.0

    @patch("time.monotonic")
    def test_eviction_of_old_records(self, mock_monotonic):
        tracker = ProviderHealthTracker()
        # Record at t=0
        mock_monotonic.return_value = 0.0
        tracker.record_success("kokoro", 100.0)
        # Advance past window (600 seconds)
        mock_monotonic.return_value = 700.0
        # Should evict old record
        rate = tracker.get_success_rate("kokoro")
        assert rate == 1.0  # No data after eviction = assume healthy


# ============================================================================
# CostEstimator
# ============================================================================


class TestCostEstimator:
    def test_default_pricing(self):
        estimator = CostEstimator()
        # Free providers
        assert estimator.estimate("kokoro", 1000) == 0.0
        assert estimator.estimate("kitten", 1000) == 0.0
        # Cloud providers
        assert estimator.estimate("openai", 1000) == 0.015
        assert estimator.estimate("elevenlabs", 1000) == 0.030

    def test_custom_pricing(self):
        estimator = CostEstimator({"custom": 0.05})
        assert estimator.estimate("custom", 2000) == 0.1

    def test_unknown_provider_zero_cost(self):
        estimator = CostEstimator()
        assert estimator.estimate("nonexistent", 1000) == 0.0

    def test_get_rate(self):
        estimator = CostEstimator()
        assert estimator.get_rate("openai") == 0.015
        assert estimator.get_rate("unknown") == 0.0

    def test_get_all_rates(self):
        pricing = {"a": 0.01, "b": 0.02}
        estimator = CostEstimator(pricing)
        rates = estimator.get_all_rates()
        assert rates == {"a": 0.01, "b": 0.02}
        # Ensure it's a copy
        rates["c"] = 0.03
        assert "c" not in estimator.get_all_rates()


# ============================================================================
# ProviderScore
# ============================================================================


class TestProviderScore:
    def test_score_creation(self):
        score = ProviderScore(
            provider_name="kokoro",
            cost_score=1.0,
            quality_score=0.8,
            latency_score=0.9,
            health_score=1.0,
            composite=0.925,
        )
        assert score.provider_name == "kokoro"
        assert score.composite == 0.925


# ============================================================================
# RoutingDecision
# ============================================================================


class TestRoutingDecision:
    def test_decision_creation(self):
        decision = RoutingDecision(
            provider_name="kokoro",
            voice="af_heart",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=["openai"],
            estimated_cost_usd=0.0,
        )
        assert decision.provider_name == "kokoro"
        assert decision.fallback_chain == ["openai"]


# ============================================================================
# ProviderRouter
# ============================================================================


class TestProviderRouter:
    def _make_mock_provider(
        self, name, available=True, provider_type="local", max_text_length=10000
    ):
        p = AsyncMock()
        p.name = name
        p.display_name = name.title()
        p.provider_type = provider_type
        p.max_text_length = max_text_length
        p.is_available = AsyncMock(return_value=available)
        p.get_default_voice = MagicMock(return_value="default_voice")
        p.list_voices = AsyncMock(
            return_value=[VoiceMetadata(id="v1", display_name="V1", language="en")]
        )
        p.sample_rate = 24000
        return p

    @pytest.mark.asyncio
    async def test_select_provider_balanced(self):
        registry = MagicMock()
        kokoro = self._make_mock_provider("kokoro")
        openai = self._make_mock_provider("openai", provider_type="cloud")
        registry.list_providers.return_value = [kokoro, openai]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello", strategy=RoutingStrategy.BALANCED)

        assert decision.provider_name in ("kokoro", "openai")
        assert decision.strategy_used == RoutingStrategy.BALANCED
        assert isinstance(decision.reason, str)
        assert isinstance(decision.fallback_chain, list)

    @pytest.mark.asyncio
    async def test_select_provider_cost_prefers_free(self):
        registry = MagicMock()
        kokoro = self._make_mock_provider("kokoro")
        openai = self._make_mock_provider("openai", provider_type="cloud")
        registry.list_providers.return_value = [kokoro, openai]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello", strategy=RoutingStrategy.COST)

        # Free provider should rank higher on cost strategy
        assert decision.provider_name == "kokoro"

    @pytest.mark.asyncio
    async def test_select_provider_quality_prefers_cloud(self):
        registry = MagicMock()
        kokoro = self._make_mock_provider("kokoro")
        elevenlabs = self._make_mock_provider("elevenlabs", provider_type="cloud")
        registry.list_providers.return_value = [kokoro, elevenlabs]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello", strategy=RoutingStrategy.QUALITY)

        # ElevenLabs has highest quality score by default
        assert decision.provider_name == "elevenlabs"

    @pytest.mark.asyncio
    async def test_select_provider_no_candidates_raises(self):
        registry = MagicMock()
        unavailable = self._make_mock_provider("kokoro", available=False)
        registry.list_providers.return_value = [unavailable]

        router = ProviderRouter(registry=registry)
        with pytest.raises(RuntimeError, match="No available providers"):
            await router.select_provider(text="Hello")

    @pytest.mark.asyncio
    async def test_select_provider_text_too_long_excluded(self):
        registry = MagicMock()
        short_provider = self._make_mock_provider("short", max_text_length=5)
        long_provider = self._make_mock_provider("long", max_text_length=10000)
        registry.list_providers.return_value = [short_provider, long_provider]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello world, this is a longer text")

        assert decision.provider_name == "long"

    @pytest.mark.asyncio
    async def test_select_provider_with_language_filter(self):
        registry = MagicMock()
        en_provider = self._make_mock_provider("en_only")
        en_provider.list_voices = AsyncMock(
            return_value=[VoiceMetadata(id="v1", display_name="V1", language="en")]
        )
        fr_provider = self._make_mock_provider("fr_support")
        fr_provider.list_voices = AsyncMock(
            return_value=[VoiceMetadata(id="v2", display_name="V2", language="fr")]
        )
        registry.list_providers.return_value = [en_provider, fr_provider]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Bonjour", language="fr")

        assert decision.provider_name == "fr_support"

    @pytest.mark.asyncio
    async def test_select_provider_default_strategy(self):
        registry = MagicMock()
        kokoro = self._make_mock_provider("kokoro")
        registry.list_providers.return_value = [kokoro]

        router = ProviderRouter(registry=registry, default_strategy="cost")
        decision = await router.select_provider(text="Hello")

        assert decision.strategy_used == RoutingStrategy.COST

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self):
        registry = MagicMock()
        provider = self._make_mock_provider("kokoro")
        expected_result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        provider.generate = AsyncMock(return_value=expected_result)
        registry.get_provider.return_value = provider

        router = ProviderRouter(registry=registry)
        decision = RoutingDecision(
            provider_name="kokoro",
            voice="af_heart",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=[],
        )

        result, used_provider = await router.execute_with_fallback(
            decision, text="Hello", voice="af_heart", speed=1.0, output_format="mp3"
        )
        assert result.audio_data == b"audio"
        assert used_provider == "kokoro"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_uses_fallback(self):
        registry = MagicMock()

        primary = self._make_mock_provider("kokoro")
        primary.generate = AsyncMock(side_effect=RuntimeError("boom"))
        fallback = self._make_mock_provider("openai")
        expected_result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        fallback.generate = AsyncMock(return_value=expected_result)

        registry.get_provider.side_effect = lambda name: primary if name == "kokoro" else fallback

        router = ProviderRouter(registry=registry)
        decision = RoutingDecision(
            provider_name="kokoro",
            voice="af_heart",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=["openai"],
        )

        result, used_provider = await router.execute_with_fallback(
            decision, text="Hello", voice="af_heart", speed=1.0, output_format="mp3"
        )
        assert result.audio_data == b"audio"
        assert used_provider == "openai"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_all_fail(self):
        registry = MagicMock()
        failing = self._make_mock_provider("kokoro")
        failing.generate = AsyncMock(side_effect=RuntimeError("boom"))
        registry.get_provider.return_value = failing

        router = ProviderRouter(registry=registry)
        decision = RoutingDecision(
            provider_name="kokoro",
            voice="af_heart",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=[],
        )

        with pytest.raises(RuntimeError, match="All providers failed"):
            await router.execute_with_fallback(
                decision, text="Hello", voice="af_heart", speed=1.0, output_format="mp3"
            )


class TestStrategyWeights:
    def test_all_strategies_defined(self):
        assert "cost" in STRATEGY_WEIGHTS
        assert "quality" in STRATEGY_WEIGHTS
        assert "latency" in STRATEGY_WEIGHTS
        assert "balanced" in STRATEGY_WEIGHTS

    def test_weights_sum_to_one(self):
        for strategy, weights in STRATEGY_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"{strategy} weights sum to {total}"


# ============================================================================
# B2 — the silent stub must never win an implicit routing decision
# ============================================================================


class TestStubIsNotAnImplicitFallback:
    """A real request that fell through to the stub returned HTTP 200 with
    silence — indistinguishable downstream from real narration."""

    def _mock(self, name, *, implicit=True, provider_type="local"):
        p = AsyncMock()
        p.name = name
        p.display_name = name.title()
        p.provider_type = provider_type
        p.max_text_length = 10000
        p.is_available = AsyncMock(return_value=True)
        p.get_default_voice = MagicMock(return_value="default_voice")
        p.list_voices = AsyncMock(
            return_value=[VoiceMetadata(id="v1", display_name="V1", language="en")]
        )
        p.sample_rate = 24000
        p.implicit_selection = implicit
        return p

    def test_real_stub_declares_itself_non_implicit(self):
        from src.providers.kokoro_provider import KokoroProvider
        from src.providers.stub_provider import StubProvider

        assert StubProvider.implicit_selection is False
        # Real synthesizers keep the permissive default.
        assert KokoroProvider.implicit_selection is True

    @pytest.mark.asyncio
    async def test_stub_is_not_selected_when_a_real_provider_exists(self):
        registry = MagicMock()
        stub = self._mock("stub", implicit=False)
        kokoro = self._mock("kokoro")
        registry.list_providers.return_value = [stub, kokoro]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello")

        assert decision.provider_name == "kokoro"
        assert "stub" not in decision.fallback_chain

    @pytest.mark.asyncio
    async def test_stub_is_absent_from_the_fallback_chain_of_every_strategy(self):
        registry = MagicMock()
        stub = self._mock("stub", implicit=False)
        registry.list_providers.return_value = [
            stub,
            self._mock("kokoro"),
            self._mock("openai", provider_type="cloud"),
            self._mock("elevenlabs", provider_type="cloud"),
        ]

        router = ProviderRouter(registry=registry, max_fallback_attempts=10)
        for strategy in RoutingStrategy:
            decision = await router.select_provider(text="Hello", strategy=strategy)
            assert decision.provider_name != "stub"
            assert "stub" not in decision.fallback_chain

    @pytest.mark.asyncio
    async def test_stub_still_wins_when_it_is_the_only_provider_configured(self):
        """``TTS_PROVIDER=stub`` / stub-only registry IS the explicit config."""
        registry = MagicMock()
        registry.list_providers.return_value = [self._mock("stub", implicit=False)]

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello")

        assert decision.provider_name == "stub"

    @pytest.mark.asyncio
    async def test_stub_not_used_when_the_only_real_provider_is_unavailable(self):
        """An unavailable real provider must surface as a routing FAILURE, not a
        silent degrade to the stub.

        The stub exemption is keyed on what is REGISTERED, not on what happens to
        be available right now — otherwise every kokoro outage would quietly
        start shipping silence.
        """
        registry = MagicMock()
        stub = self._mock("stub", implicit=False)
        kokoro = self._mock("kokoro")
        kokoro.is_available = AsyncMock(return_value=False)
        registry.list_providers.return_value = [stub, kokoro]

        router = ProviderRouter(registry=registry)
        with pytest.raises(RuntimeError, match="No available providers"):
            await router.select_provider(text="Hello")

    @pytest.mark.asyncio
    async def test_execute_with_fallback_never_reaches_the_stub_implicitly(self):
        registry = MagicMock()
        stub = self._mock("stub", implicit=False)
        kokoro = self._mock("kokoro")
        registry.list_providers.return_value = [stub, kokoro]
        registry.get_provider.side_effect = lambda n: {"stub": stub, "kokoro": kokoro}[n]

        kokoro.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"real", format="mp3", sample_rate=24000, duration_seconds=1.0
            )
        )

        router = ProviderRouter(registry=registry)
        decision = await router.select_provider(text="Hello")
        _result, used = await router.execute_with_fallback(
            decision, text="Hello", voice=None, speed=1.0, output_format="mp3"
        )

        assert used == "kokoro"
        stub.generate.assert_not_called()
