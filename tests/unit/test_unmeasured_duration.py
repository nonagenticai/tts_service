"""An UNMEASURED duration must never be indistinguishable from a real 0.0.

``probe_duration`` used to fail open to ``0.0`` on every failure mode (ffprobe
missing, non-zero exit, unparseable JSON, no ``duration`` key), and two cloud
providers re-defaulted to ``0.0`` on top of it. The fabricated value then
travelled to the API boundary as ``X-Audio-Duration: 0.0`` on a 200 with real
MP3 bytes attached, and — worse — into the Redis metadata cache, where a SINGLE
failed probe poisoned every later cache hit for that key for the whole TTL.

The repo already has a house shape for an unmeasured measurement: ``float |
None`` (``AudioMetrics.duration_seconds``, ``schemas.TTSResponse``) and, for
dict payloads, OMITTING the key rather than substituting a number
(``TTSService._measure_audio_metrics`` omits ``meta["audio_metrics"]``). These
tests pin that shape onto the duration path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.providers.base import GenerationResult
from src.utils.audio import probe_duration

# ============================================================================
# probe_duration: unmeasured is None, a real zero is still 0.0
# ============================================================================


class TestProbeDurationReportsUnmeasured:
    @patch("subprocess.run")
    def test_measured_duration_still_returned(self, mock_run):
        """Positive control: a successful probe must keep returning the float."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"format": {"duration": "5.123"}}')
        assert probe_duration(Path("/tmp/test.mp3")) == pytest.approx(5.123)

    @patch("subprocess.run")
    def test_a_real_zero_is_still_zero(self, mock_run):
        """A genuinely zero-length file measures 0.0 — that is DATA, not a gap."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"format": {"duration": "0"}}')
        assert probe_duration(Path("/tmp/test.mp3")) == 0.0

    @patch("subprocess.run")
    def test_nonzero_exit_is_unmeasured(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert probe_duration(Path("/tmp/test.mp3")) is None

    @patch("subprocess.run")
    def test_unparseable_json_is_unmeasured(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")
        assert probe_duration(Path("/tmp/test.mp3")) is None

    @patch("subprocess.run")
    def test_missing_duration_key_is_unmeasured(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout='{"format": {}}')
        assert probe_duration(Path("/tmp/test.mp3")) is None


# ============================================================================
# The two cloud providers must PROPAGATE the gap, not re-default it
# ============================================================================


def _mp3_transport(provider, body: bytes = b"\xff\xfb\x90real_mpeg_bytes") -> None:
    """Point *provider*'s httpx client at a stub returning real MP3 bytes."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "audio/mpeg"})

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestCloudProvidersPropagateTheGap:
    @pytest.mark.asyncio
    async def test_openai_unmeasured_duration_is_none_and_audio_survives(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")
        _mp3_transport(provider)
        with patch("src.providers.openai_provider.probe_duration", return_value=None):
            result = await provider.generate("Hello", "alloy", output_format="mp3")

        # The MP3 is real and valuable: it must NOT be dropped.
        assert result.audio_data == b"\xff\xfb\x90real_mpeg_bytes"
        assert result.duration_seconds is None

    @pytest.mark.asyncio
    async def test_openai_probe_exception_is_unmeasured_not_zero(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")
        _mp3_transport(provider)
        with patch(
            "src.providers.openai_provider.probe_duration", side_effect=OSError("no ffprobe")
        ):
            result = await provider.generate("Hello", "alloy", output_format="mp3")

        assert result.audio_data == b"\xff\xfb\x90real_mpeg_bytes"
        assert result.duration_seconds is None

    @pytest.mark.asyncio
    async def test_elevenlabs_unmeasured_duration_is_none_and_audio_survives(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="el-test-key")
        _mp3_transport(provider)
        with patch("src.providers.elevenlabs_provider.probe_duration", return_value=None):
            result = await provider.generate("Hello", "Rachel", output_format="mp3")

        assert result.audio_data == b"\xff\xfb\x90real_mpeg_bytes"
        assert result.duration_seconds is None

    @pytest.mark.asyncio
    async def test_elevenlabs_probe_exception_is_unmeasured_not_zero(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="el-test-key")
        _mp3_transport(provider)
        with patch(
            "src.providers.elevenlabs_provider.probe_duration", side_effect=OSError("no ffprobe")
        ):
            result = await provider.generate("Hello", "Rachel", output_format="mp3")

        assert result.audio_data == b"\xff\xfb\x90real_mpeg_bytes"
        assert result.duration_seconds is None


# ============================================================================
# API boundary: the header is OMITTED, never fabricated
# ============================================================================


class TestAudioResponseHeaderOmitsAnUnmeasuredDuration:
    def _response(self, duration):
        from src.api.tts import _build_audio_response

        result = GenerationResult(
            audio_data=b"\xff\xfb\x90audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=duration,
        )
        return _build_audio_response(result, {"resolved_provider": "openai"}, "openai")

    def test_measured_duration_is_reported(self):
        assert self._response(3.5).headers["X-Audio-Duration"] == "3.5"

    def test_a_real_zero_is_reported(self):
        """A measured zero is still a measurement and must be on the wire."""
        assert self._response(0.0).headers["X-Audio-Duration"] == "0.0"

    def test_unmeasured_duration_omits_the_header(self):
        resp = self._response(None)
        assert "X-Audio-Duration" not in resp.headers
        # The audio itself is untouched — we degrade the CLAIM, not the product.
        assert resp.body == b"\xff\xfb\x90audio"

    def test_job_audio_download_omits_the_header_when_unmeasured(self):
        from src.api.tts import _audio_duration_headers

        assert _audio_duration_headers(None) == {}
        assert _audio_duration_headers(0.0) == {"X-Audio-Duration": "0.0"}
        assert _audio_duration_headers(2.5) == {"X-Audio-Duration": "2.5"}


# ============================================================================
# ⚠️ The cache: an unmeasured duration must never be PERSISTED as measured
# ============================================================================


class TestUnmeasuredDurationNeverEntersTheCache:
    def _service(self):
        from src.services.tts_service import TTSService

        svc = TTSService.__new__(TTSService)
        svc._redis = AsyncMock()
        svc._redis.set = AsyncMock()
        settings = MagicMock()
        settings.cache.enabled = True
        settings.cache.key_prefix = "tts:cache:"
        settings.cache.default_ttl = 3600
        svc._settings = settings
        svc._storage = AsyncMock()
        return svc

    @pytest.mark.asyncio
    async def test_store_cache_omits_an_unmeasured_duration(self):
        svc = self._service()
        await svc._store_cache("key1", "audio/x.mp3", None, "mp3")

        payload = svc._redis.set.await_args.args[1]
        assert "duration" not in payload, (
            "an unmeasured duration was written to Redis as though measured — "
            "one failed ffprobe would poison every later hit on this key"
        )
        assert payload["audio_path"] == "audio/x.mp3"

    @pytest.mark.asyncio
    async def test_store_cache_keeps_a_measured_zero(self):
        svc = self._service()
        await svc._store_cache("key1", "audio/x.mp3", 0.0, "mp3")

        payload = svc._redis.set.await_args.args[1]
        assert payload["duration"] == 0.0

    @pytest.mark.asyncio
    async def test_cache_hit_without_duration_reads_back_unmeasured(self):
        svc = self._service()
        svc._redis.get = AsyncMock(return_value={"audio_path": "audio/x.mp3", "format": "mp3"})
        svc._storage.retrieve = AsyncMock(return_value=b"\xff\xfb\x90audio")

        audio, _key, duration = await svc._check_cache(
            "tenant", "hello", "openai", "alloy", 1.0, "mp3"
        )
        assert audio == b"\xff\xfb\x90audio"
        assert duration is None, "a cache entry with no duration must read back as UNMEASURED"

    @pytest.mark.asyncio
    async def test_persist_cache_does_not_launder_an_unmeasured_duration(self):
        """End to end: an unmeasured GenerationResult must not put a number in Redis."""
        svc = self._service()
        svc._storage.store = AsyncMock(return_value="audio/cache_abc.mp3")
        result = GenerationResult(
            audio_data=b"\xff\xfb\x90audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=None,
        )
        await svc._persist_cache("abc", result)

        payload = svc._redis.set.await_args.args[1]
        assert "duration" not in payload


# ============================================================================
# ⚠️ The PRE-FIX poison outlives the fix unless the key namespace moves
# ============================================================================
#
# Omitting an unmeasured duration on WRITE does nothing about the entries the
# old code already wrote. A pre-fix entry carries a literal ``"duration": 0.0``
# produced by a FAILED probe, and the new reader — correctly — treats a present
# ``duration`` as a measurement. So for up to ``default_ttl`` (24h) every hit on
# such a key launders a fabricated zero back out as a confident measurement,
# through the very code path this PR added to prevent exactly that.
#
# The remediation is a namespace bump: pre-fix keys become unreachable and
# expire on their own TTL. These tests keep the two shapes DISTINGUISHABLE —
# a measured zero written by the CURRENT writer round-trips, while an entry in
# the LEGACY namespace is never served at all.

LEGACY_KEY_PREFIX = "tts:gen:"
"""The pre-fix namespace. Any entry here may carry a laundered probe failure."""


class _DictRedis:
    """Fake Redis keyed by the FULL key, so a prefix change is OBSERVABLE.

    An ``AsyncMock`` returns the same payload for every key and would make a
    legacy-vs-current namespace test vacuous.
    """

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        self.store[key] = value


class TestLegacyCacheEntriesCannotLaunderAFabricatedZero:
    CACHE_ARGS = ("tenant", "hello", "openai", "alloy", 1.0, "mp3")

    def _live_prefix(self) -> str:
        from src.config import CacheSettings

        return CacheSettings.model_fields["key_prefix"].default

    def _service(self):
        from src.services.tts_service import TTSService

        svc = TTSService.__new__(TTSService)
        svc._redis = _DictRedis()
        settings = MagicMock()
        settings.cache.enabled = True
        # The REAL configured prefix, not a test-local literal: the namespace
        # bump is the fix, so the test has to be able to see it.
        settings.cache.key_prefix = self._live_prefix()
        settings.cache.default_ttl = 86400
        svc._settings = settings
        svc._storage = AsyncMock()
        svc._storage.retrieve = AsyncMock(return_value=b"\xff\xfb\x90audio")
        return svc

    def _key(self) -> str:
        from src.services.tts_service import _compute_cache_key

        return _compute_cache_key(*self.CACHE_ARGS)

    def test_the_live_prefix_left_the_poisoned_namespace(self):
        assert self._live_prefix() != LEGACY_KEY_PREFIX, (
            "the cache key prefix is still the pre-fix one, so entries written by "
            "the OLD code — including a fabricated `duration: 0.0` from a failed "
            "probe — are still reachable and still read back as measured"
        )

    @pytest.mark.asyncio
    async def test_a_measured_zero_written_by_the_current_writer_round_trips(self):
        """The legitimate case: 0.0 that was actually MEASURED survives a round trip.

        Constructed the way the CURRENT writer constructs it — through
        ``_store_cache`` at the live prefix — so it is not byte-identical to a
        legacy entry that happens to share the same payload shape.
        """
        svc = self._service()
        await svc._store_cache(self._key(), "audio/x.mp3", 0.0, "mp3")

        assert list(svc._redis.store) == [f"{self._live_prefix()}{self._key()}"]

        audio, _key, duration = await svc._check_cache(*self.CACHE_ARGS)
        assert audio == b"\xff\xfb\x90audio"
        assert duration == 0.0, "a measured zero is DATA and must survive the cache"

    @pytest.mark.asyncio
    async def test_a_legacy_namespace_entry_is_not_served_at_all(self):
        """A pre-fix entry — the shape a failed probe left behind — must MISS.

        Same payload the old writer produced, in the old namespace. If the
        prefix has not moved, this is a HIT and the fabricated 0.0 is handed
        back as a measurement.
        """
        svc = self._service()
        svc._redis.store[f"{LEGACY_KEY_PREFIX}{self._key()}"] = {
            "audio_path": "audio/x.mp3",
            "duration": 0.0,
            "format": "mp3",
        }

        audio, _key, duration = await svc._check_cache(*self.CACHE_ARGS)
        assert audio is None, "a legacy cache entry was served after the namespace bump"
        assert duration is None, (
            "a `duration: 0.0` written by the PRE-FIX code was laundered back out "
            "as a confident measurement — the defect outlived its own remediation"
        )
