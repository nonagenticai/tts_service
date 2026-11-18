"""Unit tests for tts_service cost-metering wiring.

The emit field-shape, stream resolution, and Redis lifecycle now live in
``w2v_common.metering`` (with their own test suite). These tests cover only
what tts_service owns: the routing identity constants (``service_key`` /
``cost_source``), the unattributed-tenant default, and that the service's
``_emit_cost_event`` helper fires one correctly-attributed XADD through the
shared GC-safe scheduler, and that NO tenant sentinel is reachable from this
package (product-owner ruling — see ``src/services/metering.py``).
"""

from __future__ import annotations

import inspect
import json
import pathlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from src.services import metering as metering_module
from src.services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    InvalidTenantIdError,
    MissingTenantError,
    init_metering_redis,
    require_tenant,
    schedule_cost_event,
)
from tests.conftest import TENANT_A, TENANT_B

if TYPE_CHECKING:
    from src.services.tts_service import TTSService


async def _drain_tasks() -> None:
    """Yield to the loop so fire-and-forget emit tasks run to completion."""
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestRoutingIdentity:
    """tts_service owns its cost-bus routing slugs; it owns NO tenant default."""

    def test_service_routing_slugs(self) -> None:
        assert SERVICE_KEY == "w2v-tts"  # pragma: allowlist secret -- routing slug
        assert COST_SOURCE == "w2v-tts"

    def test_no_tenant_sentinel_is_reachable_from_this_package(self) -> None:
        """PRODUCT-OWNER RULING: a tenant_id may NEVER hold a sentinel.

        This asserts the OPPOSITE of what it used to: there is no
        ``DEFAULT_TENANT_ID``, no ``UNATTRIBUTED_TENANT_ID`` re-export, and no
        env-driven substitution. The nil UUID is a VALID uuid, so it passes every
        cast and dirty-row check and billing then buckets it as unattributed and
        NEVER CHARGES it — real usage, silently unbilled. Re-adding either name
        (or the nil-uuid literal) turns this test red on purpose.
        """
        for banned in ("DEFAULT_TENANT_ID", "UNATTRIBUTED_TENANT_ID"):
            assert not hasattr(metering_module, banned), f"sentinel re-added: {banned}"
            assert banned not in metering_module.__all__
        source = pathlib.Path(metering_module.__file__).read_text()
        # Only in the do-not-re-add comment, never as a value.
        assert "00000000-0000-0000-0000-000000000000" not in source
        assert 'os.getenv("DEFAULT_TENANT_ID"' not in source


class TestScheduleCostEvent:
    """The shared scheduler emits one XADD carrying tts_service identity."""

    async def test_emit_carries_service_identity_and_fields(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"1234567890-0")

        schedule_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=TENANT_A,
            license_id="lic-1",
            subject_id="sub-1",
            app_id="app-1",
            provider="elevenlabs",
            model="eleven_multilingual_v2",
            input_tokens=120,
            output_tokens=0,
            latency_ms=850,
            usage_extra={"unit": "characters", "duration_s": 4.2, "audio_format": "mp3"},
        )
        await _drain_tasks()

        redis_mock.xadd.assert_awaited_once()
        call_args = redis_mock.xadd.call_args
        # Stream resolves centrally (METERING_STREAM env -> canonical default).
        assert call_args.args[0] == "billing:costs"
        fields = call_args.args[1]
        assert fields["tenant_id"] == TENANT_A
        assert fields["license_id"] == "lic-1"
        assert fields["subject_id"] == "sub-1"
        assert fields["app_id"] == "app-1"
        assert fields["service_key"] == SERVICE_KEY == "w2v-tts"  # pragma: allowlist secret
        assert fields["cost_source"] == COST_SOURCE == "w2v-tts"
        assert fields["provider"] == "elevenlabs"
        assert fields["model"] == "eleven_multilingual_v2"
        assert fields["input_tokens"] == "120"
        assert fields["latency_ms"] == "850"
        assert json.loads(fields["usage_extra"]) == {
            "audio_format": "mp3",
            "duration_s": 4.2,
            "unit": "characters",
        }
        assert call_args.kwargs["maxlen"] == 100_000
        assert call_args.kwargs["approximate"] is True

    async def test_xadd_failure_is_swallowed(self) -> None:
        """Metering must never fail the request path."""
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(side_effect=ConnectionError("redis down"))

        # Must not raise.
        schedule_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=TENANT_A,
            provider="openai",
            model="tts-1",
            input_tokens=10,
            output_tokens=0,
            latency_ms=100,
        )
        await _drain_tasks()

    @pytest.mark.parametrize("bad_tenant", ["", "   ", None, 0, object()])
    def test_falsy_tenant_is_rejected_not_coerced(self, bad_tenant: object) -> None:
        """This used to assert coercion to the nil-uuid sentinel. Now it must RAISE.

        The substitution is what made unattributed traffic un-billable; the only
        correct handling of a missing tenant is to fail loud before any emit.
        """
        with pytest.raises(MissingTenantError):
            require_tenant(bad_tenant)

    @pytest.mark.parametrize(
        "nil_spelling",
        [
            "00000000-0000-0000-0000-000000000000",
            "00000000000000000000000000000000",
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
            "  00000000-0000-0000-0000-000000000000  ",
        ],
    )
    def test_nil_uuid_is_rejected_by_value_in_every_spelling(self, nil_spelling: str) -> None:
        """The nil UUID is a VALID uuid, so a None-only guard never sees it.

        It is the most dangerous sentinel spelling: billing buckets it as
        unattributed and never charges it. Reject it by value.
        """
        with pytest.raises(MissingTenantError):
            require_tenant(nil_spelling)

    def test_require_tenant_returns_the_stripped_real_tenant(self) -> None:
        assert require_tenant(f"  {TENANT_B}  ") == TENANT_B

    @pytest.mark.parametrize(
        "slug",
        ["n8n-dev", "tenant-1", "default", "anonymous", "unknown", "Bearer abc.def.ghi"],
    )
    def test_non_uuid_tenant_is_rejected_as_invalid_not_missing(self, slug: str) -> None:
        """A slug is a BAD ARGUMENT (400), not an absent identity (401).

        Before ``005_tenant_id_uuid`` this function returned the slug untouched
        and the column swallowed it. Now the column is ``uuid``, so the same
        value would reach psycopg and come back as a ``DataError`` 500 partway
        through a write. Reject it here, with the right code.
        """
        with pytest.raises(InvalidTenantIdError):
            require_tenant(slug)

    def test_invalid_tenant_error_does_not_echo_the_value(self) -> None:
        """The message must not leak a credential.

        ``X-Tenant-Id`` on this service carries ``Bearer <license_token>``, so a
        mis-plumbed caller can land a live token in this field. Only the length
        is reported.
        """
        # Named for what it IS on the wire — a mis-plumbed X-Tenant-Id header —
        # rather than `secret`/`token`, which detect-secrets' KeywordDetector
        # flags on the variable NAME alone. The repo baselines nothing
        # (`"results": {}`), so the fix is the name, not an allowlist entry.
        mis_plumbed_header = "Bearer eyJhbGciOiJIUzI1NiJ9.not-a-real-value"
        with pytest.raises(InvalidTenantIdError) as exc:
            require_tenant(mis_plumbed_header)
        assert mis_plumbed_header not in str(exc.value)
        assert f"length {len(mis_plumbed_header)}" in str(exc.value)

    @pytest.mark.parametrize(
        "spelling",
        [
            "22222222-2222-4222-8222-222222222222",
            "22222222-2222-4222-8222-222222222222".upper(),
            "{22222222-2222-4222-8222-222222222222}",
            "urn:uuid:22222222-2222-4222-8222-222222222222",
            "22222222222242228222222222222222",
        ],
    )
    def test_every_uuid_spelling_is_folded_to_the_canonical_form(self, spelling: str) -> None:
        """Canonicalisation is load-bearing, not cosmetic.

        Postgres renders a ``uuid`` column back as lowercase-dashed text. A
        caller that sent an uppercase or braced spelling would otherwise fail
        every ``TTSJob.tenant_id == tenant_id`` comparison and subscribe to a
        Redis channel nobody publishes to — with no error anywhere.
        """
        assert require_tenant(spelling) == TENANT_B

    def test_invalid_is_not_a_subclass_of_missing(self) -> None:
        """They map to DIFFERENT status codes, so one must not catch the other."""
        assert not issubclass(InvalidTenantIdError, MissingTenantError)
        assert not issubclass(MissingTenantError, InvalidTenantIdError)


class TestInitMeteringRedis:
    """Init returns None when the URL is empty; never raises on connect failure."""

    async def test_empty_url_returns_none(self) -> None:
        client = await init_metering_redis("")
        assert client is None

    async def test_connect_failure_returns_none(self) -> None:
        # Point at a black-hole URL that will fail to ping.
        client = await init_metering_redis("redis://127.0.0.1:1/0")
        assert client is None


class TestTTSServiceEmits:
    """When TTSService.generate_sync completes, _emit_cost_event fires one XADD.

    These tests bypass the full TTSService construction and exercise only the
    _emit_cost_event helper on a minimally-stubbed instance — the surrounding
    pipeline is covered by tests/unit/test_api_tts.py.
    """

    def _make_service_with_metering(self, redis_mock: AsyncMock) -> TTSService:
        """Build a TTSService whose _metering_redis is the AsyncMock."""
        from src.services.tts_service import TTSService

        # __init__ touches a lot of plumbing; bypass with object.__new__ and
        # set only the field the helper reads (background-task lifecycle is now
        # owned by w2v_common.metering.schedule_cost_event).
        svc = TTSService.__new__(TTSService)
        svc._metering_redis = redis_mock
        return svc

    async def test_no_emit_when_metering_disabled(self) -> None:
        from src.services.tts_service import TTSService

        svc = TTSService.__new__(TTSService)
        svc._metering_redis = None
        # Must not raise; nothing to assert beyond the no-op.
        svc._emit_cost_event(
            tenant_id=TENANT_A,
            provider="openai",
            model="tts-1",
            text="hello",
            duration_s=1.0,
            audio_format="mp3",
            latency_ms=100,
            call_kind="sync",
            call_id="req-1",
        )

    async def test_emit_carries_resolved_provider_and_text_len(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")
        svc = self._make_service_with_metering(redis_mock)

        svc._emit_cost_event(
            tenant_id=TENANT_B,
            provider="elevenlabs",
            model="eleven_multilingual_v2",
            text="hello world",  # 11 chars
            duration_s=0.7,
            audio_format="mp3",
            latency_ms=420,
            call_kind="sync",
            call_id="req-1",
        )
        await _drain_tasks()

        redis_mock.xadd.assert_awaited()
        fields = redis_mock.xadd.call_args.args[1]
        assert fields["tenant_id"] == TENANT_B
        assert fields["service_key"] == SERVICE_KEY  # pragma: allowlist secret -- routing slug
        assert fields["cost_source"] == COST_SOURCE
        assert fields["provider"] == "elevenlabs"
        assert fields["input_tokens"] == "11"
        assert fields["latency_ms"] == "420"
        usage_extra = json.loads(fields["usage_extra"])
        assert usage_extra["unit"] == "characters"
        assert usage_extra["duration_s"] == 0.7
        assert usage_extra["audio_format"] == "mp3"

    @pytest.mark.parametrize("method", ["generate_sync", "generate_stream"])
    def test_tenant_id_has_no_default_at_all(self, method: str) -> None:
        """Inverted: these used to default to the nil-uuid sentinel.

        ``tenant_id`` is now a REQUIRED keyword-only argument, so a caller that
        forgets it gets a TypeError instead of silently attributing real usage
        to a bucket billing never charges.
        """
        from src.services.tts_service import TTSService

        param = inspect.signature(getattr(TTSService, method)).parameters["tenant_id"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
