"""Billing-correctness regressions: quota drain, budget overshoot, stream abort.

Three money bugs pinned here (see PR fix/billing-stream-abort-and-quota):

* FIX 2 - failed synthesis must NOT consume the tenant's monthly quota. The
  ``tts_usage`` table is a *daily aggregate* (one row per tenant/provider/date)
  with an ``error_count`` column - there is no per-request ``is_error`` row to
  filter. The correct fix is write-side: a failed synthesis is recorded as an
  error only and never accrues billable ``request_count`` / ``total_characters``
  / ``estimated_cost_usd``, so ``check_quota`` (which sums those columns)
  naturally excludes it.
* FIX 3 - the monthly budget check must add the current request's projected cost
  before comparing, mirroring the character check, so a tenant at $9.99 of a $10
  budget cannot overshoot with one huge request.
* FIX 1 - a stream that delivers bytes and then fails mid-stream must still meter
  the delivered (committed) audio; otherwise the client keeps N seconds of paid
  audio for free and repeated aborts are a deliberate billing bypass.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TenantQuota
from src.providers.base import AudioChunk
from src.providers.openai_provider import _RetryableHTTPError
from src.services.job_manager import JobManager
from src.services.quota_service import QuotaExceededError, check_quota
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid


def _enforcement_on() -> MagicMock:
    """A get_settings() stand-in with quota enforcement enabled."""
    settings = MagicMock()
    settings.quota.enforcement_enabled = True
    return settings


def _retryable_error() -> _RetryableHTTPError:
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")
    response = httpx.Response(503, request=request)
    return _RetryableHTTPError("upstream 503", request=request, response=response)


# ============================================================================
# FIX 2 - failed synthesis must not drain the monthly quota
# ============================================================================


@pytest.mark.asyncio
async def test_error_usage_does_not_count_against_character_quota(
    db_session: AsyncSession,
) -> None:
    """A failed synthesis recorded via ``record_usage(is_error=True)`` must not
    consume ``monthly_character_limit`` - the tenant received no audio.

    RED (pre-fix): the error row incremented ``total_characters`` by 950, so the
    950 + 100 projection exceeded the 1000-char limit and ``check_quota`` raised.
    GREEN: the error row accrues no billable characters, so the request is
    admitted.
    """
    tenant = tenant_uuid("tenant-fix2")
    db_session.add(TenantQuota(tenant_id=tenant, monthly_character_limit=1000, is_active=True))
    await db_session.flush()

    # A provider 5xx storm: 950 chars of FAILED synthesis for this tenant.
    await JobManager.record_usage(
        db_session,
        tenant_id=tenant,
        provider="openai",
        characters=950,
        audio_seconds=0.0,
        is_error=True,
    )
    await db_session.commit()

    with patch("src.services.quota_service.get_settings", return_value=_enforcement_on()):
        try:
            # New 100-char request: 0 billable + 100 <= 1000 -> must be ADMITTED.
            await check_quota(db_session, tenant, 100)
        except QuotaExceededError as exc:  # pragma: no cover - failure path
            pytest.fail(f"failed synthesis wrongly counted against quota: {exc}")


# ============================================================================
# FIX 3 - monthly budget must include the current request's projected cost
# ============================================================================


@pytest.mark.asyncio
async def test_budget_rejects_request_that_would_overshoot(
    db_session: AsyncSession,
) -> None:
    """A tenant one cent under its budget must be rejected when the incoming
    request's projected cost exceeds that cent.

    RED (pre-fix): budget admitted whenever ``monthly_cost < budget`` regardless
    of the incoming cost, so a $9.99 tenant sailed through. GREEN: projected cost
    is added before the comparison.
    """
    tenant = tenant_uuid("tenant-fix3")
    db_session.add(TenantQuota(tenant_id=tenant, monthly_budget_usd=10.0, is_active=True))
    await db_session.flush()

    # Spend $9.99 of the $10 budget via a real (successful) usage row.
    await JobManager.record_usage(
        db_session,
        tenant_id=tenant,
        provider="openai",
        characters=100,
        audio_seconds=1.0,
        estimated_cost_usd=9.99,
    )
    await db_session.commit()

    with patch("src.services.quota_service.get_settings", return_value=_enforcement_on()):
        # Incoming request projected at $0.50 -> 9.99 + 0.50 = 10.49 >= 10.00.
        with pytest.raises(QuotaExceededError, match="Monthly cost budget exceeded"):
            await check_quota(db_session, tenant, 100, projected_cost=0.50)


# ============================================================================
# FIX 1 - a produced-then-failed stream must meter the delivered audio
# ============================================================================


class _FailAfterFirstChunkService:
    @staticmethod
    def build() -> tuple[TTSService, AsyncMock]:
        provider = AsyncMock()
        provider.name = "kokoro"
        provider.provider_type = "local"
        provider.sample_rate = 24000
        provider.is_available = AsyncMock(return_value=True)
        provider.get_default_voice = MagicMock(return_value="kokoro_default")
        provider.list_voices = AsyncMock(return_value=[])

        async def partial_then_fail(**_kwargs: object) -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"partial-audio", chunk_index=0)
            raise _retryable_error()

        provider.generate_stream = partial_then_fail

        registry = MagicMock()
        registry.get_provider.side_effect = lambda name: provider
        service = TTSService(registry=registry, storage=AsyncMock(), router=None)
        return service, provider


@pytest.mark.asyncio
async def test_produced_then_failed_stream_meters_delivered_audio() -> None:
    """Once a stream has yielded bytes and then a provider error aborts it, the
    delivered audio has been paid for by the provider and delivered to the
    client - it must be metered even though the stream ends in an error.

    RED (pre-fix): the ``except`` re-raised immediately for a produced stream, so
    ``record_usage`` was never called -> free audio + a repeatable billing
    bypass. GREEN: the delivered portion is metered before the re-raise.

    ⚠️ THE TEARDOWN FALLBACK IS NEUTRALISED HERE, DELIBERATELY. The
    ``finally:`` arm added later for client aborts calls
    ``_meter_abandoned_stream``, which fires on THIS path too (produced,
    not-yet-metered) and would meter the attempt even if the ``except`` arm
    did nothing at all. With it live, deleting the ``_meter_attempt_once``
    call from the ``except`` arm left this test GREEN -- it was measuring the
    fallback, not the line it names. Stubbing ``_meter_abandoned_stream`` to a
    no-op isolates the provider-failure arm so it must meter on its own.
    ``test_a_produced_then_failed_stream_is_metered_exactly_once`` below runs
    the same path with BOTH arms live and pins the count at one.
    """
    service, _provider = _FailAfterFirstChunkService.build()
    db = AsyncMock()

    collected: list[AudioChunk] = []
    with (
        patch("src.services.tts_service.JobManager") as mock_jm,
        patch.object(TTSService, "_meter_abandoned_stream", AsyncMock()) as mock_teardown,
    ):
        mock_jm.record_usage = AsyncMock()
        with pytest.raises(httpx.HTTPStatusError):
            async for chunk in service.generate_stream(
                text="Hello world",
                provider="kokoro",
                voice="custom",
                tenant_id=tenant_uuid("tenant-test"),
                db=db,
            ):
                collected.append(chunk)

        # The client did receive delivered audio ...
        assert [c.data for c in collected] == [b"partial-audio"]
        # ... so a usage row MUST have been recorded for it -- BY THE ``except``
        # ARM, with the teardown fallback proven inert (stubbed).
        mock_jm.record_usage.assert_awaited_once()
        usage_kwargs = mock_jm.record_usage.call_args[1]
        assert usage_kwargs["provider"] == "kokoro"
        assert usage_kwargs["characters"] > 0
        assert not usage_kwargs.get("is_error", False)
        # Precondition for the isolation above: the fallback really was the
        # thing we disabled, i.e. it WOULD otherwise have run on this path.
        mock_teardown.assert_awaited()


@pytest.mark.asyncio
async def test_a_produced_then_failed_stream_is_metered_exactly_once() -> None:
    """With BOTH the ``except`` arm and the ``finally`` teardown live, the
    produced-then-failed path must still write exactly ONE usage row.

    The ``attempt.metered`` latch is what makes the two arms compose; without
    it, closing the client-abort bypass would double-bill every mid-stream
    provider failure.
    """
    service, _provider = _FailAfterFirstChunkService.build()
    db = AsyncMock()

    with patch("src.services.tts_service.JobManager") as mock_jm:
        mock_jm.record_usage = AsyncMock()
        with pytest.raises(httpx.HTTPStatusError):
            async for _chunk in service.generate_stream(
                text="Hello world",
                provider="kokoro",
                voice="custom",
                tenant_id=tenant_uuid("tenant-test"),
                db=db,
            ):
                pass

        assert mock_jm.record_usage.await_count == 1


# ============================================================================
# FIX 4 - a CLIENT ABORT must meter the audio it already received
# ============================================================================


class _StallsAfterFirstChunkService:
    """A provider that delivers one real chunk and then never finishes.

    The stall is what makes the abort reachable: a client hangs up while bytes
    are still expected, which is precisely when the generator is torn down from
    the OUTSIDE rather than raising from within.
    """

    @staticmethod
    def build() -> tuple[TTSService, asyncio.Event]:
        released = asyncio.Event()

        provider = AsyncMock()
        provider.name = "kokoro"
        provider.provider_type = "local"
        provider.sample_rate = 24000
        provider.is_available = AsyncMock(return_value=True)
        provider.get_default_voice = MagicMock(return_value="kokoro_default")
        provider.list_voices = AsyncMock(return_value=[])

        async def one_chunk_then_stall(**_kwargs: object) -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"delivered-audio", chunk_index=0)
            await released.wait()  # never set in these tests
            yield AudioChunk(data=b"", chunk_index=1, is_final=True)

        provider.generate_stream = one_chunk_then_stall

        registry = MagicMock()
        registry.get_provider.side_effect = lambda name: provider
        # ⚠️ `metering_redis` is REQUIRED for the cost-bus half to be observable:
        # `_emit_cost_event` no-ops when it is None, so a service built without
        # it silently exercises only the quota row and a cost-bus assertion
        # would fail on correct code.
        service = TTSService(
            registry=registry,
            storage=AsyncMock(),
            router=None,
            metering_redis=MagicMock(),
        )
        return service, released


@pytest.mark.asyncio
async def test_client_abort_mid_stream_still_meters_delivered_audio() -> None:
    """A client that hangs up mid-stream KEEPS the bytes it already got, so
    those bytes must be metered.

    This is the GeneratorExit half: starlette closes the response body
    generator, which throws ``GeneratorExit`` in at the ``yield``. That is a
    ``BaseException``, so ``except PROVIDER_FAILURE_ERRORS`` — which can only
    ever list ``Exception`` subclasses — could never see it, and there was no
    ``finally``.

    RED (pre-fix): ``record_usage`` is never awaited; the client keeps
    ``delivered-audio`` for free AND, because ``_enforce_stream_quota``
    aggregates these very rows, the abort does not count against quota either.
    """
    service, _released = _StallsAfterFirstChunkService.build()
    db = AsyncMock()

    with patch("src.services.tts_service.JobManager") as mock_jm:
        mock_jm.record_usage = AsyncMock()

        # `generate_stream` is annotated `AsyncIterator`, but it IS an async
        # generator — `aclose()` is exactly the teardown starlette performs.
        stream = cast(
            "AsyncGenerator[AudioChunk]",
            service.generate_stream(
                text="Hello world",
                provider="kokoro",
                tenant_id=tenant_uuid("tenant-test"),
                db=db,
            ),
        )
        first = await stream.__anext__()
        assert first.data == b"delivered-audio"  # the client really received bytes

        # The client hangs up: the response-body generator is closed.
        await stream.aclose()

        mock_jm.record_usage.assert_awaited_once()
        usage_kwargs = mock_jm.record_usage.call_args[1]
        assert usage_kwargs["provider"] == "kokoro"
        assert usage_kwargs["characters"] > 0
        assert not usage_kwargs.get("is_error", False)


@pytest.mark.asyncio
async def test_cancelled_stream_task_still_meters_delivered_audio() -> None:
    """The CancelledError half: starlette cancels the response task on
    disconnect, so the teardown runs while the coroutine is unwinding a
    ``CancelledError`` rather than an ``Exception``.

    ⛔ WHAT THIS TEST DOES **NOT** PROVE, stated because it looks like it does:
    it does not pin the ``asyncio.shield``. Deleting the shield leaves this
    GREEN — bare asyncio delivers a cancellation once, so the next ``await``
    proceeds, and ``record_usage`` is an ``AsyncMock`` that never really
    suspends. The shield matters under the anyio cancel scopes starlette
    actually uses, which re-deliver at every checkpoint; that is out of reach at
    this level. See ``_meter_abandoned_stream``.

    What it DOES pin — and what a mutation confirms — is that the teardown arm
    exists at all: remove it and this reports ``Awaited 0 times``.
    """
    service, _released = _StallsAfterFirstChunkService.build()
    db = AsyncMock()
    started = asyncio.Event()

    cost_events: list[dict[str, Any]] = []

    def _capture_cost_event(*_args: object, **kwargs: Any) -> None:
        cost_events.append(kwargs)

    with (
        patch("src.services.tts_service.JobManager") as mock_jm,
        patch("src.services.tts_service.schedule_cost_event", _capture_cost_event),
    ):
        mock_jm.record_usage = AsyncMock()

        delivered: list[bytes] = []

        async def consume() -> None:
            async for chunk in service.generate_stream(
                text="Hello world",
                provider="kokoro",
                tenant_id=tenant_uuid("tenant-test"),
                db=db,
            ):
                if chunk.data:
                    delivered.append(chunk.data)
                    started.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=5)
        assert delivered, "the client must really have received bytes before aborting"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Let the shielded metering task finish; it outlives the cancelled one
        # by design, so it is not complete the instant `gather` returns.
        #
        # ⚠️ Drained on the LAST sink (the cost event), not the first. Waiting on
        # `record_usage` stopped the loop mid-way through `_meter_completed_stream`
        # — the quota row was already written and the cost event had not been
        # emitted yet — so the cost-bus assertion below failed on correct code.
        # A drain that stops at the first observable effect measures the drain,
        # not the behaviour.
        for _ in range(100):
            if cost_events:
                break
            await asyncio.sleep(0)

        # WHAT was billed, not merely that something was: the aborted stream
        # must charge the same tenant, the same provider and the same character
        # unit as a completed one, and must NOT be filed as an error (an error
        # row carries no billable characters, so it would re-open the bypass by
        # another route — see the FIX 2 tests at the top of this file).
        assert mock_jm.record_usage.await_count == 1
        usage_kwargs = mock_jm.record_usage.call_args[1]
        assert usage_kwargs["provider"] == "kokoro"
        assert usage_kwargs["characters"] == len("Hello world")
        assert usage_kwargs["tenant_id"] == tenant_uuid("tenant-test")
        assert not usage_kwargs.get("is_error", False)

        # THE BILLED QUANTITY ON THE COST BUS, not just the quota row. These are
        # two different sinks (`JobManager.record_usage` -> `tts_usage`;
        # `schedule_cost_event` -> the cost bus) and only the second one is money
        # — a fix that populated one and not the other would look correct here
        # without it.
        assert cost_events, "an aborted stream must still reach the cost bus"
        assert len(cost_events) == 1
        event = cost_events[0]
        assert event["provider"] == "kokoro"
        assert event["tenant_id"] == tenant_uuid("tenant-test")
        # `input_tokens` carries the character count for this service (unit is
        # stamped "characters" in the event metadata).
        assert event["input_tokens"] == len("Hello world")
        assert event["cost_usd"] >= 0.0


@pytest.mark.asyncio
async def test_a_completed_stream_is_metered_exactly_once() -> None:
    """THE CONTROL THAT MATTERS MOST HERE.

    The teardown arm is a ``finally``, so it also runs on the success path and
    on the produced-then-failed path. If it were not latched by
    ``_StreamAttempt.metered`` this fix would DOUBLE-BILL every successful
    stream — strictly worse than the bypass it closes. Asserted on the await
    COUNT, not merely on "was it called".
    """
    provider = AsyncMock()
    provider.name = "kokoro"
    provider.provider_type = "local"
    provider.sample_rate = 24000
    provider.is_available = AsyncMock(return_value=True)
    provider.get_default_voice = MagicMock(return_value="kokoro_default")
    provider.list_voices = AsyncMock(return_value=[])

    async def full_stream(**_kwargs: object) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"all-of-it", chunk_index=0)
        yield AudioChunk(data=b"", chunk_index=1, is_final=True)

    provider.generate_stream = full_stream
    registry = MagicMock()
    registry.get_provider.side_effect = lambda name: provider
    service = TTSService(registry=registry, storage=AsyncMock(), router=None)

    with patch("src.services.tts_service.JobManager") as mock_jm:
        mock_jm.record_usage = AsyncMock()

        chunks = [
            c
            async for c in service.generate_stream(
                text="Hello world",
                provider="kokoro",
                tenant_id=tenant_uuid("tenant-test"),
                db=AsyncMock(),
            )
        ]

        assert [c.data for c in chunks if c.data] == [b"all-of-it"]
        assert mock_jm.record_usage.await_count == 1
