"""Regression tests: redis-py and SQLAlchemy errors are OUTSIDE the builtin tree.

The premise of the whole defect class, measured against the installed redis-py::

    redis.exceptions.ConnectionError.__mro__
        -> (ConnectionError, RedisError, Exception, BaseException, object)
    issubclass(redis.exceptions.ConnectionError, builtins.ConnectionError) -> False
    issubclass(redis.exceptions.TimeoutError,    builtins.TimeoutError)    -> False
    issubclass(redis.exceptions.RedisError,      OSError)                  -> False

The names collide, the classes do not. So every
``except (OSError, ConnectionError, TimeoutError, RuntimeError)`` in this repo
caught only the *builtins* and let every real Redis failure escape. The same
holds for ``sqlalchemy.exc.OperationalError`` (``DatabaseError -> DBAPIError ->
StatementError -> SQLAlchemyError -> Exception``), into which psycopg3 wraps its
errors — so a pgbouncer/Postgres blip escaped too.

Consequences pinned here:

* the fail-soft ``src.cache.redis_client.RedisClient`` wrapper was not
  fail-soft at all — it propagated instead of returning ``None``/``False``/``0``;
* ``NotificationService._subscriber_loop`` died on the first Redis blip
  (never reaching its own reconnect ``sleep``), leaking the pubsub connection
  and leaving notifications dead while ``/health`` stayed green;
* **the money bug** — ``_notify("job_completed", ...)`` ran INSIDE ``process_job``'s
  try block, BEFORE ``await db.commit()``. A dead Redis there rolled back
  ``mark_completed`` AND ``record_usage`` and skipped the billing emit, for a job
  whose audio had already been synthesized, paid for and stored;
* a DB fault in the background job never reached ``_mark_job_failed``, so the
  job sat in ``processing`` forever.
"""

from __future__ import annotations

import asyncio
import builtins
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.exceptions
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.redis_client import RedisClient
from src.providers.base import GenerationResult
from src.services.job_manager import JobManager
from src.services.notification_service import PUBSUB_CHANNEL, JobEvent, NotificationService
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid


def _redis_conn_error() -> redis.exceptions.ConnectionError:
    """The exact exception redis-py raises when the server is unreachable."""
    return redis.exceptions.ConnectionError(
        "Error 111 connecting to redis:6379. Connection refused."
    )


def _db_operational_error() -> OperationalError:
    """The exact exception SQLAlchemy raises when psycopg3 loses the server."""
    return OperationalError(
        "SELECT 1",
        {},
        Exception("server closed the connection unexpectedly"),
    )


# ---------------------------------------------------------------------------
# 5 - the canary: if redis-py ever re-parents these, this file's premise moves
# ---------------------------------------------------------------------------


class TestExceptionHierarchyCanary:
    def test_redis_errors_are_not_the_builtins_they_shadow(self) -> None:
        assert not issubclass(redis.exceptions.ConnectionError, builtins.ConnectionError)
        assert not issubclass(redis.exceptions.TimeoutError, builtins.TimeoutError)
        assert not issubclass(redis.exceptions.RedisError, OSError)
        # ... which is exactly why RedisError is the only handle that works.
        assert issubclass(redis.exceptions.ConnectionError, redis.exceptions.RedisError)
        assert issubclass(redis.exceptions.TimeoutError, redis.exceptions.RedisError)

    def test_sqlalchemy_operational_error_is_outside_every_builtin_tuple(self) -> None:
        assert not issubclass(
            OperationalError,
            (OSError, builtins.ConnectionError, builtins.TimeoutError, RuntimeError, ValueError),
        )


# ---------------------------------------------------------------------------
# 1 - the fail-soft wrapper must actually fail soft
# ---------------------------------------------------------------------------


class TestRedisClientIsFailSoftOnRealRedisErrors:
    @pytest.mark.asyncio
    async def test_publish_returns_zero(self) -> None:
        client = RedisClient(redis_url="redis://localhost:6379/0")
        # MagicMock (not AsyncMock) as the base: only the method production is
        # supposed to call is awaitable, so calling a different one fails loudly.
        backing = MagicMock()
        backing.publish = AsyncMock(side_effect=_redis_conn_error())
        client._client = backing

        assert await client.publish("tts:job_events", "{}") == 0
        backing.publish.assert_awaited_once_with("tts:job_events", "{}")

    @pytest.mark.asyncio
    async def test_get_bytes_returns_none(self) -> None:
        client = RedisClient(redis_url="redis://localhost:6379/0")
        backing = MagicMock()
        backing.get = AsyncMock(side_effect=_redis_conn_error())
        client._binary_client = backing

        assert await client.get_bytes("audio:abc") is None
        backing.get.assert_awaited_once_with("audio:abc")

    @pytest.mark.asyncio
    async def test_set_bytes_returns_false(self) -> None:
        client = RedisClient(redis_url="redis://localhost:6379/0")
        backing = MagicMock()
        backing.setex = AsyncMock(side_effect=_redis_conn_error())
        client._binary_client = backing

        assert await client.set_bytes("audio:abc", b"data", ttl=300) is False
        backing.setex.assert_awaited_once_with("audio:abc", 300, b"data")


# ---------------------------------------------------------------------------
# 2 - notification publish/subscribe must survive a Redis outage
# ---------------------------------------------------------------------------


class TestNotificationServiceSurvivesRedisOutage:
    @pytest.mark.asyncio
    async def test_publish_event_does_not_propagate(self) -> None:
        backing = MagicMock()
        backing.publish = AsyncMock(side_effect=_redis_conn_error())
        service = NotificationService(redis_client=backing)

        # Must NOT raise: publishing an event is best-effort.
        await service.publish_event(
            JobEvent(event="job_completed", job_id="j1", tenant_id=tenant_uuid("t1"))
        )

        backing.publish.assert_awaited_once()
        assert backing.publish.call_args[0][0] == PUBSUB_CHANNEL

    @pytest.mark.asyncio
    async def test_subscriber_loop_reconnects_instead_of_dying(self) -> None:
        """The one error the reconnect arm exists to survive must reach the sleep."""
        backing = MagicMock()
        backing.get_pubsub = MagicMock(side_effect=_redis_conn_error())
        service = NotificationService(redis_client=backing)
        service._running = True

        slept: list[float] = []

        async def _stop_after_first_wait(delay: float) -> None:
            slept.append(delay)
            service._running = False  # bound the loop; one reconnect wait is enough

        with patch.object(asyncio, "sleep", _stop_after_first_wait):
            # RED: redis.exceptions.ConnectionError escaped the loop's except
            # tuple, so this call raised and the background task was dead.
            await service._subscriber_loop()

        assert slept == [5]


# ---------------------------------------------------------------------------
# 3 - THE MONEY TEST: a notification failure must not undo committed billing
# ---------------------------------------------------------------------------


def _completed_job() -> MagicMock:
    job = MagicMock()
    job.text = "Hello"
    job.provider = "kokoro"
    job.voice = "af_heart"
    job.speed = 1.0
    job.output_format = "mp3"
    job.model = None
    job.text_length = 5
    job.routing_strategy = "balanced"
    return job


@pytest.mark.asyncio
async def test_job_completed_notification_failure_does_not_undo_commit_or_billing() -> None:
    """A dead Redis at the ``job_completed`` notification must cost nothing.

    RED (pre-fix): ``_notify`` ran inside the try block before ``db.commit()``
    and ``redis.exceptions.ConnectionError`` was in none of the except tuples,
    so ``process_job`` raised -> the session rolled back (losing
    ``mark_completed`` + ``record_usage``) and the cost event was never emitted,
    for audio the provider had already been paid for.
    GREEN: the job commits, usage is recorded, the charge is emitted, and the
    notification failure is swallowed.
    """
    storage = AsyncMock()
    storage.store = AsyncMock(return_value="/path/audio.mp3")
    service = TTSService(registry=MagicMock(), storage=storage)
    db = AsyncMock()

    result = GenerationResult(
        audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
    )

    async def _notify_side_effect(event_type: str, *_args: object, **_kwargs: object) -> None:
        if event_type == "job_completed":
            raise _redis_conn_error()

    with (
        patch.object(
            service,
            "generate_sync",
            AsyncMock(
                return_value=(
                    result,
                    {"cached": False, "estimated_cost": 0.01, "latency_ms": 50},
                )
            ),
        ),
        patch.object(service, "_notify", AsyncMock(side_effect=_notify_side_effect)) as mock_notify,
        patch.object(service, "_emit_cost_event") as mock_emit,
        patch("src.services.tts_service.JobManager") as mock_jm,
    ):
        mock_jm.get_job = AsyncMock(return_value=_completed_job())
        mock_jm.mark_processing = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.record_usage = AsyncMock()

        # Must NOT raise.
        await service.process_job(db, "job-1", tenant_uuid("t1"))

        # The job is terminally COMPLETED, not failed and not stranded.
        mock_jm.mark_completed.assert_awaited_once()
        mock_jm.mark_failed.assert_not_awaited()

        # The usage/billing write survived ...
        mock_jm.record_usage.assert_awaited_once()
        usage_kwargs = mock_jm.record_usage.call_args[1]
        assert usage_kwargs["tenant_id"] == tenant_uuid("t1")
        assert not usage_kwargs.get("is_error", False)

        # ... and was durably committed, then charged on the cost bus.
        assert db.commit.await_count >= 2  # mark_processing commit + completion commit
        mock_emit.assert_called_once()

        # The notification must still actually BE SENT, with its payload intact.
        # Without this the test is satisfied by deleting the notify call outright
        # (the side_effect simply never fires) — i.e. it would go green on code
        # that silently stopped notifying every WebSocket subscriber. Proven: an
        # adversarial pass ran this exact assertion set against a _notify that
        # never raises, and it passed.
        mock_notify.assert_any_await(
            "job_completed",
            "job-1",
            tenant_uuid("t1"),
            audio_url="/api/tts/jobs/job-1/audio",
            duration=1.0,
        )
        # And it must be the LAST thing that happens — after the commit and after
        # the billing emit — which is the whole point of the reordering.
        assert mock_notify.await_args_list[-1].args[0] == "job_completed"


# ---------------------------------------------------------------------------
# 4 - a DB fault must still terminalise the background job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_job_is_terminalised_on_database_fault() -> None:
    """RED (pre-fix): OperationalError matched no arm, so the task died silently
    and the job row stayed ``processing`` until a restart swept it."""
    from src.api.tts import _process_job_background

    tts = MagicMock()
    tts.process_job = AsyncMock(side_effect=_db_operational_error())

    with (
        patch("src.api.tts.get_db_session") as mock_get_db,
        patch("src.api.tts._mark_job_failed", AsyncMock()) as mock_mark,
    ):
        mock_get_db.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

        await _process_job_background(tts, "job-1", tenant_uuid("tenant-1"))

    mock_mark.assert_awaited_once()
    assert mock_mark.call_args[0][0] == "job-1"


# ---------------------------------------------------------------------------
# 5 - widening the caller's except must NOT let it demote a committed job
# ---------------------------------------------------------------------------


async def test_mark_failed_cannot_demote_a_completed_job(db_session: AsyncSession) -> None:
    """Against the REAL database, not an AsyncMock.

    ``process_job`` commits the completion and only THEN emits the cost event,
    so anything raising after that commit lands in the caller's ``except`` arm
    in ``api/tts.py``, which calls ``JobManager.mark_failed`` directly. That
    UPDATE used to be unconditional -- it would overwrite a durably-completed,
    already-billed row as ``failed``. Adding ``RedisError``/``SQLAlchemyError``
    to that arm (the point of this change) widens the window, so the terminal
    guard ships with it.

    RED: unconditional UPDATE -> status flips to 'failed', the real completion
    and its audio metadata are lost.
    """
    job = await JobManager.create_job(
        db_session,
        tenant_id=tenant_uuid("t-demote"),
        text="hello",
        provider="openai",
        voice="alloy",
    )
    await JobManager.mark_completed(
        db_session,
        job.job_id,
        audio_path="/audio.mp3",
        audio_size_bytes=123,
        audio_duration_seconds=1.5,
        estimated_cost_usd=0.01,
    )
    await db_session.flush()

    moved = await JobManager.mark_failed(db_session, job.job_id, "Unexpected error: redis blip")
    await db_session.flush()

    assert moved is False, "a completed job must not be moved by mark_failed"

    row = await JobManager.get_job(db_session, job.job_id, tenant_id=tenant_uuid("t-demote"))
    assert row is not None
    assert row.status == "completed", "a billed, completed job was demoted to failed"
    assert row.error_message is None
    assert row.audio_path == "/audio.mp3"


async def test_mark_failed_still_moves_a_live_job_and_keeps_the_first_error(
    db_session: AsyncSession,
) -> None:
    """The guard must not break the case it exists to serve, and the FIRST
    (specific) error message must survive a later generic one."""
    job = await JobManager.create_job(
        db_session, tenant_id=tenant_uuid("t-live"), text="hello", provider="openai", voice="alloy"
    )
    await db_session.flush()

    assert await JobManager.mark_failed(db_session, job.job_id, "real provider 500") is True
    await db_session.flush()

    # A second, generic failure must not clobber the operator-useful first one.
    assert (
        await JobManager.mark_failed(db_session, job.job_id, "Unexpected error: cleanup") is False
    )
    await db_session.flush()

    row = await JobManager.get_job(db_session, job.job_id, tenant_id=tenant_uuid("t-live"))
    assert row is not None
    assert row.status == "failed"
    assert row.error_message == "real provider 500"
    assert row.completed_at is not None
