"""Edge cases and invariants for ``JobManager``.

Core CRUD coverage lives in ``test_job_manager_real_db.py``; this file
holds the non-obvious edge cases that the main suite does not exercise:
unicode ``text_length``, UUID7 uniqueness under load, ``None`` vs ``{}``
distinction in JSONB columns, empty-set query behavior, and the
semantics of calling ``mark_*`` methods across unusual state transitions.

Every test runs against the real postgres CI database supplied by the
``db_session`` fixture — no mocks of anything under test.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TTSJob
from src.services.job_manager import JobManager
from tests.conftest import tenant_uuid


async def _create_default_job(
    db: AsyncSession,
    *,
    tenant_id: str = tenant_uuid("tenant-1"),
    text: str = "Hello world",
    provider: str = "kokoro",
    voice: str = "af_heart",
    batch_id: str | None = None,
    caller_item_id: str | None = None,
    extra_params: dict | None = None,
    routing_strategy: str | None = None,
    model: str | None = None,
    speed: float = 1.0,
    output_format: str = "mp3",
) -> TTSJob:
    """Shared test-data helper (same shape as ``test_job_manager_real_db.py``)."""
    job = await JobManager.create_job(
        db,
        tenant_id=tenant_id,
        text=text,
        provider=provider,
        voice=voice,
        speed=speed,
        output_format=output_format,
        model=model,
        batch_id=batch_id,
        caller_item_id=caller_item_id,
        extra_params=extra_params,
        routing_strategy=routing_strategy,
    )
    await db.commit()
    return job


@pytest.mark.asyncio
class TestCreateJobEdgeCases:
    async def test_text_length_counts_codepoints_not_bytes_for_multibyte_unicode(
        self, db_session: AsyncSession
    ) -> None:
        """``text_length`` must be ``len(text)`` in Python character units.

        Tripwire for ``job_manager.py:60`` — if someone changes it to
        ``len(text.encode())`` to match a database byte column, this fails.
        """
        text = "こんにちは🎉"  # 5 kana + 1 emoji = 6 code points; ~19 UTF-8 bytes
        job = await _create_default_job(db_session, text=text)
        assert job.text_length == 6
        assert job.text_length == len(text)

        # And the value survives a round-trip through the database.
        fetched = await JobManager.get_job(
            db_session, job.job_id, tenant_id=tenant_uuid("tenant-1")
        )
        assert fetched is not None
        assert fetched.text == text
        assert fetched.text_length == 6

    async def test_text_length_zero_for_empty_string(self, db_session: AsyncSession) -> None:
        """Empty text must persist with ``text_length == 0`` (no NOT NULL violation)."""
        job = await _create_default_job(db_session, text="")
        assert job.text_length == 0
        assert job.text == ""

        fetched = await JobManager.get_job(
            db_session, job.job_id, tenant_id=tenant_uuid("tenant-1")
        )
        assert fetched is not None
        assert fetched.text_length == 0
        assert fetched.text == ""

    async def test_job_ids_unique_across_many_concurrent_creations(
        self, db_session: AsyncSession
    ) -> None:
        """100 back-to-back ``create_job`` calls produce 100 distinct UUID7s.

        This would have failed loudly at flush time if ``uuid.uuid7()``
        collided (unique constraint on ``job_id``). It also sanity-checks
        that job_ids really are stable/persisted rather than regenerated.
        """
        jobs: list[TTSJob] = []
        for i in range(100):
            jobs.append(await _create_default_job(db_session, text=f"msg-{i}"))

        ids = {j.job_id for j in jobs}
        assert len(ids) == 100

        # All ids are well-formed UUID7 strings.
        for job_id in ids:
            parsed = uuid.UUID(job_id)
            assert parsed.version == 7

        # And they all persist (no silent flush-drop).
        count_result = await db_session.execute(select(TTSJob.id))
        persisted_count = len(list(count_result.scalars().all()))
        assert persisted_count == 100

    async def test_extra_params_distinguishes_none_from_empty_dict(
        self, db_session: AsyncSession
    ) -> None:
        """A ``None`` ``extra_params`` must round-trip as ``None``, not as ``{}``.

        Callers that check ``job.extra_params is None`` vs ``not job.extra_params``
        branch differently on this distinction; JSONB columns sometimes collapse
        the two.
        """
        job_none = await _create_default_job(db_session, extra_params=None)
        job_empty = await _create_default_job(db_session, extra_params={})

        fetched_none = await JobManager.get_job(
            db_session, job_none.job_id, tenant_id=tenant_uuid("tenant-1")
        )
        fetched_empty = await JobManager.get_job(
            db_session, job_empty.job_id, tenant_id=tenant_uuid("tenant-1")
        )

        assert fetched_none is not None
        assert fetched_empty is not None
        assert fetched_none.extra_params is None
        assert fetched_empty.extra_params == {}
        assert fetched_empty.extra_params is not None


@pytest.mark.asyncio
class TestGetBatchJobsEdgeCases:
    async def test_returns_empty_list_for_unknown_batch_id(self, db_session: AsyncSession) -> None:
        """``get_batch_jobs`` must return ``[]`` for an unknown batch, not ``None``."""
        result = await JobManager.get_batch_jobs(
            db_session, "does-not-exist", tenant_id=tenant_uuid("tenant-1")
        )
        assert result == []
        assert isinstance(result, list)

    async def test_returns_empty_list_when_other_batches_exist(
        self, db_session: AsyncSession
    ) -> None:
        """Sibling batches must not leak into an unrelated batch lookup."""
        await _create_default_job(db_session, batch_id="batch-a", text="a1")
        await _create_default_job(db_session, batch_id="batch-a", text="a2")
        await _create_default_job(db_session, batch_id="batch-b", text="b1")

        result = await JobManager.get_batch_jobs(
            db_session, "batch-c", tenant_id=tenant_uuid("tenant-1")
        )
        assert result == []


@pytest.mark.asyncio
class TestMarkMethodsEdgeCases:
    async def test_mark_processing_on_nonexistent_job_is_silent_noop(
        self, db_session: AsyncSession
    ) -> None:
        """``mark_processing`` on an unknown ``job_id`` is currently a silent no-op.

        This documents the present ``UPDATE ... WHERE job_id = X`` semantics
        (zero rows matched, no exception raised). If the TOCTOU fix is ever
        applied — adding a ``WHERE status = 'pending'`` guard and a rowcount
        check — this test must be updated to expect a ``JobStateConflict``.
        """
        # Pre-existing job for a different id (proves we're not just hitting an empty DB).
        existing = await _create_default_job(db_session)

        await JobManager.mark_processing(db_session, "does-not-exist")
        await db_session.commit()

        # No exception raised; no row with that id exists.
        missing = await JobManager.get_job(
            db_session, "does-not-exist", tenant_id=tenant_uuid("tenant-1")
        )
        assert missing is None

        # And the existing pending job is untouched.
        await db_session.refresh(existing)
        assert existing.status == "pending"
        assert existing.started_at is None

    async def test_mark_completed_twice_keeps_final_state_consistent(
        self, db_session: AsyncSession
    ) -> None:
        """Calling ``mark_completed`` twice is safe and produces final-write state.

        Demonstrates last-write-wins for ``completed_at`` and observable
        idempotency for the other fields.
        """
        job = await _create_default_job(db_session)

        await JobManager.mark_completed(
            db_session,
            job.job_id,
            audio_path="/audio/first.mp3",
            audio_size_bytes=100,
            audio_duration_seconds=1.0,
            estimated_cost_usd=0.01,
        )
        await db_session.commit()
        await db_session.refresh(job)
        first_completed_at = job.completed_at
        assert first_completed_at is not None

        await JobManager.mark_completed(
            db_session,
            job.job_id,
            audio_path="/audio/second.mp3",
            audio_size_bytes=200,
            audio_duration_seconds=2.0,
            estimated_cost_usd=0.02,
        )
        await db_session.commit()
        await db_session.refresh(job)

        assert job.status == "completed"
        assert job.audio_path == "/audio/second.mp3"
        assert job.audio_size_bytes == 200
        assert job.audio_duration_seconds == 2.0
        assert job.estimated_cost_usd == 0.02
        # completed_at was overwritten by the second call; still set.
        assert job.completed_at is not None

    async def test_mark_completed_preserves_started_at_from_mark_processing(
        self, db_session: AsyncSession
    ) -> None:
        """``mark_completed`` must not reset ``started_at`` — only ``completed_at``.

        Documents the behavior at ``job_manager.py:112-126``: the ``values``
        dict intentionally omits ``started_at``, so latency-from-started-at
        math in dashboards stays correct after transition.
        """
        job = await _create_default_job(db_session)

        await JobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()
        await db_session.refresh(job)
        original_started_at = job.started_at
        assert original_started_at is not None

        await JobManager.mark_completed(
            db_session,
            job.job_id,
            audio_path="/audio/final.mp3",
            audio_size_bytes=500,
            audio_duration_seconds=5.0,
        )
        await db_session.commit()
        await db_session.refresh(job)

        assert job.started_at == original_started_at
        assert job.completed_at is not None
        assert job.status == "completed"

    async def test_mark_failed_preserves_started_at_and_leaves_audio_path_unset(
        self, db_session: AsyncSession
    ) -> None:
        """``mark_failed`` after ``mark_processing`` keeps ``started_at``, leaves audio fields ``None``."""
        job = await _create_default_job(db_session)

        await JobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()
        await db_session.refresh(job)
        started_at_before_failure = job.started_at
        assert started_at_before_failure is not None

        await JobManager.mark_failed(db_session, job.job_id, "Provider crashed")
        await db_session.commit()
        await db_session.refresh(job)

        assert job.status == "failed"
        assert job.error_message == "Provider crashed"
        assert job.started_at == started_at_before_failure
        assert job.completed_at is not None
        assert job.audio_path is None
        assert job.audio_size_bytes is None


@pytest.mark.asyncio
class TestSetBatchConcatPathEdgeCases:
    async def test_set_batch_concat_path_on_empty_batch_is_silent_noop(
        self, db_session: AsyncSession
    ) -> None:
        """Updating a batch with no jobs must not raise."""
        await JobManager.set_batch_concat_path(
            db_session, "empty-batch", "/a/b.mp3", tenant_id=tenant_uuid("tenant-1")
        )
        await db_session.commit()

        # Nothing created or modified.
        result = await JobManager.get_batch_jobs(
            db_session, "empty-batch", tenant_id=tenant_uuid("tenant-1")
        )
        assert result == []

    async def test_set_batch_concat_path_only_updates_matching_batch(
        self, db_session: AsyncSession
    ) -> None:
        """Jobs in unrelated batches must not be touched."""
        j_a1 = await _create_default_job(db_session, batch_id="A", text="a1")
        j_a2 = await _create_default_job(db_session, batch_id="A", text="a2")
        j_b = await _create_default_job(db_session, batch_id="B", text="b1")
        j_none = await _create_default_job(db_session, batch_id=None, text="solo")

        await JobManager.set_batch_concat_path(
            db_session, "A", "/concat/a.mp3", tenant_id=tenant_uuid("tenant-1")
        )
        await db_session.commit()

        for j in (j_a1, j_a2, j_b, j_none):
            await db_session.refresh(j)

        assert j_a1.concatenated_audio_path == "/concat/a.mp3"
        assert j_a2.concatenated_audio_path == "/concat/a.mp3"
        assert j_b.concatenated_audio_path is None
        assert j_none.concatenated_audio_path is None
