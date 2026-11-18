"""Real-database tests for JobManager against the postgres CI database."""

from __future__ import annotations

import uuid

import pytest

from src.database.models import TTSJob
from src.services.job_manager import JobManager
from tests.conftest import tenant_uuid


@pytest.mark.asyncio
class TestJobManagerCRUD:
    """Tests for JobManager static methods against the real postgres CI database."""

    # -- helpers ----------------------------------------------------------

    @staticmethod
    async def _create_default_job(
        db,
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

    # -- create_job -------------------------------------------------------

    async def test_create_job_fields(self, db_session):
        """create_job sets all fields correctly including UUID and text_length."""
        job = await self._create_default_job(
            db_session,
            tenant_id=tenant_uuid("t-abc"),
            text="Say something",
            provider="openai",
            voice="alloy",
            speed=1.25,
            output_format="wav",
            model="tts-1-hd",
            batch_id="batch-99",
            caller_item_id="item-7",
            routing_strategy="round_robin",
        )

        assert job.id is not None
        # job_id is a valid UUID7 string
        parsed = uuid.UUID(job.job_id)
        assert parsed.version == 7

        assert job.tenant_id == tenant_uuid("t-abc")
        assert job.text == "Say something"
        assert job.text_length == len("Say something")
        assert job.provider == "openai"
        assert job.voice == "alloy"
        assert job.speed == 1.25
        assert job.output_format == "wav"
        assert job.model == "tts-1-hd"
        assert job.batch_id == "batch-99"
        assert job.caller_item_id == "item-7"
        assert job.routing_strategy == "round_robin"
        assert job.status == "pending"
        assert job.error_message is None
        assert job.audio_path is None

    async def test_create_job_text_length_calculation(self, db_session):
        """text_length matches the actual character count of the text."""
        long_text = "A" * 500
        job = await self._create_default_job(db_session, text=long_text)
        assert job.text_length == 500

    async def test_create_job_defaults(self, db_session):
        """create_job applies correct defaults for optional fields."""
        job = await self._create_default_job(db_session)
        assert job.speed == 1.0
        assert job.output_format == "mp3"
        assert job.model is None
        assert job.batch_id is None
        assert job.caller_item_id is None
        assert job.extra_params is None
        assert job.routing_strategy is None
        assert job.cached is False

    # -- get_job ----------------------------------------------------------

    async def test_get_job_returns_matching_job(self, db_session):
        """get_job retrieves a job by its job_id."""
        created = await self._create_default_job(db_session, text="fetch me", provider="elevenlabs")
        fetched = await JobManager.get_job(
            db_session, created.job_id, tenant_id=tenant_uuid("tenant-1")
        )

        assert fetched is not None
        assert fetched.job_id == created.job_id
        assert fetched.text == "fetch me"
        assert fetched.provider == "elevenlabs"

    async def test_get_job_returns_none_when_not_found(self, db_session):
        """get_job returns None for a non-existent job_id."""
        result = await JobManager.get_job(
            db_session, "non-existent-id", tenant_id=tenant_uuid("tenant-1")
        )
        assert result is None

    # -- get_batch_jobs ---------------------------------------------------

    async def test_get_batch_jobs_returns_all_in_order(self, db_session):
        """get_batch_jobs returns all jobs for a batch_id ordered by id."""
        bid = "batch-ordered"
        j1 = await self._create_default_job(db_session, batch_id=bid, text="first")
        j2 = await self._create_default_job(db_session, batch_id=bid, text="second")
        j3 = await self._create_default_job(db_session, batch_id=bid, text="third")
        # Also create a job with a different batch_id to ensure filtering.
        await self._create_default_job(db_session, batch_id="other-batch", text="other")

        jobs = await JobManager.get_batch_jobs(db_session, bid, tenant_id=tenant_uuid("tenant-1"))

        assert len(jobs) == 3
        assert [j.job_id for j in jobs] == [j1.job_id, j2.job_id, j3.job_id]
        assert [j.text for j in jobs] == ["first", "second", "third"]

    # -- mark_processing --------------------------------------------------

    async def test_mark_processing_updates_status_and_started_at(self, db_session):
        """mark_processing transitions status and sets started_at."""
        job = await self._create_default_job(db_session)
        assert job.status == "pending"
        assert job.started_at is None

        await JobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()

        await db_session.refresh(job)
        assert job.status == "processing"
        assert job.started_at is not None

    # -- mark_completed ---------------------------------------------------

    async def test_mark_completed_sets_audio_fields(self, db_session):
        """mark_completed stores audio metadata and transitions to completed."""
        job = await self._create_default_job(db_session)

        await JobManager.mark_completed(
            db_session,
            job.job_id,
            audio_path="/audio/test.mp3",
            audio_size_bytes=12345,
            audio_duration_seconds=3.5,
            cached=True,
            estimated_cost_usd=0.015,
            latency_ms=250.0,
            detected_language="en",
        )
        await db_session.commit()

        await db_session.refresh(job)
        assert job.status == "completed"
        assert job.audio_path == "/audio/test.mp3"
        assert job.audio_size_bytes == 12345
        assert job.audio_duration_seconds == 3.5
        assert job.cached is True
        assert job.estimated_cost_usd == 0.015
        assert job.latency_ms == 250.0
        assert job.detected_language == "en"
        assert job.completed_at is not None

    async def test_mark_completed_minimal_fields(self, db_session):
        """mark_completed works with only required audio fields."""
        job = await self._create_default_job(db_session)

        await JobManager.mark_completed(
            db_session,
            job.job_id,
            audio_path="/audio/min.mp3",
            audio_size_bytes=100,
            audio_duration_seconds=1.0,
        )
        await db_session.commit()

        await db_session.refresh(job)
        assert job.status == "completed"
        assert job.cached is False
        assert job.estimated_cost_usd is None
        assert job.latency_ms is None
        assert job.detected_language is None

    # -- mark_failed ------------------------------------------------------

    async def test_mark_failed_sets_error_and_status(self, db_session):
        """mark_failed stores the error message and sets status to failed."""
        job = await self._create_default_job(db_session)

        await JobManager.mark_failed(db_session, job.job_id, "Provider timeout")
        await db_session.commit()

        await db_session.refresh(job)
        assert job.status == "failed"
        assert job.error_message == "Provider timeout"
        assert job.completed_at is not None

    # -- set_batch_concat_path -------------------------------------------

    async def test_set_batch_concat_path_updates_all_batch_jobs(self, db_session):
        """set_batch_concat_path sets concatenated_audio_path on all batch jobs."""
        bid = "batch-concat"
        j1 = await self._create_default_job(db_session, batch_id=bid, text="one")
        j2 = await self._create_default_job(db_session, batch_id=bid, text="two")
        # Job in different batch should be unaffected.
        j_other = await self._create_default_job(db_session, batch_id="other", text="other")

        await JobManager.set_batch_concat_path(
            db_session, bid, "/audio/concat.mp3", tenant_id=tenant_uuid("tenant-1")
        )
        await db_session.commit()

        await db_session.refresh(j1)
        await db_session.refresh(j2)
        await db_session.refresh(j_other)

        assert j1.concatenated_audio_path == "/audio/concat.mp3"
        assert j2.concatenated_audio_path == "/audio/concat.mp3"
        assert j_other.concatenated_audio_path is None

    # -- JSONB round-trip -------------------------------------------------

    async def test_extra_params_jsonb_round_trip(self, db_session):
        """extra_params dict survives a database round-trip."""
        params = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "nested": {"key": "value", "list": [1, 2, 3]},
        }
        job = await self._create_default_job(db_session, extra_params=params)

        fetched = await JobManager.get_job(
            db_session, job.job_id, tenant_id=tenant_uuid("tenant-1")
        )
        assert fetched is not None
        assert fetched.extra_params == params
        assert isinstance(fetched.extra_params, dict)
        nested = fetched.extra_params["nested"]
        assert isinstance(nested, dict)
        assert nested["list"] == [1, 2, 3]
