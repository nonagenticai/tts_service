"""Cross-tenant job access must be impossible on every read path.

This service has **no RLS** — unlike most of the fleet, there is no database
backstop here. The ``tenant_id`` predicate inside ``JobManager`` is the entire
isolation boundary, and the endpoints below are the only things that supply it.
Both halves are therefore tested:

* :class:`TestJobManagerTenantPredicate` — the SQL actually filters (real DB).
* :class:`TestEndpointsPassTheCallersTenant` — every endpoint actually passes the
  authenticated caller's tenant down, and renders a foreign job as **404**.

The 404 (never 403) matters: a 403 would confirm that a job id exists, turning
these endpoints into an existence oracle for other tenants' job ids.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.services.job_manager import JobManager
from tests.conftest import tenant_uuid

OWNER = tenant_uuid("tenant-owner")
# How that uuid appears INLINED in compiled SQL. SQLAlchemy's postgres UUID
# literal processor emits the 32-char undashed hex form, not the dashed text
# form, so asserting on `OWNER` itself would never match and these three
# tenant-predicate tests would go vacuously... loud, in fact: they fail.
OWNER_SQL_LITERAL = OWNER.replace("-", "")
ATTACKER = tenant_uuid("tenant-attacker")


# ===========================================================================
# Half 1: the SQL predicate (real database)
# ===========================================================================


@pytest.mark.asyncio
class TestJobManagerTenantPredicate:
    """The ``WHERE tenant_id = :tenant_id`` clauses are load-bearing."""

    @staticmethod
    async def _job(db, *, tenant_id: str, batch_id: str | None = None):
        job = await JobManager.create_job(
            db,
            tenant_id=tenant_id,
            text="Hello world",
            provider="kokoro",
            voice="af_heart",
            batch_id=batch_id,
        )
        await db.commit()
        return job

    async def test_get_job_hides_another_tenants_job(self, db_session):
        """Mutation this kills: drop ``TTSJob.tenant_id == tenant_id`` from
        ``JobManager.get_job``. The attacker's lookup would then return the
        owner's row and ``assert ... is None`` fails.
        """
        job = await self._job(db_session, tenant_id=OWNER)

        assert await JobManager.get_job(db_session, job.job_id, tenant_id=OWNER) is not None
        assert await JobManager.get_job(db_session, job.job_id, tenant_id=ATTACKER) is None

    async def test_get_batch_jobs_hides_another_tenants_batch(self, db_session):
        """Mutation this kills: drop ``TTSJob.tenant_id == tenant_id`` from
        ``JobManager.get_batch_jobs`` — the attacker would receive the owner's
        two jobs instead of an empty list.
        """
        batch_id = "batch-shared-id"
        await self._job(db_session, tenant_id=OWNER, batch_id=batch_id)
        await self._job(db_session, tenant_id=OWNER, batch_id=batch_id)

        assert len(await JobManager.get_batch_jobs(db_session, batch_id, tenant_id=OWNER)) == 2
        assert await JobManager.get_batch_jobs(db_session, batch_id, tenant_id=ATTACKER) == []

    async def test_get_batch_jobs_returns_only_the_callers_rows_in_a_shared_batch_id(
        self, db_session
    ):
        """A batch id colliding across tenants must never merge their rows.

        Mutation this kills: the same dropped predicate — the count would be 2,
        and the attacker's concatenated download would contain the owner's audio.
        """
        batch_id = "batch-collision"
        await self._job(db_session, tenant_id=OWNER, batch_id=batch_id)
        await self._job(db_session, tenant_id=ATTACKER, batch_id=batch_id)

        owner_jobs = await JobManager.get_batch_jobs(db_session, batch_id, tenant_id=OWNER)
        assert len(owner_jobs) == 1
        assert owner_jobs[0].tenant_id == OWNER

    async def test_set_batch_concat_path_cannot_stamp_another_tenants_rows(self, db_session):
        """Mutation this kills: drop the tenant predicate from
        ``JobManager.set_batch_concat_path``. The owner's row would acquire the
        attacker's storage key, which ``download_batch_audio`` then serves.
        """
        batch_id = "batch-collision-write"
        owner_job = await self._job(db_session, tenant_id=OWNER, batch_id=batch_id)

        await JobManager.set_batch_concat_path(
            db_session, batch_id, "/evil/attacker.mp3", tenant_id=ATTACKER
        )
        await db_session.commit()

        refreshed = await JobManager.get_job(db_session, owner_job.job_id, tenant_id=OWNER)
        assert refreshed is not None
        assert refreshed.concatenated_audio_path is None


# ===========================================================================
# Half 1b: the same predicate, provable WITHOUT a database
#
# The tests above are the real proof, but they need postgres and therefore only
# run where a DB is reachable. These compile the emitted SQL instead, so the
# single most important invariant in this module is verified on every machine —
# including a laptop with no database up.
# ===========================================================================


class _CapturingSession:
    """Async-session stand-in that records the statement and returns nothing.

    Typed as ``Any`` at the call sites below: it is a structural stand-in for
    ``AsyncSession`` (only ``execute`` is ever reached), and spelling that out
    keeps the tests free of type-checker suppressions.
    """

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        self.statements.append(statement)
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        result.scalars.return_value.all.return_value = []
        return result

    def compiled_sql(self) -> str:
        assert len(self.statements) == 1, self.statements
        return str(self.statements[0].compile(compile_kwargs={"literal_binds": True}))


def _session() -> Any:
    """A fresh capturing session, typed ``Any`` for the checker."""
    return _CapturingSession()


def _unchecked(obj: object) -> Any:
    """Return ``obj`` erased to ``Any``.

    Used only to make DELIBERATELY invalid calls invisible to the type checker,
    so a test can assert the runtime ``TypeError`` without the file failing
    ``ty check``.
    """
    return obj


def _await_kwargs(mock: Any) -> dict[str, Any]:
    """The kwargs of a mock's single await, asserting it was actually awaited."""
    assert mock.await_args is not None, "expected the call to have been awaited"
    return dict(mock.await_args.kwargs)


@pytest.mark.asyncio
class TestEmittedSQLCarriesTheTenantPredicate:
    """Every tenant-scoped JobManager statement must name ``tenant_id`` in WHERE.

    Shared mutation these kill: delete ``TTSJob.tenant_id == tenant_id`` from any
    of the three statements. The compiled SQL loses the clause and the assertion
    fails — with no database involved.
    """

    async def test_get_job_sql_filters_on_tenant(self):
        session = _session()
        await JobManager.get_job(session, "job-1", tenant_id=OWNER)

        sql = session.compiled_sql()
        assert f"tts_jobs.tenant_id = '{OWNER_SQL_LITERAL}'" in sql, sql
        assert "tts_jobs.job_id = 'job-1'" in sql, sql

    async def test_get_batch_jobs_sql_filters_on_tenant(self):
        session = _session()
        await JobManager.get_batch_jobs(session, "batch-1", tenant_id=OWNER)

        sql = session.compiled_sql()
        assert f"tts_jobs.tenant_id = '{OWNER_SQL_LITERAL}'" in sql, sql
        assert "tts_jobs.batch_id = 'batch-1'" in sql, sql

    async def test_set_batch_concat_path_sql_filters_on_tenant(self):
        session = _session()
        await JobManager.set_batch_concat_path(
            session,
            "batch-1",
            "/audio/concat.mp3",
            tenant_id=OWNER,
        )

        sql = session.compiled_sql()
        assert f"tts_jobs.tenant_id = '{OWNER_SQL_LITERAL}'" in sql, sql
        assert "UPDATE" in sql.upper(), sql

    async def test_tenant_id_is_required_and_keyword_only(self):
        """The signature itself is the guard rail — no caller can forget it.

        Mutation this kills: give ``tenant_id`` a default (e.g. ``= ""``) or make
        it positional-optional. A future unscoped caller would then compile and
        silently read across tenants; here it raises ``TypeError`` instead.

        ``ty`` already rejects these calls statically (``missing-argument`` /
        ``too-many-positional-arguments``), which is the first line of defence.
        They are routed through an untyped alias so the file still type-checks
        while the RUNTIME guarantee — the one that holds for dynamically
        constructed calls the checker never sees — is asserted here too.
        """
        session = _session()
        manager = _unchecked(JobManager)

        with pytest.raises(TypeError):
            await manager.get_job(session, "job-1")
        with pytest.raises(TypeError):
            await manager.get_batch_jobs(session, "batch-1")
        with pytest.raises(TypeError):
            # Positional must not satisfy it either.
            await manager.get_job(session, "job-1", OWNER)


# ===========================================================================
# Half 2: the endpoints supply the tenant (no database needed)
# ===========================================================================


def _request(tenant_id: str) -> MagicMock:
    """A request whose auth middleware resolved ``tenant_id``."""
    request = MagicMock()
    request.state.tenant_id = tenant_id
    storage = AsyncMock()
    # If an endpoint ever reached storage for a foreign job, these would be the
    # leak. The tests below assert they are never called.
    storage.retrieve = AsyncMock(return_value=b"secret-audio-bytes")
    storage.get_presigned_url = AsyncMock(return_value="https://minio/presigned-secret")
    request.app.state.storage_service = storage
    return request


class TestEndpointsPassTheCallersTenant:
    """Every job/batch read endpoint must scope by the authenticated tenant.

    Each test patches ``JobManager`` and asserts on the ``tenant_id`` kwarg, so
    the shared mutation these all kill is: **remove ``tenant_id=_get_tenant_id(
    request)`` from the endpoint's ``JobManager`` call.** That is a TypeError
    against the real signature and an assertion failure against the mock — the
    endpoint can no longer silently go unscoped.
    """

    @pytest.mark.asyncio
    async def test_get_job_status_scopes_and_404s_without_minting_a_presigned_url(self):
        """The worst of the three: this response body carries a presigned MinIO
        URL that needs no auth once handed out.

        Mutation this kills: unscoped ``get_job`` here — the mock would be called
        without ``tenant_id=ATTACKER`` and the kwarg assertion fails. The
        ``get_presigned_url`` assertion additionally proves no URL is minted on
        the reject path.
        """
        from src.api.tts import get_job_status

        request = _request(ATTACKER)
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)  # foreign job ⇒ invisible
            with pytest.raises(HTTPException) as exc_info:
                await get_job_status("owners-job-id", request, db)

        assert exc_info.value.status_code == 404, "must be 404, not 403 — no existence oracle"
        assert _await_kwargs(mock_jm.get_job)["tenant_id"] == ATTACKER
        request.app.state.storage_service.get_presigned_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_download_audio_scopes_and_404s_without_reading_storage(self):
        """Mutation this kills: unscoped ``get_job`` in ``download_audio``."""
        from src.api.tts import download_audio

        request = _request(ATTACKER)
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await download_audio("owners-job-id", request, db)

        assert exc_info.value.status_code == 404
        assert _await_kwargs(mock_jm.get_job)["tenant_id"] == ATTACKER
        request.app.state.storage_service.retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_audio_presigned_url_scopes_and_404s(self):
        """The fourth unscoped endpoint — not in the original report, found by
        grepping every ``get_job`` caller rather than fixing only the named ones.

        Mutation this kills: unscoped ``get_job`` in ``get_audio_presigned_url``.
        """
        from src.api.tts import get_audio_presigned_url

        request = _request(ATTACKER)
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_audio_presigned_url("owners-job-id", request, db)

        assert exc_info.value.status_code == 404
        assert _await_kwargs(mock_jm.get_job)["tenant_id"] == ATTACKER
        request.app.state.storage_service.get_presigned_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_download_batch_audio_scopes_and_404s(self):
        """Mutation this kills: unscoped ``get_batch_jobs`` in
        ``download_batch_audio``.
        """
        from src.api.tts import download_batch_audio

        request = _request(ATTACKER)
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[])
            with pytest.raises(HTTPException) as exc_info:
                await download_batch_audio("owners-batch-id", request, db)

        assert exc_info.value.status_code == 404
        assert _await_kwargs(mock_jm.get_batch_jobs)["tenant_id"] == ATTACKER
        request.app.state.storage_service.retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_job_scopes_by_the_tenant_it_bills(self):
        """The background worker must not synthesise/bill a job it does not own.

        Mutation this kills: unscoped ``get_job`` in ``TTSService.process_job``
        — the kwarg assertion fails. (The ``ValueError`` is the endpoint-free
        equivalent of the 404: an unknown job is indistinguishable from a
        foreign one.)
        """
        from src.services.tts_service import TTSService

        service = TTSService(registry=MagicMock(), storage=AsyncMock())
        db = AsyncMock()

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(ValueError, match="Job not found"):
                await service.process_job(db, "owners-job-id", ATTACKER)

        assert _await_kwargs(mock_jm.get_job)["tenant_id"] == ATTACKER

    @pytest.mark.asyncio
    async def test_batch_concat_background_carries_the_creating_tenant(self):
        """The detached background task must remember whose batch it works on.

        Mutation this kills: drop ``tenant_id=tenant_id`` from either
        ``get_batch_jobs`` call inside ``_concatenate_batch_background`` (or stop
        threading ``tenant_id`` in from ``create_batch``).
        """
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()
        mock_db = AsyncMock()

        with (
            patch("src.api.tts.get_db_session") as mock_get_db,
            patch("src.api.tts.JobManager") as mock_jm,
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_jm.get_batch_jobs = AsyncMock(return_value=[])

            await _concatenate_batch_background(mock_tts, "batch-1", "mp3", 0, tenant_id=OWNER)

        assert _await_kwargs(mock_jm.get_batch_jobs)["tenant_id"] == OWNER
