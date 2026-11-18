"""Real-database tests for the usage dashboard, quota, and alerts endpoints.

These rewrite a previous mock-heavy suite that only asserted mocks echoed
their inputs. Every test here constructs real rows via the ``db_session``
postgres fixture, invokes the endpoint function, and asserts on the
aggregation math, filtering, ordering, and persistence behavior.

The endpoint functions are called directly — the ``Request`` parameter is
only ever read for ``request.state.tenant_id`` (plus ``.roles`` by
``update_quota``), so a ``SimpleNamespace`` stand-in is sufficient and typed via ``cast`` (no ``type: ignore``).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request as StarletteRequest

from src.api.schemas import QuotaUpdateRequest
from src.api.usage import get_alerts, get_dashboard, get_quota, update_quota
from src.api.usage import router as usage_router
from src.database.models import TenantQuota, TTSUsage, UsageAlert
from src.database.session import get_db
from tests.conftest import tenant_uuid


def _make_request(tenant_id: str | None, roles: tuple[str, ...] | None = None) -> Request:
    """Minimal ``Request`` stand-in.

    The endpoints read ``.state.tenant_id``; ``update_quota`` also reads
    ``.state.roles``. ``roles=None`` leaves the attribute UNSET, which is the
    license-token / anonymous shape where AuthMiddleware stashes no roles.
    """
    state = SimpleNamespace(tenant_id=tenant_id)
    if roles is not None:
        state.roles = list(roles)
    return cast(Request, SimpleNamespace(state=state))


_SUPER_ADMIN = ("super_admin",)


async def _mk_usage(
    db: AsyncSession,
    *,
    tenant_id: str,
    day: _dt.date,
    provider: str,
    request_count: int = 1,
    total_characters: int = 100,
    total_audio_seconds: float = 1.0,
    error_count: int = 0,
    estimated_cost_usd: float = 0.0,
    avg_latency_ms: float = 0.0,
    cache_hits: int = 0,
    cache_misses: int = 0,
) -> TTSUsage:
    row = TTSUsage(
        tenant_id=tenant_id,
        date=day,
        provider=provider,
        request_count=request_count,
        total_characters=total_characters,
        total_audio_seconds=total_audio_seconds,
        error_count=error_count,
        estimated_cost_usd=estimated_cost_usd,
        avg_latency_ms=avg_latency_ms,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    db.add(row)
    await db.commit()
    return row


async def _mk_alert(
    db: AsyncSession,
    *,
    tenant_id: str,
    created_at: _dt.datetime,
    acknowledged: bool = False,
    alert_type: str = "budget_warning",
    message: str = "test alert",
) -> UsageAlert:
    alert = UsageAlert(
        tenant_id=tenant_id,
        alert_type=alert_type,
        threshold_pct=80,
        current_value=40.0,
        limit_value=50.0,
        period="monthly",
        message=message,
        acknowledged=acknowledged,
        created_at=created_at,
    )
    db.add(alert)
    await db.commit()
    return alert


# ============================================================================
# get_dashboard
# ============================================================================


@pytest.mark.asyncio
class TestGetDashboardReal:
    async def test_empty_returns_empty_series(self, db_session: AsyncSession) -> None:
        result = await get_dashboard(_make_request(tenant_uuid("t1")), db_session, days=7)
        assert result.tenant_id == tenant_uuid("t1")
        assert result.days == 7
        assert result.usage_over_time == []
        assert result.provider_distribution == []

    async def test_sums_request_count_per_day(self, db_session: AsyncSession) -> None:
        """The day bucket in ``usage_over_time`` is a sum across providers.

        Tripwire for ``usage.py:219`` — anyone who regresses the day bucket
        to per-row instead of per-date aggregation causes this test to fail.
        """
        today = _dt.date.today()
        yesterday = today - _dt.timedelta(days=1)
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="kokoro",
            request_count=10,
            total_characters=1000,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="openai",
            request_count=5,
            total_characters=500,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=20,
            total_characters=2000,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="elevenlabs",
            request_count=3,
            total_characters=300,
        )

        result = await get_dashboard(_make_request(tenant_uuid("t1")), db_session, days=7)

        by_date = {point.date: point for point in result.usage_over_time}
        assert by_date[yesterday.isoformat()].request_count == 15
        assert by_date[yesterday.isoformat()].total_characters == 1500
        assert by_date[today.isoformat()].request_count == 23
        assert by_date[today.isoformat()].total_characters == 2300

    async def test_provider_distribution_uses_weighted_average_latency(
        self, db_session: AsyncSession
    ) -> None:
        """Weighted latency formula at ``usage.py:240`` / ``:252``.

        For ``{kokoro: 100 req @ 200ms}`` and ``{kokoro: 50 req @ 400ms}``
        the expected weighted average is
        ``(100*200 + 50*400) / 150 == 266.666...``.
        If anyone rewrites this as a naive average of the two
        ``avg_latency_ms`` values (``(200+400)/2 == 300``), this test fails.
        """
        today = _dt.date.today()
        yesterday = today - _dt.timedelta(days=1)

        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="kokoro",
            request_count=100,
            avg_latency_ms=200.0,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=50,
            avg_latency_ms=400.0,
        )

        result = await get_dashboard(_make_request(tenant_uuid("t1")), db_session, days=7)

        kokoro = next(p for p in result.provider_distribution if p.provider == "kokoro")
        assert kokoro.request_count == 150
        expected = (100 * 200.0 + 50 * 400.0) / 150
        assert kokoro.avg_latency_ms == pytest.approx(expected)

    async def test_filters_by_tenant(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=10,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t2"),
            day=today,
            provider="kokoro",
            request_count=99,
        )

        result = await get_dashboard(_make_request(tenant_uuid("t1")), db_session, days=7)

        # Only t1's row is visible; t2's 99-req row must not leak in.
        assert len(result.usage_over_time) == 1
        assert result.usage_over_time[0].request_count == 10

    async def test_filters_by_date_window(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        inside = today - _dt.timedelta(days=2)
        outside = today - _dt.timedelta(days=30)

        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=inside,
            provider="kokoro",
            request_count=5,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=outside,
            provider="kokoro",
            request_count=999,
        )

        result = await get_dashboard(_make_request(tenant_uuid("t1")), db_session, days=7)

        assert len(result.usage_over_time) == 1
        assert result.usage_over_time[0].date == inside.isoformat()
        assert result.usage_over_time[0].request_count == 5

    async def test_usage_over_time_is_sorted_ascending_by_date(
        self, db_session: AsyncSession
    ) -> None:
        """Pins the ``sorted(daily_map.items())`` call at ``usage.py:242``."""
        today = _dt.date.today()
        day_minus_3 = today - _dt.timedelta(days=3)
        day_minus_1 = today - _dt.timedelta(days=1)

        # Insert out of order to prove the sort is real (not insertion-order).
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=day_minus_1,
            provider="kokoro",
            request_count=1,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=day_minus_3,
            provider="kokoro",
            request_count=1,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=1,
        )

        result = await get_dashboard(_make_request(tenant_uuid("t1")), db_session, days=7)

        dates = [point.date for point in result.usage_over_time]
        assert dates == [
            day_minus_3.isoformat(),
            day_minus_1.isoformat(),
            today.isoformat(),
        ]


# ============================================================================
# update_quota
# ============================================================================


@pytest.mark.asyncio
class TestUpdateQuotaReal:
    async def test_creates_new_quota_when_missing(self, db_session: AsyncSession) -> None:
        body = QuotaUpdateRequest(monthly_budget_usd=50.0)
        result = await update_quota(
            body, _make_request(tenant_uuid("new-tenant"), _SUPER_ADMIN), db_session
        )

        assert result.tenant_id == tenant_uuid("new-tenant")
        assert result.monthly_budget_usd == 50.0

        # Re-query TenantQuota directly to verify persistence.
        row = (
            await db_session.execute(
                select(TenantQuota).where(TenantQuota.tenant_id == tenant_uuid("new-tenant"))
            )
        ).scalar_one()
        assert row.monthly_budget_usd == 50.0

    async def test_modifies_existing_quota(self, db_session: AsyncSession) -> None:
        db_session.add(TenantQuota(tenant_id=tenant_uuid("t1"), monthly_budget_usd=10.0))
        await db_session.commit()

        body = QuotaUpdateRequest(monthly_budget_usd=100.0)
        result = await update_quota(
            body, _make_request(tenant_uuid("t1"), _SUPER_ADMIN), db_session
        )

        assert result.monthly_budget_usd == 100.0

        row = (
            await db_session.execute(
                select(TenantQuota).where(TenantQuota.tenant_id == tenant_uuid("t1"))
            )
        ).scalar_one()
        assert row.monthly_budget_usd == 100.0

    async def test_partial_update_preserves_unrelated_fields(
        self, db_session: AsyncSession
    ) -> None:
        """Only fields set on the request body should be touched.

        The tautological mock version of this test could not possibly
        catch a regression like "always overwrite all fields with body
        values, using ``None`` as a sentinel". This one does.
        """
        db_session.add(
            TenantQuota(
                tenant_id=tenant_uuid("t1"),
                monthly_budget_usd=10.0,
                monthly_character_limit=1000,
                monthly_request_limit=50,
                daily_request_limit=5,
            )
        )
        await db_session.commit()

        body = QuotaUpdateRequest(monthly_budget_usd=75.0)  # only the budget
        await update_quota(body, _make_request(tenant_uuid("t1"), _SUPER_ADMIN), db_session)

        row = (
            await db_session.execute(
                select(TenantQuota).where(TenantQuota.tenant_id == tenant_uuid("t1"))
            )
        ).scalar_one()
        assert row.monthly_budget_usd == 75.0
        assert row.monthly_character_limit == 1000
        assert row.monthly_request_limit == 50
        assert row.daily_request_limit == 5

    async def test_returns_fresh_monthly_usage_from_tail_get_quota_call(
        self, db_session: AsyncSession
    ) -> None:
        """``update_quota`` ends with ``return await get_quota(...)`` — the
        returned usage must reflect real month-to-date rows, not zero.

        Pins the tail call at ``usage.py:415`` and the month-to-date
        aggregation path inside ``get_quota``.
        """
        # Seed a quota.
        db_session.add(TenantQuota(tenant_id=tenant_uuid("t1"), monthly_budget_usd=200.0))
        # Seed a usage row inside the current month.
        today = _dt.date.today()
        month_start = today.replace(day=1)
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=month_start,  # guaranteed to be >= month_start
            provider="kokoro",
            request_count=7,
            total_characters=5000,
            estimated_cost_usd=0.42,
        )

        body = QuotaUpdateRequest(monthly_budget_usd=300.0)
        result = await update_quota(
            body, _make_request(tenant_uuid("t1"), _SUPER_ADMIN), db_session
        )

        assert result.monthly_budget_usd == 300.0
        assert result.monthly_requests_used == 7
        assert result.monthly_characters_used == 5000
        assert result.monthly_budget_used_usd == pytest.approx(0.42)


@pytest.mark.asyncio
class TestUpdateQuotaIsSuperAdminOnly:
    """B20 / Decision 12: quota LIMITS are operator-owned.

    Before the gate, ``PUT /api/tts/usage/quota`` let any authenticated tenant
    rewrite its own ``tenant_quotas`` row — i.e. lift its own budget and request
    caps. Each test here asserts the row is UNCHANGED after a refused write, not
    just that an exception was raised, so a gate placed after the flush would
    still fail them.
    """

    @pytest.mark.parametrize(
        "roles",
        [
            pytest.param(None, id="license-token-no-roles-attr"),
            pytest.param((), id="jwt-no-roles"),
            pytest.param(("member",), id="member"),
            pytest.param(("tenant_admin",), id="tenant_admin"),
        ],
    )
    async def test_non_super_admin_gets_403_and_row_is_untouched(
        self, db_session: AsyncSession, roles: tuple[str, ...] | None
    ) -> None:
        db_session.add(
            TenantQuota(tenant_id=tenant_uuid("t1"), monthly_budget_usd=10.0, daily_request_limit=5)
        )
        await db_session.commit()

        body = QuotaUpdateRequest(monthly_budget_usd=1_000_000.0, daily_request_limit=999_999)
        with pytest.raises(HTTPException) as exc_info:
            await update_quota(body, _make_request(tenant_uuid("t1"), roles), db_session)
        assert exc_info.value.status_code == 403

        db_session.expire_all()
        row = (
            await db_session.execute(
                select(TenantQuota).where(TenantQuota.tenant_id == tenant_uuid("t1"))
            )
        ).scalar_one()
        assert row.monthly_budget_usd == 10.0
        assert row.daily_request_limit == 5

    async def test_non_super_admin_cannot_create_a_quota_row(
        self, db_session: AsyncSession
    ) -> None:
        body = QuotaUpdateRequest(monthly_budget_usd=50.0)
        with pytest.raises(HTTPException) as exc_info:
            await update_quota(body, _make_request(tenant_uuid("t1"), ()), db_session)
        assert exc_info.value.status_code == 403
        count = (await db_session.execute(select(func.count()).select_from(TenantQuota))).scalar()
        assert count == 0

    async def test_missing_tenant_is_401_not_403(self, db_session: AsyncSession) -> None:
        """Unauthenticated stays 401 — the role check runs after tenant resolution."""
        body = QuotaUpdateRequest(monthly_budget_usd=50.0)
        with pytest.raises(HTTPException) as exc_info:
            await update_quota(body, _make_request(None, ()), db_session)
        assert exc_info.value.status_code == 401

    async def test_tenant_can_still_read_own_quota(self, db_session: AsyncSession) -> None:
        db_session.add(TenantQuota(tenant_id=tenant_uuid("t1"), monthly_budget_usd=42.0))
        db_session.add(TenantQuota(tenant_id=tenant_uuid("t2"), monthly_budget_usd=7.0))
        await db_session.commit()

        result = await get_quota(_make_request(tenant_uuid("t1"), ("member",)), db_session)
        assert result.tenant_id == tenant_uuid("t1")
        assert result.monthly_budget_usd == 42.0


@pytest.mark.asyncio
class TestQuotaRouteOverHttp:
    """The same contract through the real router, so the route wiring is proven,
    not only the handler body."""

    @staticmethod
    def _client(db_session: AsyncSession) -> httpx.AsyncClient:
        app = FastAPI()

        @app.middleware("http")
        async def _seed_auth_state(request: StarletteRequest, call_next):
            # Stand-in for w2v_common AuthMiddleware: tenant + realm roles on state.
            request.state.tenant_id = request.headers.get("x-test-tenant")
            roles = request.headers.get("x-test-roles")
            request.state.roles = roles.split(",") if roles else []
            return await call_next(request)

        app.include_router(usage_router)

        async def _db() -> AsyncIterator[AsyncSession]:
            yield db_session

        app.dependency_overrides[get_db] = _db
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")

    async def test_tenant_put_403_get_200_super_admin_put_200(
        self, db_session: AsyncSession
    ) -> None:
        tenant = tenant_uuid("t1")
        async with self._client(db_session) as client:
            denied = await client.put(
                "/api/tts/usage/quota",
                json={"monthly_budget_usd": 1_000_000.0},
                headers={"x-test-tenant": tenant, "x-test-roles": "member,tenant_admin"},
            )
            assert denied.status_code == 403

            read = await client.get("/api/tts/usage/quota", headers={"x-test-tenant": tenant})
            assert read.status_code == 200
            assert read.json()["tenant_id"] == tenant
            assert read.json()["monthly_budget_usd"] is None

            allowed = await client.put(
                "/api/tts/usage/quota",
                json={"monthly_budget_usd": 50.0},
                headers={"x-test-tenant": tenant, "x-test-roles": "super_admin"},
            )
            assert allowed.status_code == 200
            assert allowed.json()["monthly_budget_usd"] == 50.0

            reread = await client.get("/api/tts/usage/quota", headers={"x-test-tenant": tenant})
            assert reread.json()["monthly_budget_usd"] == 50.0


# ============================================================================
# get_alerts
# ============================================================================


@pytest.mark.asyncio
class TestGetAlertsReal:
    async def test_empty(self, db_session: AsyncSession) -> None:
        result = await get_alerts(_make_request(tenant_uuid("t1")), db_session)
        assert result == []

    async def test_filter_by_acknowledged_false(self, db_session: AsyncSession) -> None:
        now = _dt.datetime.now(_dt.UTC)
        await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("t1"),
            created_at=now,
            acknowledged=True,
            message="acked-1",
        )
        await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("t1"),
            created_at=now,
            acknowledged=True,
            message="acked-2",
        )
        for i in range(3):
            await _mk_alert(
                db_session,
                tenant_id=tenant_uuid("t1"),
                created_at=now,
                acknowledged=False,
                message=f"unacked-{i}",
            )

        result = await get_alerts(_make_request(tenant_uuid("t1")), db_session, acknowledged=False)

        assert len(result) == 3
        assert all(not a.acknowledged for a in result)
        assert {a.message for a in result} == {"unacked-0", "unacked-1", "unacked-2"}

    async def test_ordered_by_created_at_desc(self, db_session: AsyncSession) -> None:
        """Newest-first ordering at ``usage.py:436``.

        Explicit ``created_at`` values ensure deterministic ordering —
        ``func.now()`` defaults could give identical timestamps for
        inserts in a single transaction.
        """
        base = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
        await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("t1"),
            created_at=base,
            message="oldest",
        )
        await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("t1"),
            created_at=base + _dt.timedelta(seconds=10),
            message="middle",
        )
        await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("t1"),
            created_at=base + _dt.timedelta(seconds=20),
            message="newest",
        )

        result = await get_alerts(_make_request(tenant_uuid("t1")), db_session)

        assert [a.message for a in result] == ["newest", "middle", "oldest"]

    async def test_limit_caps_result_length(self, db_session: AsyncSession) -> None:
        base = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
        for i in range(15):
            await _mk_alert(
                db_session,
                tenant_id=tenant_uuid("t1"),
                created_at=base + _dt.timedelta(seconds=i),
                message=f"a-{i}",
            )

        result = await get_alerts(_make_request(tenant_uuid("t1")), db_session, limit=5)
        assert len(result) == 5
        # And limit=5 combined with desc ordering → we get the newest 5.
        assert [a.message for a in result] == [
            "a-14",
            "a-13",
            "a-12",
            "a-11",
            "a-10",
        ]

    async def test_tenant_isolation(self, db_session: AsyncSession) -> None:
        now = _dt.datetime.now(_dt.UTC)
        await _mk_alert(db_session, tenant_id=tenant_uuid("t1"), created_at=now, message="t1-a")
        await _mk_alert(db_session, tenant_id=tenant_uuid("t1"), created_at=now, message="t1-b")
        await _mk_alert(db_session, tenant_id=tenant_uuid("t2"), created_at=now, message="t2-a")

        result = await get_alerts(_make_request(tenant_uuid("t1")), db_session)

        assert len(result) == 2
        assert all(a.tenant_id == tenant_uuid("t1") for a in result)
        assert {a.message for a in result} == {"t1-a", "t1-b"}
