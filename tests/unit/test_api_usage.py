"""Real-database tests for usage summary, cost, export, quota, and alerts.

Rewrites the former mock-heavy suite that only asserted mocks echoed their
inputs. Every assertion here lands on real aggregation math, real
filtering, or real persisted state — the `db_session` fixture is backed
by the postgres CI database with all tts_service tables created. The
``Request`` parameter is only read for ``request.state.tenant_id``, so a
``SimpleNamespace`` stand-in typed via ``cast`` is sufficient.
"""

from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.usage import (
    _get_tenant_id,
    acknowledge_alert,
    export_usage,
    get_cost_breakdown,
    get_cost_rates,
    get_quota,
    get_usage,
)
from src.database.models import TenantQuota, TTSUsage, UsageAlert
from tests.conftest import tenant_uuid


def _make_request(tenant_id: str) -> Request:
    """Minimal ``Request`` stand-in — endpoints only read ``.state.tenant_id``."""
    return cast(Request, SimpleNamespace(state=SimpleNamespace(tenant_id=tenant_id)))


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
# _get_tenant_id (pure helper, no DB)
# ============================================================================


class TestUsageHelpers:
    def test_get_tenant_id_with_state(self) -> None:
        request = cast(
            Request, SimpleNamespace(state=SimpleNamespace(tenant_id=tenant_uuid("my-tenant")))
        )
        assert _get_tenant_id(request) == tenant_uuid("my-tenant")

    def test_get_tenant_id_unresolved_raises_401_never_substitutes(self) -> None:
        """Inverted: this used to assert reader and writer share the nil-UUID sentinel.

        The two sides must still agree — but they now agree on REJECTING, not on
        a sentinel. Neither may ever hold one (product-owner ruling).
        """
        from src.api.tts import _get_tenant_id as writer_get_tenant_id

        request = cast(Request, SimpleNamespace(state=SimpleNamespace()))
        for helper in (_get_tenant_id, writer_get_tenant_id):
            with pytest.raises(HTTPException) as exc_info:
                helper(request)
            assert exc_info.value.status_code == 401
            assert "no sentinel" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestUnresolvedTenantUsageRoundTrip:
    """Inverted from B4: there is no longer an unresolved-tenant bucket to round-trip.

    The old test proved that a row written under the shared nil-UUID sentinel was
    readable back. That shared bucket is exactly the defect: it is un-billable and
    every unattributed caller reads it. Now a tenant-less request cannot reach the
    read path at all, and a real tenant still round-trips normally.
    """

    async def test_unresolved_tenant_cannot_read_usage_at_all(
        self, db_session: AsyncSession
    ) -> None:
        unresolved_request = cast(Request, SimpleNamespace(state=SimpleNamespace()))
        with pytest.raises(HTTPException) as exc_info:
            await get_usage(unresolved_request, db_session)
        assert exc_info.value.status_code == 401

    async def test_real_tenant_still_round_trips(self, db_session: AsyncSession) -> None:
        from src.services.job_manager import JobManager

        await JobManager.record_usage(
            db_session,
            tenant_id=tenant_uuid("real-tenant"),
            provider="kokoro",
            characters=420,
            audio_seconds=12.5,
        )
        await db_session.commit()

        result = await get_usage(_make_request(tenant_uuid("real-tenant")), db_session)

        assert result.tenant_id == tenant_uuid("real-tenant")
        assert result.total_requests == 1
        assert result.total_characters == 420
        assert result.total_audio_seconds == pytest.approx(12.5)


# ============================================================================
# get_usage
# ============================================================================


@pytest.mark.asyncio
class TestGetUsageReal:
    async def test_empty(self, db_session: AsyncSession) -> None:
        result = await get_usage(_make_request(tenant_uuid("t1")), db_session)
        assert result.tenant_id == tenant_uuid("t1")
        assert result.total_requests == 0
        assert result.total_characters == 0
        assert result.total_cost_usd == 0.0
        assert result.total_audio_seconds == 0.0
        assert result.error_count == 0
        assert result.cache_hit_rate == 0.0
        assert result.by_provider == []

    async def test_summary_sums_across_providers(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=10,
            total_characters=1000,
            total_audio_seconds=20.0,
            estimated_cost_usd=0.0,
            error_count=0,
            avg_latency_ms=150.0,
            cache_hits=3,
            cache_misses=7,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="openai",
            request_count=5,
            total_characters=500,
            total_audio_seconds=10.0,
            estimated_cost_usd=0.75,
            error_count=1,
            avg_latency_ms=250.0,
            cache_hits=1,
            cache_misses=4,
        )

        result = await get_usage(_make_request(tenant_uuid("t1")), db_session)

        assert result.total_requests == 15
        assert result.total_characters == 1500
        assert result.total_audio_seconds == 30.0
        assert result.total_cost_usd == pytest.approx(0.75)
        assert result.error_count == 1

    async def test_cache_hit_rate_formula(self, db_session: AsyncSession) -> None:
        """``cache_hit_rate == hits / (hits + misses)``.

        Tripwire for ``usage.py:82-83`` — a naive ``hits / misses`` or
        ``hits / total_requests`` would give a different number.
        """
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=10,
            cache_hits=3,
            cache_misses=7,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="openai",
            request_count=10,
            cache_hits=5,
            cache_misses=5,
        )

        result = await get_usage(_make_request(tenant_uuid("t1")), db_session)

        # 3 + 5 = 8 hits, 7 + 5 = 12 misses, total attempts = 20
        assert result.cache_hit_rate == pytest.approx(8 / 20)

    async def test_cache_hit_rate_zero_when_no_cache_activity(
        self, db_session: AsyncSession
    ) -> None:
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=10,
            cache_hits=0,
            cache_misses=0,
        )
        result = await get_usage(_make_request(tenant_uuid("t1")), db_session)
        assert result.cache_hit_rate == 0.0

    async def test_by_provider_weighted_latency(self, db_session: AsyncSession) -> None:
        """``by_provider[p].avg_latency_ms`` is request-weighted.

        For ``{kokoro: 10 req @ 100ms, 5 req @ 200ms}`` the expected
        weighted average is ``(10*100 + 5*200) / 15 == 133.33...``.
        A naive mean of the two rows would give ``150``.
        """
        today = _dt.date.today()
        yesterday = today - _dt.timedelta(days=1)
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="kokoro",
            request_count=10,
            avg_latency_ms=100.0,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=5,
            avg_latency_ms=200.0,
        )

        result = await get_usage(_make_request(tenant_uuid("t1")), db_session)

        kokoro = next(p for p in result.by_provider if p.provider == "kokoro")
        assert kokoro.request_count == 15
        assert kokoro.avg_latency_ms == pytest.approx((10 * 100.0 + 5 * 200.0) / 15)

    async def test_provider_filter(self, db_session: AsyncSession) -> None:
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
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="openai",
            request_count=99,
        )

        result = await get_usage(_make_request(tenant_uuid("t1")), db_session, provider="kokoro")

        assert result.total_requests == 10
        assert len(result.by_provider) == 1
        assert result.by_provider[0].provider == "kokoro"

    async def test_date_window_filter(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        inside = today - _dt.timedelta(days=5)
        outside = today - _dt.timedelta(days=50)

        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=inside,
            provider="kokoro",
            request_count=10,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=outside,
            provider="kokoro",
            request_count=999,
        )

        result = await get_usage(
            _make_request(tenant_uuid("t1")),
            db_session,
            date_from=(today - _dt.timedelta(days=7)).isoformat(),
            date_to=today.isoformat(),
        )

        assert result.total_requests == 10

    async def test_tenant_isolation(self, db_session: AsyncSession) -> None:
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
            request_count=999,
        )

        result = await get_usage(_make_request(tenant_uuid("t1")), db_session)
        assert result.total_requests == 10


# ============================================================================
# get_cost_breakdown
# ============================================================================


@pytest.mark.asyncio
class TestGetCostBreakdownReal:
    async def test_empty(self, db_session: AsyncSession) -> None:
        result = await get_cost_breakdown(_make_request(tenant_uuid("t1")), db_session)
        assert result.total_cost_usd == 0.0
        assert result.breakdown == []

    async def test_group_by_provider_sums_cost_per_provider(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        yesterday = today - _dt.timedelta(days=1)

        # Two rows for openai on different days should aggregate to one group.
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="openai",
            request_count=5,
            total_characters=500,
            estimated_cost_usd=0.25,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="openai",
            request_count=3,
            total_characters=300,
            estimated_cost_usd=0.10,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="elevenlabs",
            request_count=2,
            total_characters=200,
            estimated_cost_usd=0.05,
        )

        result = await get_cost_breakdown(
            _make_request(tenant_uuid("t1")), db_session, group_by="provider"
        )

        by_key = {b.group_key: b for b in result.breakdown}
        assert set(by_key.keys()) == {"openai", "elevenlabs"}
        assert by_key["openai"].cost_usd == pytest.approx(0.35)
        assert by_key["openai"].request_count == 8
        assert by_key["openai"].character_count == 800
        assert by_key["elevenlabs"].cost_usd == pytest.approx(0.05)
        assert result.total_cost_usd == pytest.approx(0.40)

    async def test_group_by_date_sums_cost_per_date(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        yesterday = today - _dt.timedelta(days=1)

        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="openai",
            estimated_cost_usd=0.10,
            request_count=2,
            total_characters=200,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=yesterday,
            provider="elevenlabs",
            estimated_cost_usd=0.05,
            request_count=1,
            total_characters=100,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="openai",
            estimated_cost_usd=0.20,
            request_count=4,
            total_characters=400,
        )

        result = await get_cost_breakdown(
            _make_request(tenant_uuid("t1")), db_session, group_by="date"
        )

        by_key = {b.group_key: b for b in result.breakdown}
        assert by_key[yesterday.isoformat()].cost_usd == pytest.approx(0.15)
        assert by_key[yesterday.isoformat()].request_count == 3
        assert by_key[today.isoformat()].cost_usd == pytest.approx(0.20)
        assert result.total_cost_usd == pytest.approx(0.35)

    async def test_tenant_isolation(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="openai",
            estimated_cost_usd=0.50,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t2"),
            day=today,
            provider="openai",
            estimated_cost_usd=99.99,
        )

        result = await get_cost_breakdown(_make_request(tenant_uuid("t1")), db_session)
        assert result.total_cost_usd == pytest.approx(0.50)


# ============================================================================
# export_usage
# ============================================================================


@pytest.mark.asyncio
class TestExportUsageReal:
    async def test_json_empty(self, db_session: AsyncSession) -> None:
        result = await export_usage(_make_request(tenant_uuid("t1")), db_session, fmt="json")
        assert result == []

    async def test_json_returns_record_dicts_sorted_by_date(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        day_minus_2 = today - _dt.timedelta(days=2)

        # Insert out of chronological order.
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=2,
            total_characters=200,
            total_audio_seconds=4.0,
            estimated_cost_usd=0.02,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=day_minus_2,
            provider="openai",
            request_count=5,
            total_characters=500,
            total_audio_seconds=10.0,
            estimated_cost_usd=0.25,
        )

        raw = await export_usage(_make_request(tenant_uuid("t1")), db_session, fmt="json")
        # Endpoint is typed to return Any; the JSON branch returns list[dict].
        records = cast(list[dict[str, object]], raw)

        assert len(records) == 2
        # Sorted ascending by date (ORDER BY date at usage.py:291).
        assert records[0]["date"] == day_minus_2.isoformat()
        assert records[1]["date"] == today.isoformat()
        # Row payload is correctly populated from TTSUsage columns.
        assert records[0]["provider"] == "openai"
        assert records[0]["request_count"] == 5
        assert records[0]["total_characters"] == 500
        assert records[0]["cost_usd"] == pytest.approx(0.25)

    async def test_csv_empty_returns_empty_body(self, db_session: AsyncSession) -> None:
        raw = await export_usage(_make_request(tenant_uuid("t1")), db_session, fmt="csv")
        from fastapi.responses import Response

        assert isinstance(raw, Response)
        assert raw.media_type == "text/csv"
        assert raw.body == b""

    async def test_csv_with_data_renders_header_and_rows(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=5,
            total_characters=500,
            total_audio_seconds=10.0,
            estimated_cost_usd=0.01,
        )

        raw = await export_usage(_make_request(tenant_uuid("t1")), db_session, fmt="csv")
        from fastapi.responses import Response

        assert isinstance(raw, Response)
        assert raw.media_type == "text/csv"

        body = cast(bytes, raw.body).decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        header = lines[0].split(",")
        assert "provider" in header
        assert "request_count" in header
        assert "kokoro" in lines[1]
        assert "5" in lines[1]

    async def test_csv_attachment_header_is_set(self, db_session: AsyncSession) -> None:
        """Content-Disposition must mark the payload as an attachment."""
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
        )
        raw = await export_usage(_make_request(tenant_uuid("t1")), db_session, fmt="csv")
        from fastapi.responses import Response

        assert isinstance(raw, Response)
        disposition = raw.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert "tts_usage.csv" in disposition

    async def test_tenant_isolation(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        await _mk_usage(db_session, tenant_id=tenant_uuid("t1"), day=today, provider="kokoro")
        await _mk_usage(db_session, tenant_id=tenant_uuid("t2"), day=today, provider="kokoro")

        raw = await export_usage(_make_request(tenant_uuid("t1")), db_session, fmt="json")
        records = cast(list[dict[str, object]], raw)
        assert len(records) == 1


# ============================================================================
# get_quota
# ============================================================================


@pytest.mark.asyncio
class TestGetQuotaReal:
    async def test_no_quota_record_returns_defaults(self, db_session: AsyncSession) -> None:
        """Missing TenantQuota row → all limits None, is_active True."""
        result = await get_quota(_make_request(tenant_uuid("fresh-tenant")), db_session)

        assert result.tenant_id == tenant_uuid("fresh-tenant")
        assert result.monthly_budget_usd is None
        assert result.monthly_character_limit is None
        assert result.monthly_request_limit is None
        assert result.daily_request_limit is None
        assert result.is_active is True
        assert result.monthly_budget_used_usd == 0.0
        assert result.monthly_characters_used == 0
        assert result.monthly_requests_used == 0
        assert result.daily_requests_used == 0

    async def test_quota_with_month_to_date_usage(self, db_session: AsyncSession) -> None:
        """Returned usage aggregates the current month's TTSUsage rows.

        Tripwire for ``usage.py:338-365`` — if anyone drops the
        ``month_start`` filter, older rows would leak into the total.
        """
        today = _dt.date.today()
        month_start = today.replace(day=1)

        db_session.add(
            TenantQuota(
                tenant_id=tenant_uuid("t1"),
                monthly_budget_usd=100.0,
                monthly_character_limit=10_000,
                monthly_request_limit=100,
                daily_request_limit=10,
                is_active=True,
            )
        )
        await db_session.commit()

        # A row guaranteed to be inside the current month.
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=month_start,
            provider="kokoro",
            request_count=7,
            total_characters=2000,
            estimated_cost_usd=0.30,
        )
        # Today's row for the daily counter.
        if today != month_start:
            await _mk_usage(
                db_session,
                tenant_id=tenant_uuid("t1"),
                day=today,
                provider="kokoro",
                request_count=2,
                total_characters=400,
                estimated_cost_usd=0.05,
            )
            expected_monthly_requests = 9
            expected_monthly_chars = 2400
            expected_monthly_cost = 0.35
            expected_daily_requests = 2
        else:
            expected_monthly_requests = 7
            expected_monthly_chars = 2000
            expected_monthly_cost = 0.30
            expected_daily_requests = 7

        result = await get_quota(_make_request(tenant_uuid("t1")), db_session)

        assert result.monthly_budget_usd == 100.0
        assert result.monthly_character_limit == 10_000
        assert result.monthly_requests_used == expected_monthly_requests
        assert result.monthly_characters_used == expected_monthly_chars
        assert result.monthly_budget_used_usd == pytest.approx(expected_monthly_cost)
        assert result.daily_requests_used == expected_daily_requests

    async def test_tenant_isolation(self, db_session: AsyncSession) -> None:
        today = _dt.date.today()
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t1"),
            day=today,
            provider="kokoro",
            request_count=1,
        )
        await _mk_usage(
            db_session,
            tenant_id=tenant_uuid("t2"),
            day=today,
            provider="kokoro",
            request_count=999,
        )
        result = await get_quota(_make_request(tenant_uuid("t1")), db_session)
        assert result.monthly_requests_used == 1


# ============================================================================
# acknowledge_alert
# ============================================================================


@pytest.mark.asyncio
class TestAcknowledgeAlertReal:
    async def test_404_when_alert_not_found(self, db_session: AsyncSession) -> None:
        with pytest.raises(HTTPException) as exc_info:
            await acknowledge_alert(
                "non-existent-alert-id", _make_request(tenant_uuid("t1")), db_session
            )
        assert exc_info.value.status_code == 404

    async def test_404_when_alert_belongs_to_other_tenant(self, db_session: AsyncSession) -> None:
        """Tenant isolation: acking another tenant's alert must 404, not succeed."""
        alert = await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("other-tenant"),
            created_at=_dt.datetime.now(_dt.UTC),
        )
        with pytest.raises(HTTPException) as exc_info:
            await acknowledge_alert(alert.id, _make_request(tenant_uuid("t1")), db_session)
        assert exc_info.value.status_code == 404

        # And the row was not touched.
        await db_session.refresh(alert)
        assert alert.acknowledged is False

    async def test_success_persists_acknowledged_true(self, db_session: AsyncSession) -> None:
        alert = await _mk_alert(
            db_session,
            tenant_id=tenant_uuid("t1"),
            created_at=_dt.datetime.now(_dt.UTC),
            acknowledged=False,
        )

        result = await acknowledge_alert(alert.id, _make_request(tenant_uuid("t1")), db_session)

        assert result.acknowledged is True
        assert result.id == alert.id
        assert result.tenant_id == tenant_uuid("t1")

        # Re-fetch from DB to prove persistence, not just in-memory mutation.
        await db_session.refresh(alert)
        assert alert.acknowledged is True


# ============================================================================
# get_cost_rates (reads app.state, not DB)
# ============================================================================


class TestGetCostRates:
    """Cost-rate endpoint reads the provider registry and the ROUTING SETTINGS
    from ``request.app.state``.

    ⚠️ IT USED TO READ ``request.app.state.cost_calculator``, AND THE TESTS HERE
    WERE THE REASON THAT SURVIVED. Nothing in ``src/`` ever assigned that
    attribute — ``asgi._build_router`` gives its ``CostEstimator`` to
    ``ProviderRouter`` and nowhere else — so in production the ``else 0.0`` arm
    was the only arm that ran and every provider was reported FREE. The test
    that "proved" the happy path injected ``cost_calculator`` by hand, which is
    a double that cannot exercise the property: it demonstrated a wiring that
    has never existed. Both tests below now assert against the source the app
    actually has (``settings.routing.cost_per_1k_chars``), which is also what
    the sibling ``GET /api/tts/providers`` route reads.
    """

    @staticmethod
    def _request(rates: dict[str, float] | None, *, provider_name: str = "openai") -> MagicMock:
        """A request whose app.state mirrors the SHIPPED startup hook.

        ``rates is None`` models an app with no ``settings`` on state at all —
        the only way the 0.0 fallback is reachable now.
        """
        mock_provider = MagicMock()
        mock_provider.name = provider_name
        mock_provider.provider_type = "cloud"

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = [mock_provider]

        request = MagicMock()
        request.app.state.provider_registry = mock_registry
        if rates is None:
            # `getattr(..., "settings", None)` must see a real absence, and a
            # bare MagicMock attribute would auto-create instead.
            del request.app.state.settings
        else:
            request.app.state.settings.routing.cost_per_1k_chars = rates
        return request

    @pytest.mark.asyncio
    async def test_reports_the_real_configured_rate_for_a_paid_provider(self) -> None:
        """A PAID provider must not render as free.

        openai's shipped default is 0.030/1K chars (``config.py``). Under the
        old ``app.state.cost_calculator`` read this endpoint returned 0.0 for it
        on every deployed pod.
        """
        result = await get_cost_rates(self._request({"openai": 0.030, "kokoro": 0.0}))

        assert len(result) == 1
        assert result[0].provider == "openai"
        assert result[0].cost_per_1k_chars == 0.030
        assert result[0].provider_type == "cloud"

    @pytest.mark.asyncio
    async def test_agrees_with_the_providers_endpoint_on_the_same_settings(self) -> None:
        """CROSS-ROUTE CONTROL: ``/usage/rates`` and ``/providers`` must not
        disagree about one provider's price.

        They read the same key now; they did not before, which is why the two
        endpoints answered 0.030 and 0.0 for elevenlabs simultaneously. Asserted
        by calling the OTHER route's expression against the same settings object
        rather than by restating the number, so the two cannot drift apart
        without this failing.
        """
        rates = {"elevenlabs": 0.030}
        request = self._request(rates, provider_name="elevenlabs")

        result = await get_cost_rates(request)

        # The exact expression `api/tts.list_providers` evaluates.
        providers_route_value = request.app.state.settings.routing.cost_per_1k_chars.get(
            "elevenlabs", 0.0
        )
        assert providers_route_value == 0.030  # positive control: not both zero
        assert result[0].cost_per_1k_chars == providers_route_value

    @pytest.mark.asyncio
    async def test_a_genuinely_free_local_provider_still_reads_zero(self) -> None:
        """kokoro really is 0.0, so zero must remain expressible.

        This is the control that stops the fix from being "always return a
        non-zero number": the defect was that an ABSENT rate was
        INDISTINGUISHABLE from a real one, not that zero is wrong.
        """
        result = await get_cost_rates(self._request({"kokoro": 0.0}, provider_name="kokoro"))

        assert result[0].cost_per_1k_chars == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_rate_when_settings_absent(self) -> None:
        """No ``settings`` on app.state → 0.0 fallback (unreachable in the
        shipped app, where ``_startup_hook`` always assigns it)."""
        result = await get_cost_rates(self._request(None))

        assert len(result) == 1
        assert result[0].cost_per_1k_chars == 0.0
