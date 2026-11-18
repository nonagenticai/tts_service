"""Usage analytics and cost tracking API endpoints."""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import LICENSE_GATE

from src.api.schemas import (
    CostBreakdownItem,
    CostBreakdownResponse,
    DailyUsagePoint,
    DashboardDataResponse,
    ProviderCostRateResponse,
    ProviderUsageSummary,
    QuotaStatusResponse,
    QuotaUpdateRequest,
    UsageAlertResponse,
    UsageSummaryResponse,
)
from src.database.models import TenantQuota, TTSUsage, UsageAlert
from src.database.session import get_db
from src.services.metering import InvalidTenantIdError, MissingTenantError, require_tenant

logger = logging.getLogger(__name__)

# ENTITLEMENT GATE — see the long note in src/api/tts.py for the full contract
# (w2v_common.auth.entitlement, dormant until W2V_LICENSE_GATE_ENABLED is set).
# This router is entirely tenant-scoped usage/cost/quota analytics, so gating
# it router-wide is correct; health/ws stay ungated for the reasons in tts.py.
router = APIRouter(prefix="/api/tts/usage", tags=["usage"], dependencies=list(LICENSE_GATE))


def _get_tenant_id(request: Request) -> str:
    """Return the caller's REAL forwarded tenant, or 401 — never a sentinel.

    PRODUCT-OWNER RULING: a ``tenant_id`` may never hold a sentinel. All traffic
    carries a real registered tenant forwarded from the requestor; an absent one
    is an ERROR, not a substituted default. This helper used to return the
    nil-UUID ``DEFAULT_TENANT_ID`` — the most dangerous spelling, because it is a
    VALID uuid that passes every cast and dirty-row check, and billing then
    buckets it as unattributed and NEVER CHARGES it. Real usage, silently
    unbilled. Do not re-add a fallback here.

    Two outcomes, two status codes — they are NOT the same failure:

    * **401 on ABSENT.** ``request.state.tenant_id`` is populated exclusively by
      ``AuthMiddleware`` from the ``X-Tenant-Id: Bearer <license_token>``
      handshake, so its absence means authentication resolved no identity, not
      that the payload was malformed. Same choice as ``slide_service``, and
      consistent with this repo's own WS path, which already closes 1008 on the
      identical condition.
    * **400 on PRESENT-BUT-NOT-A-UUID.** Since revision ``005_tenant_id_uuid``
      the tenant is a postgres ``uuid`` COLUMN here too, so an unparseable value
      is a bad argument — exactly ``video_compiler_service``'s reasoning, which
      reached the uuid column first. Catching it here is what keeps a legacy
      slug from becoming a psycopg ``DataError`` 500 midway through a job write.

    The returned value is canonical lowercase-dashed uuid text, so it compares
    equal to what postgres renders back out of the column.
    """
    try:
        return require_tenant(getattr(request.state, "tenant_id", None))
    except MissingTenantError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except InvalidTenantIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Portal-realm role that owns quota LIMITS. Same name and same source as
# w2v_orchestrator's ``require_super_admin`` (``src/api/routes/_common.py``):
# ``w2v_common`` AuthMiddleware stashes the Keycloak ``realm_access.roles`` on
# ``request.state.roles`` (JWT path); license-token / anonymous callers get
# ``[]`` and therefore can never write a limit.
SUPER_ADMIN = "super_admin"


def _require_super_admin(request: Request) -> None:
    """403 unless the caller holds ``super_admin``.

    Quota limits are an operator decision, not a tenant one: a tenant that can
    write its own ``tenant_quotas`` row can lift its own monthly budget and
    request caps. Tenants may READ their own quota (``GET /quota``) but never
    write it. Call AFTER ``_get_tenant_id`` so an unauthenticated caller still
    gets 401, not 403.
    """
    roles = set(getattr(request.state, "roles", None) or [])
    if SUPER_ADMIN not in roles:
        raise HTTPException(status_code=403, detail="super_admin role required")


# ============================================================================
# Usage Summary
# ============================================================================


@router.get("", operation_id="get_tts_usage_summary")
async def get_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
    date_from: str | None = None,
    date_to: str | None = None,
    provider: str | None = None,
) -> UsageSummaryResponse:
    """Get usage summary for the current tenant."""
    tenant_id = _get_tenant_id(request)
    logger.debug(
        "Usage summary request: tenant=%s date_from=%s date_to=%s", tenant_id, date_from, date_to
    )

    d_from = date.fromisoformat(date_from) if date_from else date.today() - timedelta(days=30)
    d_to = date.fromisoformat(date_to) if date_to else date.today()

    query = select(TTSUsage).where(
        TTSUsage.tenant_id == tenant_id,
        TTSUsage.date >= d_from,
        TTSUsage.date <= d_to,
    )
    if provider:
        query = query.where(TTSUsage.provider == provider)

    result = await db.execute(query)
    rows = result.scalars().all()

    total_requests = sum(r.request_count for r in rows)
    total_chars = sum(r.total_characters for r in rows)
    total_audio = sum(r.total_audio_seconds for r in rows)
    total_cost = sum(r.estimated_cost_usd or 0.0 for r in rows)
    total_errors = sum(r.error_count for r in rows)
    total_cache_hits = sum(getattr(r, "cache_hits", 0) or 0 for r in rows)
    total_cache_misses = sum(getattr(r, "cache_misses", 0) or 0 for r in rows)

    cache_total = total_cache_hits + total_cache_misses
    cache_hit_rate = total_cache_hits / cache_total if cache_total > 0 else 0.0

    # Group by provider
    provider_map: dict[str, list] = {}
    for r in rows:
        provider_map.setdefault(str(r.provider), []).append(r)

    by_provider = []
    for prov, prov_rows in provider_map.items():
        prov_requests = sum(r.request_count for r in prov_rows)
        prov_latency = sum((r.avg_latency_ms or 0.0) * r.request_count for r in prov_rows)
        by_provider.append(
            ProviderUsageSummary(
                provider=prov,
                request_count=prov_requests,
                total_characters=sum(r.total_characters for r in prov_rows),
                total_audio_seconds=sum(r.total_audio_seconds for r in prov_rows),
                cost_usd=sum(r.estimated_cost_usd or 0.0 for r in prov_rows),
                error_count=sum(r.error_count for r in prov_rows),
                avg_latency_ms=prov_latency / prov_requests if prov_requests > 0 else 0.0,
            )
        )

    return UsageSummaryResponse(
        tenant_id=tenant_id,
        period_start=d_from.isoformat(),
        period_end=d_to.isoformat(),
        total_requests=total_requests,
        total_characters=total_chars,
        total_audio_seconds=total_audio,
        total_cost_usd=total_cost,
        error_count=total_errors,
        cache_hit_rate=cache_hit_rate,
        by_provider=by_provider,
    )


# ============================================================================
# Cost Breakdown
# ============================================================================


@router.get("/costs", operation_id="get_tts_cost_breakdown")
async def get_cost_breakdown(
    request: Request,
    db: AsyncSession = Depends(get_db),
    date_from: str | None = None,
    date_to: str | None = None,
    group_by: str = "provider",
) -> CostBreakdownResponse:
    """Get cost breakdown grouped by provider or date."""
    tenant_id = _get_tenant_id(request)
    logger.debug("Cost breakdown request: tenant=%s group_by=%s", tenant_id, group_by)
    d_from = date.fromisoformat(date_from) if date_from else date.today() - timedelta(days=30)
    d_to = date.fromisoformat(date_to) if date_to else date.today()

    query = select(TTSUsage).where(
        TTSUsage.tenant_id == tenant_id,
        TTSUsage.date >= d_from,
        TTSUsage.date <= d_to,
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    breakdown_map: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = r.date.isoformat() if group_by == "date" else str(r.provider)
        if key not in breakdown_map:
            breakdown_map[key] = {"cost_usd": 0.0, "request_count": 0, "character_count": 0}
        breakdown_map[key]["cost_usd"] += r.estimated_cost_usd or 0.0
        breakdown_map[key]["request_count"] += r.request_count
        breakdown_map[key]["character_count"] += r.total_characters

    breakdown = [
        CostBreakdownItem(
            group_key=k,
            cost_usd=v["cost_usd"],
            request_count=v["request_count"],
            character_count=v["character_count"],
        )
        for k, v in breakdown_map.items()
    ]
    total_cost = sum(b.cost_usd for b in breakdown)

    return CostBreakdownResponse(
        period_start=d_from.isoformat(),
        period_end=d_to.isoformat(),
        total_cost_usd=total_cost,
        breakdown=breakdown,
    )


# ============================================================================
# Dashboard
# ============================================================================


@router.get("/dashboard", operation_id="get_tts_dashboard")
async def get_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
    days: int = 30,
) -> DashboardDataResponse:
    """Get dashboard time-series data."""
    tenant_id = _get_tenant_id(request)
    d_from = date.today() - timedelta(days=days)
    d_to = date.today()

    query = (
        select(TTSUsage)
        .where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date >= d_from,
            TTSUsage.date <= d_to,
        )
        .order_by(TTSUsage.date)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    # Time series by date
    daily_map: dict[str, dict] = {}
    provider_map: dict[str, dict] = {}

    for r in rows:
        d = r.date.isoformat()
        if d not in daily_map:
            daily_map[d] = {
                "request_count": 0,
                "total_characters": 0,
                "total_audio_seconds": 0.0,
                "cost_usd": 0.0,
            }
        daily_map[d]["request_count"] += r.request_count
        daily_map[d]["total_characters"] += r.total_characters
        daily_map[d]["total_audio_seconds"] += r.total_audio_seconds
        daily_map[d]["cost_usd"] += r.estimated_cost_usd or 0.0

        prov = str(r.provider)
        if prov not in provider_map:
            provider_map[prov] = {
                "request_count": 0,
                "total_characters": 0,
                "total_audio_seconds": 0.0,
                "cost_usd": 0.0,
                "error_count": 0,
                "total_latency_ms": 0.0,
            }
        pm = provider_map[prov]
        pm["request_count"] += r.request_count
        pm["total_characters"] += r.total_characters
        pm["total_audio_seconds"] += r.total_audio_seconds
        pm["cost_usd"] += r.estimated_cost_usd or 0.0
        pm["error_count"] += r.error_count
        pm["total_latency_ms"] += (r.avg_latency_ms or 0.0) * r.request_count

    usage_over_time = [DailyUsagePoint(date=d, **v) for d, v in sorted(daily_map.items())]
    provider_dist = [
        ProviderUsageSummary(
            provider=p,
            request_count=v["request_count"],
            total_characters=v["total_characters"],
            total_audio_seconds=v["total_audio_seconds"],
            cost_usd=v["cost_usd"],
            error_count=v["error_count"],
            avg_latency_ms=(
                v["total_latency_ms"] / v["request_count"] if v["request_count"] > 0 else 0.0
            ),
        )
        for p, v in provider_map.items()
    ]

    return DashboardDataResponse(
        tenant_id=tenant_id,
        days=days,
        usage_over_time=usage_over_time,
        provider_distribution=provider_dist,
    )


# ============================================================================
# Export
# ============================================================================


@router.get("/export", operation_id="export_tts_usage")
async def export_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
    date_from: str | None = None,
    date_to: str | None = None,
    fmt: str = Query(default="json", alias="format"),
) -> Any:
    """Export usage data as JSON or CSV."""
    tenant_id = _get_tenant_id(request)
    d_from = date.fromisoformat(date_from) if date_from else date.today() - timedelta(days=30)
    d_to = date.fromisoformat(date_to) if date_to else date.today()

    query = (
        select(TTSUsage)
        .where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date >= d_from,
            TTSUsage.date <= d_to,
        )
        .order_by(TTSUsage.date)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    records = [
        {
            "date": r.date.isoformat(),
            "provider": r.provider,
            "request_count": r.request_count,
            "total_characters": r.total_characters,
            "total_audio_seconds": r.total_audio_seconds,
            "error_count": r.error_count,
            "cost_usd": r.estimated_cost_usd or 0.0,
        }
        for r in rows
    ]

    if fmt == "csv":
        output = io.StringIO()
        if records:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=tts_usage.csv"},
        )

    return records


# ============================================================================
# Quota
# ============================================================================


@router.get("/quota", operation_id="get_tts_quota")
async def get_quota(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> QuotaStatusResponse:
    """Get current tenant quota and usage against it."""
    tenant_id = _get_tenant_id(request)

    # Get quota record
    result = await db.execute(select(TenantQuota).where(TenantQuota.tenant_id == tenant_id))
    quota = result.scalar_one_or_none()

    # Get current month usage
    today = date.today()
    month_start = today.replace(day=1)
    usage_result = await db.execute(
        select(
            func.coalesce(func.sum(TTSUsage.request_count), 0),
            func.coalesce(func.sum(TTSUsage.total_characters), 0),
            func.coalesce(func.sum(TTSUsage.estimated_cost_usd), 0),
        ).where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date >= month_start,
        )
    )
    row = usage_result.one()

    # Get daily usage
    daily_result = await db.execute(
        select(
            func.coalesce(func.sum(TTSUsage.request_count), 0),
        ).where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date == today,
        )
    )
    daily_row = daily_result.one()

    return QuotaStatusResponse(
        tenant_id=tenant_id,
        monthly_budget_usd=quota.monthly_budget_usd if quota else None,
        monthly_budget_used_usd=float(row[2]) if row[2] else 0.0,
        monthly_character_limit=quota.monthly_character_limit if quota else None,
        monthly_characters_used=int(row[1]),
        monthly_request_limit=quota.monthly_request_limit if quota else None,
        monthly_requests_used=int(row[0]),
        daily_request_limit=quota.daily_request_limit if quota else None,
        daily_requests_used=int(daily_row[0]),
        is_active=quota.is_active if quota else True,
    )


@router.put(
    "/quota",
    operation_id="update_tts_quota",
    responses={
        401: {"description": "No authenticated tenant"},
        403: {"description": "Caller lacks the super_admin role"},
        422: {"description": "Request body validation error"},
    },
)
async def update_quota(
    body: QuotaUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> QuotaStatusResponse:
    """Update tenant quota settings — ``super_admin`` only.

    Limits are owned by the operator; a tenant caller may only read its own
    quota via ``GET /quota``.
    """
    tenant_id = _get_tenant_id(request)
    _require_super_admin(request)
    logger.info("Quota update request: tenant=%s", tenant_id)

    result = await db.execute(select(TenantQuota).where(TenantQuota.tenant_id == tenant_id))
    quota = result.scalar_one_or_none()

    if quota is None:
        quota = TenantQuota(tenant_id=tenant_id)
        db.add(quota)

    if body.monthly_budget_usd is not None:
        quota.monthly_budget_usd = body.monthly_budget_usd
    if body.monthly_character_limit is not None:
        quota.monthly_character_limit = body.monthly_character_limit
    if body.monthly_request_limit is not None:
        quota.monthly_request_limit = body.monthly_request_limit
    if body.daily_request_limit is not None:
        quota.daily_request_limit = body.daily_request_limit

    await db.flush()
    await db.commit()

    return await get_quota(request, db)


# ============================================================================
# Alerts
# ============================================================================


@router.get("/alerts", operation_id="list_tts_usage_alerts")
async def get_alerts(
    request: Request,
    db: AsyncSession = Depends(get_db),
    acknowledged: bool | None = None,
    limit: int = 50,
) -> list[UsageAlertResponse]:
    """Get usage alerts for the current tenant."""
    tenant_id = _get_tenant_id(request)

    query = (
        select(UsageAlert)
        .where(UsageAlert.tenant_id == tenant_id)
        .order_by(UsageAlert.created_at.desc())
        .limit(limit)
    )

    if acknowledged is not None:
        query = query.where(UsageAlert.acknowledged == acknowledged)

    result = await db.execute(query)
    alerts = result.scalars().all()

    return [
        UsageAlertResponse(
            id=a.id,
            tenant_id=a.tenant_id,
            alert_type=a.alert_type,
            threshold_pct=a.threshold_pct,
            current_value=a.current_value,
            limit_value=a.limit_value,
            period=a.period,
            message=a.message,
            acknowledged=a.acknowledged,
            created_at=a.created_at.isoformat() if a.created_at else "",
        )
        for a in alerts
    ]


@router.post(
    "/alerts/{alert_id}/acknowledge",
    operation_id="acknowledge_tts_usage_alert",
    responses={
        404: {"description": "Alert not found"},
    },
)
async def acknowledge_alert(
    alert_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> UsageAlertResponse:
    """Acknowledge a usage alert."""
    tenant_id = _get_tenant_id(request)
    logger.info("Alert acknowledge request: id=%s tenant=%s", alert_id, tenant_id)

    result = await db.execute(
        select(UsageAlert).where(
            UsageAlert.id == alert_id,
            UsageAlert.tenant_id == tenant_id,
        )
    )
    alert = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.acknowledged = True
    await db.flush()
    await db.commit()

    return UsageAlertResponse(
        id=alert.id,
        tenant_id=alert.tenant_id,
        alert_type=alert.alert_type,
        threshold_pct=alert.threshold_pct,
        current_value=alert.current_value,
        limit_value=alert.limit_value,
        period=alert.period,
        message=alert.message,
        acknowledged=alert.acknowledged,
        created_at=alert.created_at.isoformat() if alert.created_at else "",
    )


# ============================================================================
# Provider Cost Rates
# ============================================================================


@router.get("/rates", operation_id="get_tts_cost_rates")
async def get_cost_rates(
    request: Request,
) -> list[ProviderCostRateResponse]:
    """Get provider cost rates.

    ⛔ READS ``settings.routing.cost_per_1k_chars``, NOT ``app.state``.

    This used to read ``getattr(request.app.state, "cost_calculator", None)``
    and fall back to ``0.0``. NOTHING IN ``src/`` HAS EVER ASSIGNED
    ``app.state.cost_calculator``: ``asgi._build_router`` constructs a
    ``CostEstimator`` and hands it only to ``ProviderRouter``, and the startup
    hook assigns ``provider_registry``, ``storage_service``, ``metering_redis``,
    ``settings``, ``tts_service``, ``notification_service`` and
    ``connection_manager`` — never this one. So the ``else 0.0`` arm was the
    ONLY arm that ever ran in production, and every provider was reported FREE:
    openai at 0.030 and elevenlabs at 0.030 both rendered as $0.00.

    That is the fail-open shape, not merely a wrong number. An ABSENT
    calculator scored identically to a genuinely-$0 local provider (kokoro and
    kitten really are 0.0), so the zero was unreadable — there was no value the
    endpoint could return that meant "unknown".

    ⚠️ It also DISAGREED with its own sibling. ``GET /api/tts/providers``
    (``api/tts.py``) already reads ``settings.routing.cost_per_1k_chars``
    directly and returns the correct rate, so the two endpoints have been
    answering different numbers for the same provider. This converges on the
    sibling's source rather than wiring ``app.state.cost_calculator`` up,
    because ``_build_router`` returns ``None`` when ``settings.routing.enabled``
    is false — a state-based fix would silently re-open the same $0.00 hole in
    that configuration, while settings are always present.
    """
    settings = getattr(request.app.state, "settings", None)
    registry = request.app.state.provider_registry

    rates: list[ProviderCostRateResponse] = []
    for p in registry.list_providers():
        rate = settings.routing.cost_per_1k_chars.get(p.name, 0.0) if settings else 0.0
        rates.append(
            ProviderCostRateResponse(
                provider=p.name,
                cost_per_1k_chars=rate,
                provider_type=p.provider_type,
            )
        )
    return rates
