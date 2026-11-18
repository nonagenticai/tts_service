"""Quota enforcement and usage alert generation for TTS service."""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database.models import TenantQuota, TTSUsage, UsageAlert

logger = logging.getLogger(__name__)

# Alert thresholds as percentages
ALERT_THRESHOLDS = [80, 90, 100]


class QuotaExceededError(Exception):
    """Raised when a tenant has exceeded their quota."""

    def __init__(
        self, message: str, limit_type: str, current_value: float, limit_value: float
    ) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


def _log_quota_thresholds(
    tenant_id: str,
    monthly_cost: float,
    monthly_characters: int,
    monthly_requests: int,
    quota: object,
) -> None:
    """Log structured warnings when quota approaches thresholds (80%, 90%)."""
    checks = [
        ("monthly_budget", monthly_cost, getattr(quota, "monthly_budget_usd", None)),
        (
            "monthly_characters",
            float(monthly_characters),
            getattr(quota, "monthly_character_limit", None),
        ),
        (
            "monthly_requests",
            float(monthly_requests),
            getattr(quota, "monthly_request_limit", None),
        ),
    ]
    for limit_type, current, limit in checks:
        if limit is None or float(limit) <= 0:
            continue
        pct = (current / float(limit)) * 100
        if pct >= 90:
            logger.warning(
                "quota_approaching: tenant_id=%s limit_type=%s pct=%.0f current=%.2f limit=%.2f",
                tenant_id,
                limit_type,
                pct,
                current,
                float(limit),
            )
        elif pct >= 80:
            logger.info(
                "quota_approaching: tenant_id=%s limit_type=%s pct=%.0f current=%.2f limit=%.2f",
                tenant_id,
                limit_type,
                pct,
                current,
                float(limit),
            )


def _check_monthly_budget(
    tenant_id: str, monthly_cost: float, projected_cost: float, quota: TenantQuota
) -> None:
    """Raise QuotaExceededError if monthly cost budget would be exceeded.

    ``projected_cost`` is the estimated cost of the *incoming* request. It is
    added to the already-spent ``monthly_cost`` before the comparison, mirroring
    ``_check_monthly_character_limit`` — without it a tenant at $9.99 of a $10
    budget could submit one arbitrarily large request and overshoot the cap.
    """
    if quota.monthly_budget_usd is None:
        return
    projected_total = monthly_cost + projected_cost
    if projected_total < quota.monthly_budget_usd:
        return
    logger.warning(
        "quota_exceeded: tenant_id=%s limit_type=monthly_budget current=%.2f limit=%.2f",
        tenant_id,
        projected_total,
        quota.monthly_budget_usd,
    )
    raise QuotaExceededError(
        message=(
            f"Monthly cost budget exceeded: "
            f"${projected_total:.2f} >= ${quota.monthly_budget_usd:.2f}"
        ),
        limit_type="monthly_budget_usd",
        current_value=projected_total,
        limit_value=quota.monthly_budget_usd,
    )


def _check_monthly_character_limit(
    monthly_characters: int, character_count: int, quota: TenantQuota
) -> None:
    """Raise QuotaExceededError if the projected monthly character total exceeds the limit."""
    if quota.monthly_character_limit is None:
        return
    projected = monthly_characters + character_count
    if projected <= quota.monthly_character_limit:
        return
    raise QuotaExceededError(
        message=(
            f"Monthly character limit exceeded: {projected} > {quota.monthly_character_limit}"
        ),
        limit_type="monthly_character_limit",
        current_value=float(monthly_characters),
        limit_value=float(quota.monthly_character_limit),
    )


def _check_monthly_request_limit(monthly_requests: int, quota: TenantQuota) -> None:
    """Raise QuotaExceededError if one more request would exceed the monthly request limit."""
    if quota.monthly_request_limit is None:
        return
    if monthly_requests + 1 <= quota.monthly_request_limit:
        return
    raise QuotaExceededError(
        message=(
            f"Monthly request limit exceeded: "
            f"{monthly_requests + 1} > {quota.monthly_request_limit}"
        ),
        limit_type="monthly_request_limit",
        current_value=float(monthly_requests),
        limit_value=float(quota.monthly_request_limit),
    )


async def _check_daily_request_limit(
    db: AsyncSession, tenant_id: str, today: date, quota: TenantQuota
) -> None:
    """Raise QuotaExceededError if one more request would exceed the daily request limit."""
    if quota.daily_request_limit is None:
        return
    daily_result = await db.execute(
        select(
            func.coalesce(func.sum(TTSUsage.request_count), 0),
        ).where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date == today,
        )
    )
    daily_requests = int(daily_result.one()[0])
    if daily_requests + 1 <= quota.daily_request_limit:
        return
    raise QuotaExceededError(
        message=(
            f"Daily request limit exceeded: {daily_requests + 1} > {quota.daily_request_limit}"
        ),
        limit_type="daily_request_limit",
        current_value=float(daily_requests),
        limit_value=float(quota.daily_request_limit),
    )


async def check_quota(
    db: AsyncSession,
    tenant_id: str,
    character_count: int,
    projected_cost: float = 0.0,
) -> None:
    """Check if the tenant has remaining quota for this request.

    Raises QuotaExceededError if any quota limit would be exceeded.
    Does nothing if quota enforcement is disabled or no quota is configured.
    """
    settings = get_settings()
    if not settings.quota.enforcement_enabled:
        return

    # Look up tenant quota record
    result = await db.execute(select(TenantQuota).where(TenantQuota.tenant_id == tenant_id))
    quota = result.scalar_one_or_none()

    if quota is None:
        return

    if not quota.is_active:
        logger.warning("Tenant %s account is deactivated", tenant_id)
        raise QuotaExceededError(
            message="Tenant account is deactivated",
            limit_type="account_active",
            current_value=0,
            limit_value=0,
        )

    today = date.today()
    month_start = today.replace(day=1)

    # Fetch monthly aggregates in one query
    monthly_result = await db.execute(
        select(
            func.coalesce(func.sum(TTSUsage.request_count), 0),
            func.coalesce(func.sum(TTSUsage.total_characters), 0),
            func.coalesce(func.sum(TTSUsage.estimated_cost_usd), 0.0),
        ).where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date >= month_start,
        )
    )
    monthly_row = monthly_result.one()
    monthly_requests = int(monthly_row[0])
    monthly_characters = int(monthly_row[1])
    monthly_cost = float(monthly_row[2])

    _log_quota_thresholds(tenant_id, monthly_cost, monthly_characters, monthly_requests, quota)
    _check_monthly_budget(tenant_id, monthly_cost, projected_cost, quota)
    _check_monthly_character_limit(monthly_characters, character_count, quota)
    _check_monthly_request_limit(monthly_requests, quota)
    await _check_daily_request_limit(db, tenant_id, today, quota)


async def check_and_create_alerts(
    db: AsyncSession,
    tenant_id: str,
) -> list[UsageAlert]:
    """Check usage against quota thresholds and create alert records if crossed.

    Returns a list of newly created alerts. Does nothing if quota enforcement
    is disabled or no quota is configured.
    """
    settings = get_settings()
    if not settings.quota.enforcement_enabled:
        return []

    # Look up tenant quota record
    result = await db.execute(select(TenantQuota).where(TenantQuota.tenant_id == tenant_id))
    quota = result.scalar_one_or_none()

    if quota is None:
        return []

    today = date.today()
    month_start = today.replace(day=1)

    # Fetch monthly aggregates
    monthly_result = await db.execute(
        select(
            func.coalesce(func.sum(TTSUsage.request_count), 0),
            func.coalesce(func.sum(TTSUsage.total_characters), 0),
            func.coalesce(func.sum(TTSUsage.estimated_cost_usd), 0.0),
        ).where(
            TTSUsage.tenant_id == tenant_id,
            TTSUsage.date >= month_start,
        )
    )
    monthly_row = monthly_result.one()
    monthly_requests = int(monthly_row[0])
    monthly_characters = int(monthly_row[1])
    monthly_cost = float(monthly_row[2])

    new_alerts: list[UsageAlert] = []

    # Define checks: (alert_type, current_value, limit_value, period)
    checks: list[tuple[str, float, float | None, str]] = [
        (
            "monthly_budget_usd",
            monthly_cost,
            float(quota.monthly_budget_usd) if quota.monthly_budget_usd else None,
            "monthly",
        ),
        (
            "monthly_character_limit",
            float(monthly_characters),
            float(quota.monthly_character_limit) if quota.monthly_character_limit else None,
            "monthly",
        ),
        (
            "monthly_request_limit",
            float(monthly_requests),
            float(quota.monthly_request_limit) if quota.monthly_request_limit else None,
            "monthly",
        ),
    ]

    for alert_type, current_value, limit_value, period in checks:
        if limit_value is None or limit_value <= 0:
            continue

        usage_pct = (current_value / limit_value) * 100

        for threshold in ALERT_THRESHOLDS:
            if usage_pct < threshold:
                continue

            # Check if alert already exists for this threshold in current period
            existing = await db.execute(
                select(UsageAlert).where(
                    UsageAlert.tenant_id == tenant_id,
                    UsageAlert.alert_type == alert_type,
                    UsageAlert.threshold_pct == threshold,
                    UsageAlert.period == period,
                    UsageAlert.created_at >= month_start,
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue

            # Create new alert
            message = (
                f"{alert_type} usage at {usage_pct:.0f}% ({current_value:.2f} / {limit_value:.2f})"
            )
            alert = UsageAlert(
                tenant_id=tenant_id,
                alert_type=alert_type,
                threshold_pct=threshold,
                current_value=current_value,
                limit_value=limit_value,
                period=period,
                message=message,
            )
            db.add(alert)
            new_alerts.append(alert)
            logger.info(
                "Created usage alert for tenant %s: %s at %d%% threshold",
                tenant_id,
                alert_type,
                threshold,
            )

    if new_alerts:
        await db.flush()

    return new_alerts
