"""Unit tests for quota enforcement and usage alert generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.quota_service import (
    ALERT_THRESHOLDS,
    QuotaExceededError,
    check_and_create_alerts,
    check_quota,
)
from tests.conftest import tenant_uuid

# ============================================================================
# QuotaExceededError
# ============================================================================


class TestQuotaExceededError:
    def test_attributes(self):
        err = QuotaExceededError(
            message="over budget",
            limit_type="monthly_budget_usd",
            current_value=50.0,
            limit_value=40.0,
        )
        assert str(err) == "over budget"
        assert err.limit_type == "monthly_budget_usd"
        assert err.current_value == 50.0
        assert err.limit_value == 40.0


# ============================================================================
# check_quota
# ============================================================================


class TestCheckQuota:
    @pytest.mark.asyncio
    async def test_disabled_enforcement_does_nothing(self):
        """When enforcement is disabled, check_quota should return immediately."""
        db = AsyncMock()
        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = False
            # Should not raise or query DB
            await check_quota(db, tenant_uuid("tenant-1"), 100)
            db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_quota_record_passes(self):
        """When no TenantQuota record exists, check_quota should pass."""
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=mock_result)

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            await check_quota(db, tenant_uuid("tenant-1"), 100)

    @pytest.mark.asyncio
    async def test_inactive_tenant_raises(self):
        """Deactivated tenant should be rejected."""
        db = AsyncMock()
        mock_quota = MagicMock()
        mock_quota.is_active = False

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_quota
        db.execute = AsyncMock(return_value=mock_result)

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            with pytest.raises(QuotaExceededError, match="deactivated"):
                await check_quota(db, tenant_uuid("tenant-1"), 100)

    @pytest.mark.asyncio
    async def test_monthly_budget_exceeded_raises(self):
        """Should raise when monthly cost exceeds budget."""
        db = AsyncMock()
        mock_quota = MagicMock()
        mock_quota.is_active = True
        mock_quota.monthly_budget_usd = 10.0
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None
        mock_quota.daily_request_limit = None

        # First call: select TenantQuota
        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Second call: monthly usage aggregates (requests, chars, cost)
        mock_usage_result = MagicMock()
        mock_usage_result.one.return_value = (5, 5000, 15.0)  # cost > budget

        db.execute = AsyncMock(side_effect=[mock_quota_result, mock_usage_result])

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            with pytest.raises(QuotaExceededError, match="Monthly cost budget exceeded"):
                await check_quota(db, tenant_uuid("tenant-1"), 100)

    @pytest.mark.asyncio
    async def test_monthly_character_limit_exceeded_raises(self):
        """Should raise when projected characters exceed monthly limit."""
        db = AsyncMock()
        mock_quota = MagicMock()
        mock_quota.is_active = True
        mock_quota.monthly_budget_usd = None
        mock_quota.monthly_character_limit = 10000
        mock_quota.monthly_request_limit = None
        mock_quota.daily_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: 9500 chars used, request adds 600 -> 10100 > 10000
        mock_usage_result = MagicMock()
        mock_usage_result.one.return_value = (10, 9500, 0.0)

        db.execute = AsyncMock(side_effect=[mock_quota_result, mock_usage_result])

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            with pytest.raises(QuotaExceededError, match="Monthly character limit exceeded"):
                await check_quota(db, tenant_uuid("tenant-1"), 600)

    @pytest.mark.asyncio
    async def test_monthly_request_limit_exceeded_raises(self):
        """Should raise when monthly request count exceeds limit."""
        db = AsyncMock()
        mock_quota = MagicMock()
        mock_quota.is_active = True
        mock_quota.monthly_budget_usd = None
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = 100
        mock_quota.daily_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: 100 requests used, +1 = 101 > 100
        mock_usage_result = MagicMock()
        mock_usage_result.one.return_value = (100, 50000, 0.0)

        db.execute = AsyncMock(side_effect=[mock_quota_result, mock_usage_result])

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            with pytest.raises(QuotaExceededError, match="Monthly request limit exceeded"):
                await check_quota(db, tenant_uuid("tenant-1"), 100)

    @pytest.mark.asyncio
    async def test_daily_request_limit_exceeded_raises(self):
        """Should raise when daily request count exceeds limit."""
        db = AsyncMock()
        mock_quota = MagicMock()
        mock_quota.is_active = True
        mock_quota.monthly_budget_usd = None
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None
        mock_quota.daily_request_limit = 50

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly aggregates: within limits
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (50, 10000, 0.0)

        # Daily: 50 requests -> +1 = 51 > 50
        mock_daily_result = MagicMock()
        mock_daily_result.one.return_value = (50,)

        db.execute = AsyncMock(
            side_effect=[mock_quota_result, mock_monthly_result, mock_daily_result]
        )

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            with pytest.raises(QuotaExceededError, match="Daily request limit exceeded"):
                await check_quota(db, tenant_uuid("tenant-1"), 100)

    @pytest.mark.asyncio
    async def test_within_all_limits_passes(self):
        """Should pass when all usage is within limits."""
        db = AsyncMock()
        mock_quota = MagicMock()
        mock_quota.is_active = True
        mock_quota.monthly_budget_usd = 100.0
        mock_quota.monthly_character_limit = 100000
        mock_quota.monthly_request_limit = 1000
        mock_quota.daily_request_limit = 200

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: well within limits
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (10, 5000, 1.0)

        # Daily: well within limits
        mock_daily_result = MagicMock()
        mock_daily_result.one.return_value = (5,)

        db.execute = AsyncMock(
            side_effect=[mock_quota_result, mock_monthly_result, mock_daily_result]
        )

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            # Should not raise
            await check_quota(db, tenant_uuid("tenant-1"), 100)


# ============================================================================
# check_and_create_alerts
# ============================================================================


class TestCheckAndCreateAlerts:
    @pytest.mark.asyncio
    async def test_disabled_enforcement_returns_empty(self):
        db = AsyncMock()
        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = False
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))
            assert alerts == []
            db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_quota_record_returns_empty(self):
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=mock_result)

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))
            assert alerts == []

    @pytest.mark.asyncio
    async def test_creates_alert_at_80_percent(self):
        """Should create an alert when usage crosses 80% threshold."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        mock_quota = MagicMock()
        mock_quota.monthly_budget_usd = 100.0
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: $85 used out of $100 = 85% -> crosses 80%
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (50, 50000, 85.0)

        # Alert check: no existing alert for 80%
        mock_no_alert = MagicMock()
        mock_no_alert.scalar_one_or_none.return_value = None

        db.execute = AsyncMock(side_effect=[mock_quota_result, mock_monthly_result, mock_no_alert])

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))

        assert len(alerts) == 1
        assert alerts[0].threshold_pct == 80
        assert alerts[0].alert_type == "monthly_budget_usd"
        db.add.assert_called_once()
        db.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_multiple_alerts_for_100_percent(self):
        """When usage >= 100%, all thresholds (80, 90, 100) should fire."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        mock_quota = MagicMock()
        mock_quota.monthly_budget_usd = 100.0
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: $105 used out of $100 = 105%
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (100, 100000, 105.0)

        # All existing alert checks return None (no prior alerts)
        mock_no_alert = MagicMock()
        mock_no_alert.scalar_one_or_none.return_value = None

        db.execute = AsyncMock(
            side_effect=[
                mock_quota_result,
                mock_monthly_result,
                mock_no_alert,  # 80% check
                mock_no_alert,  # 90% check
                mock_no_alert,  # 100% check
            ]
        )

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))

        assert len(alerts) == 3
        assert {a.threshold_pct for a in alerts} == {80, 90, 100}

    @pytest.mark.asyncio
    async def test_skips_already_existing_alert(self):
        """Should not duplicate an alert that already exists."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        mock_quota = MagicMock()
        mock_quota.monthly_budget_usd = 100.0
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: $85 used -> 85% -> should check 80% threshold
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (50, 50000, 85.0)

        # Already have 80% alert
        mock_existing_alert = MagicMock()
        mock_existing_alert.scalar_one_or_none.return_value = MagicMock()  # exists

        db.execute = AsyncMock(
            side_effect=[mock_quota_result, mock_monthly_result, mock_existing_alert]
        )

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))

        assert len(alerts) == 0
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_limits_configured_returns_empty(self):
        """When quota exists but no limits are set, no alerts should fire."""
        db = AsyncMock()

        mock_quota = MagicMock()
        mock_quota.monthly_budget_usd = None
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly aggregates
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (50, 50000, 5.0)

        db.execute = AsyncMock(side_effect=[mock_quota_result, mock_monthly_result])

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))

        assert alerts == []

    @pytest.mark.asyncio
    async def test_below_threshold_returns_empty(self):
        """When usage is below 80%, no alerts should be created."""
        db = AsyncMock()

        mock_quota = MagicMock()
        mock_quota.monthly_budget_usd = 100.0
        mock_quota.monthly_character_limit = None
        mock_quota.monthly_request_limit = None

        mock_quota_result = MagicMock()
        mock_quota_result.scalar_one_or_none.return_value = mock_quota

        # Monthly: $50 used -> 50% -> below all thresholds
        mock_monthly_result = MagicMock()
        mock_monthly_result.one.return_value = (30, 30000, 50.0)

        db.execute = AsyncMock(side_effect=[mock_quota_result, mock_monthly_result])

        with patch("src.services.quota_service.get_settings") as mock_settings:
            mock_settings.return_value.quota.enforcement_enabled = True
            alerts = await check_and_create_alerts(db, tenant_uuid("tenant-1"))

        assert alerts == []


class TestAlertThresholds:
    def test_thresholds_are_80_90_100(self):
        assert ALERT_THRESHOLDS == [80, 90, 100]
