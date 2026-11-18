"""Unit tests for WebSocket connection manager."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.connection_manager import ConnectionManager
from tests.conftest import tenant_uuid


class TestConnectionManager:
    def test_init_defaults(self):
        mgr = ConnectionManager()
        assert mgr._max_per_tenant == 10
        assert mgr.total_connections == 0

    def test_init_custom_limit(self):
        mgr = ConnectionManager(max_per_tenant=5)
        assert mgr._max_per_tenant == 5

    def test_connect(self):
        mgr = ConnectionManager()
        ws = MagicMock()
        result = mgr.connect(tenant_uuid("tenant-1"), ws)
        assert result is True
        assert mgr.total_connections == 1

    def test_connect_limit_exceeded(self):
        mgr = ConnectionManager(max_per_tenant=2)
        ws1 = MagicMock()
        ws2 = MagicMock()
        ws3 = MagicMock()

        assert mgr.connect(tenant_uuid("tenant-1"), ws1) is True
        assert mgr.connect(tenant_uuid("tenant-1"), ws2) is True
        assert mgr.connect(tenant_uuid("tenant-1"), ws3) is False
        assert mgr.total_connections == 2

    def test_disconnect(self):
        mgr = ConnectionManager()
        ws = MagicMock()
        mgr.connect(tenant_uuid("tenant-1"), ws)
        mgr.disconnect(tenant_uuid("tenant-1"), ws)
        assert mgr.total_connections == 0

    def test_disconnect_removes_tenant_when_empty(self):
        mgr = ConnectionManager()
        ws = MagicMock()
        mgr.connect(tenant_uuid("tenant-1"), ws)
        mgr.disconnect(tenant_uuid("tenant-1"), ws)
        assert tenant_uuid("tenant-1") not in mgr._connections

    def test_disconnect_nonexistent_tenant(self):
        mgr = ConnectionManager()
        ws = MagicMock()
        # Should not raise
        mgr.disconnect("nonexistent", ws)

    def test_total_connections_multiple_tenants(self):
        mgr = ConnectionManager()
        for _i in range(3):
            mgr.connect(tenant_uuid("tenant-1"), MagicMock())
        for _i in range(2):
            mgr.connect(tenant_uuid("tenant-2"), MagicMock())
        assert mgr.total_connections == 5

    @pytest.mark.asyncio
    async def test_close_all(self):
        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr.connect(tenant_uuid("tenant-1"), ws1)
        mgr.connect(tenant_uuid("tenant-2"), ws2)

        await mgr.close_all()

        ws1.close.assert_called_once_with(code=1001, reason="Server shutting down")
        ws2.close.assert_called_once_with(code=1001, reason="Server shutting down")
        assert mgr.total_connections == 0

    @pytest.mark.asyncio
    async def test_close_all_handles_errors(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        ws.close = AsyncMock(side_effect=RuntimeError("already closed"))
        mgr.connect(tenant_uuid("tenant-1"), ws)

        # Should not raise
        await mgr.close_all()
        assert mgr.total_connections == 0
