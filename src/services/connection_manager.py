"""WebSocket connection tracking and management."""

from __future__ import annotations

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks active WebSocket connections per tenant."""

    def __init__(self, max_per_tenant: int = 10) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._max_per_tenant = max_per_tenant

    def connect(self, tenant_id: str, ws: WebSocket) -> bool:
        """Register connection. Returns False if limit exceeded."""
        if tenant_id not in self._connections:
            self._connections[tenant_id] = set()
        if len(self._connections[tenant_id]) >= self._max_per_tenant:
            return False
        self._connections[tenant_id].add(ws)
        return True

    def disconnect(self, tenant_id: str, ws: WebSocket) -> None:
        """Unregister connection."""
        if tenant_id in self._connections:
            self._connections[tenant_id].discard(ws)
            if not self._connections[tenant_id]:
                del self._connections[tenant_id]

    @property
    def total_connections(self) -> int:
        """Return the total number of active WebSocket connections across all tenants."""
        return sum(len(s) for s in self._connections.values())

    async def close_all(self) -> None:
        """Graceful shutdown of all connections."""
        for _tenant_id, connections in list(self._connections.items()):
            for ws in list(connections):
                try:
                    await ws.close(code=1001, reason="Server shutting down")
                except OSError, ConnectionError, RuntimeError:
                    logger.debug("Failed to close websocket during shutdown", exc_info=True)
            connections.clear()
        self._connections.clear()
