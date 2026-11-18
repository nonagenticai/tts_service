"""Unit tests for WebSocket endpoint."""

import asyncio
import json
import time
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import WebSocket

from src.api.ws import websocket_job_notifications
from tests.conftest import tenant_uuid

#: Default authenticated tenant for the fake socket. The endpoint now fails
#: closed, so every test that exercises the *post-auth* behaviour has to arrive
#: authenticated; only the auth tests below opt out with ``tenant_id=None``.
AUTHED_TENANT = tenant_uuid("tenant-authed-1")


class _FakeWebSocket:
    """Minimal WebSocket fake for testing the endpoint handler."""

    def __init__(
        self,
        messages: list[str] | None = None,
        query_params: dict[str, str] | None = None,
        tenant_id: str | None = AUTHED_TENANT,
        headers: dict[str, str] | None = None,
    ):
        self._messages = list(messages or [])
        self._idx = 0
        self._sent: list[str | dict[str, Any]] = []
        self._closed = False
        self._accepted = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.query_params: dict[str, str] = query_params or {}
        self.headers: dict[str, str] = headers or {}

        # Minimal state
        self.state = MagicMock()
        self.app: Any = None
        if tenant_id is None:
            del self.state.tenant_id  # unauthenticated socket
        else:
            self.state.tenant_id = tenant_id

    async def accept(self) -> None:
        self._accepted = True

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self._closed = True
        self.close_code = code
        self.close_reason = reason

    async def receive_text(self) -> str:
        if self._idx >= len(self._messages):
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect()
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_text(self, data: str) -> None:
        self._sent.append(data)

    async def send_json(self, data: dict[str, Any]) -> None:
        self._sent.append(data)


def _as_ws(fake: _FakeWebSocket) -> WebSocket:
    """Cast a _FakeWebSocket to WebSocket for the type checker."""
    return cast(WebSocket, fake)


class TestWebSocketNoNotificationService:
    @pytest.mark.asyncio
    async def test_closes_when_no_notification_service(self):
        ws = _FakeWebSocket()
        app = MagicMock()
        app.state = MagicMock(spec=[])  # no notification_service attr
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))
        assert ws._closed is True
        assert ws.close_code == 4500


class TestWebSocketConnectionLimit:
    @pytest.mark.asyncio
    async def test_rejects_when_connection_limit_exceeded(self):
        ws = _FakeWebSocket()
        app = MagicMock()
        notification_service = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = False
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))
        assert ws._closed is True
        assert ws.close_code == 4008


class TestWebSocketActions:
    @pytest.mark.asyncio
    async def test_ping_action(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "ping"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        # Should have sent pong
        pong_msgs = [m for m in ws._sent if isinstance(m, dict) and m.get("event") == "pong"]
        assert len(pong_msgs) == 1
        assert "timestamp" in pong_msgs[0]

    @pytest.mark.asyncio
    async def test_subscribe_job(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "subscribe", "job_id": "j-123"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.subscribe = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        notification_service.subscribe.assert_called_once()
        call_args = notification_service.subscribe.call_args
        assert call_args[0][0] == "job:j-123"

    @pytest.mark.asyncio
    async def test_subscribe_batch(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "subscribe", "batch_id": "b-456"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.subscribe = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        notification_service.subscribe.assert_called_once()
        call_args = notification_service.subscribe.call_args
        assert call_args[0][0] == "batch:b-456"

    @pytest.mark.asyncio
    async def test_subscribe_missing_id(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "subscribe"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        error_msgs = [m for m in ws._sent if isinstance(m, dict) and m.get("event") == "error"]
        assert len(error_msgs) == 1
        assert "Missing" in error_msgs[0]["message"]

    @pytest.mark.asyncio
    async def test_subscribe_tenant(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "subscribe_tenant"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.subscribe = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        notification_service.subscribe.assert_called_once()
        call_args = notification_service.subscribe.call_args
        assert call_args[0][0] == f"tenant:{AUTHED_TENANT}"

    @pytest.mark.asyncio
    async def test_unsubscribe_job(self):
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"action": "subscribe", "job_id": "j-1"}),
                json.dumps({"action": "unsubscribe", "job_id": "j-1"}),
            ]
        )
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.subscribe = MagicMock()
        notification_service.unsubscribe = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        notification_service.unsubscribe.assert_called_once()
        assert notification_service.unsubscribe.call_args[0][0] == "job:j-1"

    @pytest.mark.asyncio
    async def test_unsubscribe_batch(self):
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"action": "subscribe", "batch_id": "b-1"}),
                json.dumps({"action": "unsubscribe", "batch_id": "b-1"}),
            ]
        )
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.subscribe = MagicMock()
        notification_service.unsubscribe = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        notification_service.unsubscribe.assert_called_once()
        assert notification_service.unsubscribe.call_args[0][0] == "batch:b-1"

    @pytest.mark.asyncio
    async def test_unsubscribe_missing_id(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "unsubscribe"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))
        # Should just continue, not send an error

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        ws = _FakeWebSocket(messages=[json.dumps({"action": "foobar"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        error_msgs = [m for m in ws._sent if isinstance(m, dict) and m.get("event") == "error"]
        assert len(error_msgs) == 1
        assert "Unknown action" in error_msgs[0]["message"]

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        ws = _FakeWebSocket(messages=["not-json!!!"])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        error_msgs = [m for m in ws._sent if isinstance(m, dict) and m.get("event") == "error"]
        assert len(error_msgs) == 1
        assert "Invalid JSON" in error_msgs[0]["message"]

    @pytest.mark.parametrize("frame", ["[]", '"just-a-string"', "3", "null"])
    @pytest.mark.asyncio
    async def test_non_object_json_frame_is_answered_and_the_socket_survives(
        self, frame: str
    ) -> None:
        """VALID JSON THAT IS NOT AN OBJECT must not kill the connection.

        ``json.loads`` accepts every JSON value, so ``[]``/``"s"``/``3``/``null``
        all parse and then reach ``msg.get("action")`` — ``AttributeError`` on
        each. That is not in the receive loop's
        ``(OSError, ConnectionError, TimeoutError, RuntimeError)``, so pre-fix it
        escaped ``websocket_job_notifications`` and one such frame from any
        authenticated client tore the socket down.

        The SECOND frame is the real assertion: it proves the loop reached the
        next message, which is exactly what ``_dispatch_message``'s docstring
        promises and what the crash broke. Asserting only on the error event
        would pass on a socket that then died.
        """
        ws = _FakeWebSocket(messages=[frame, json.dumps({"action": "ping"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        # Pre-fix this RAISED AttributeError out of the handler.
        await websocket_job_notifications(_as_ws(ws))

        error_msgs = [m for m in ws._sent if isinstance(m, dict) and m.get("event") == "error"]
        assert len(error_msgs) == 1
        assert "JSON object" in error_msgs[0]["message"]
        # The loop continued: the frame AFTER the bad one was served.
        assert any(m.get("event") == "pong" for m in ws._sent if isinstance(m, dict))

    @pytest.mark.asyncio
    async def test_cleanup_on_disconnect(self):
        ws = _FakeWebSocket(messages=[])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        connection_manager.disconnect = MagicMock()
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        notification_service.unsubscribe_all.assert_called_once()
        connection_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_connection_manager(self):
        """Tests the path where connection_manager is None."""
        ws = _FakeWebSocket(messages=[json.dumps({"action": "ping"})])
        app = MagicMock()
        notification_service = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        app.state.notification_service = notification_service
        app.state.connection_manager = None
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        # Should work fine without connection manager
        pong_msgs = [m for m in ws._sent if isinstance(m, dict) and m.get("event") == "pong"]
        assert len(pong_msgs) == 1

    @pytest.mark.asyncio
    async def test_tenant_id_from_state(self):
        """Tests that tenant_id is pulled from websocket.state if available."""
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
        )
        # Set tenant_id on state
        ws.state = MagicMock()
        ws.state.tenant_id = tenant_uuid("state-tenant-42")

        app = MagicMock()
        notification_service = MagicMock()
        notification_service.subscribe = MagicMock()
        notification_service.unsubscribe_all = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        ws.app = app

        await websocket_job_notifications(_as_ws(ws))

        call_args = notification_service.subscribe.call_args
        assert call_args[0][0] == f"tenant:{tenant_uuid('state-tenant-42')}"


class TestWebSocketTenantFailClosed:
    """The socket must never take its tenant from the client.

    ``/api/tts/ws`` used to resolve ``query_params["tenant_id"]`` (falling back
    to ``DEFAULT_TENANT_ID``) whenever ``websocket.state.tenant_id`` was absent
    — which is ALWAYS in production, because every middleware in the fleet auth
    stack is a ``BaseHTTPMiddleware`` and those skip the ``websocket`` scope
    entirely. That let any unauthenticated client subscribe to any tenant's
    Redis notification channel, with no RLS backstop behind it.
    """

    @staticmethod
    def _app(notification_service: MagicMock) -> MagicMock:
        app = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        return app

    @pytest.mark.asyncio
    async def test_unauthenticated_socket_with_query_tenant_is_closed_1008(self):
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            query_params={"tenant_id": tenant_uuid("victim-tenant")},
            tenant_id=None,
        )
        ws.app = self._app(notification_service)

        await websocket_job_notifications(_as_ws(ws))

        assert ws._closed is True
        assert ws.close_code == 1008
        # Never accepted, and — the part that actually matters — never
        # subscribed to anything on the victim tenant's channel.
        assert ws._accepted is False
        notification_service.subscribe.assert_not_called()
        notification_service.unsubscribe_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthenticated_socket_without_query_tenant_is_closed_1008(self):
        """No query param either: no DEFAULT_TENANT_ID fallback on this path."""
        notification_service = MagicMock()
        ws = _FakeWebSocket(messages=[json.dumps({"action": "subscribe_tenant"})], tenant_id=None)
        ws.app = self._app(notification_service)

        await websocket_job_notifications(_as_ws(ws))

        assert ws.close_code == 1008
        notification_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_blank_state_tenant_is_closed_1008(self):
        notification_service = MagicMock()
        ws = _FakeWebSocket(messages=[], tenant_id="   ")
        ws.app = self._app(notification_service)

        await websocket_job_notifications(_as_ws(ws))

        assert ws.close_code == 1008
        notification_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_authenticated_socket_connects_and_subscribes_to_own_tenant(self):
        notification_service = MagicMock()
        ws = _FakeWebSocket(messages=[json.dumps({"action": "subscribe_tenant"})])
        ws.app = self._app(notification_service)

        await websocket_job_notifications(_as_ws(ws))

        assert ws._accepted is True
        assert ws.close_code != 1008
        notification_service.subscribe.assert_called_once()
        assert notification_service.subscribe.call_args[0][0] == f"tenant:{AUTHED_TENANT}"

    @pytest.mark.asyncio
    async def test_query_tenant_does_not_override_authenticated_tenant(self):
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            query_params={"tenant_id": tenant_uuid("victim-tenant")},
        )
        ws.app = self._app(notification_service)

        await websocket_job_notifications(_as_ws(ws))

        notification_service.subscribe.assert_called_once()
        subscribed_key = notification_service.subscribe.call_args[0][0]
        assert subscribed_key == f"tenant:{AUTHED_TENANT}"
        assert tenant_uuid("victim-tenant") not in subscribed_key

    @pytest.mark.asyncio
    async def test_handshake_license_token_authenticates(self):
        """``X-Tenant-Id: Bearer <token>`` is the supported WS auth transport."""
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            query_params={"tenant_id": tenant_uuid("victim-tenant")},
            tenant_id=None,
            headers={"x-tenant-id": "Bearer tok-abc"},  # pragma: allowlist secret
        )
        ws.app = self._app(notification_service)

        async def _fake_introspect(token: str) -> dict[str, Any]:
            assert token == "tok-abc"  # pragma: allowlist secret
            return {"active": True, "tenant_id": tenant_uuid("licensed-tenant")}

        with patch("src.api.ws._introspect_license", _fake_introspect):
            await websocket_job_notifications(_as_ws(ws))

        notification_service.subscribe.assert_called_once()
        assert notification_service.subscribe.call_args[0][0] == (
            f"tenant:{tenant_uuid('licensed-tenant')}"
        )

    @pytest.mark.asyncio
    async def test_inactive_license_token_is_closed_1008(self):
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            tenant_id=None,
            headers={"x-tenant-id": "Bearer revoked"},  # pragma: allowlist secret
        )
        ws.app = self._app(notification_service)

        async def _fake_introspect(token: str) -> dict[str, Any]:
            return {"active": False, "reason": "token_invalid"}

        with patch("src.api.ws._introspect_license", _fake_introspect):
            await websocket_job_notifications(_as_ws(ws))

        assert ws.close_code == 1008
        assert ws._accepted is False
        notification_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_license_without_tenant_is_closed_1008(self):
        """An active license carrying no tenant is a bug, never an identity."""
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[],
            tenant_id=None,
            headers={"x-tenant-id": "Bearer tok"},  # pragma: allowlist secret
        )
        ws.app = self._app(notification_service)

        async def _fake_introspect(token: str) -> dict[str, Any]:
            return {"active": True, "tenant_id": "  "}

        with patch("src.api.ws._introspect_license", _fake_introspect):
            await websocket_job_notifications(_as_ws(ws))

        assert ws.close_code == 1008
        notification_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_introspection_raising_still_closes_the_socket(self):
        """An exception ESCAPING ``_introspect_license`` must fail closed, not out.

        ``_introspect_license`` swallows the transport errors it anticipates, but
        it is a shared upstream: a misconfigured ``LICENSE_INTROSPECT_URL``
        (``httpx.InvalidURL``), a signature change (``TypeError``) or a cache-layer
        ``AttributeError`` all surface here. Without the ``try`` in
        ``_resolve_identity`` the exception propagates out of the endpoint and
        the reject behaviour becomes the ASGI server's business rather than ours
        — and, critically, ``close()`` is never called, so the connection-manager
        and notification bookkeeping never run.

        Mutation this kills: delete the ``try/except Exception`` from
        ``_resolve_identity`` (or narrow it to a tuple that misses
        ``RuntimeError``) and this test fails — the raise escapes
        ``websocket_job_notifications`` and pytest reports the error instead of a
        clean 1008 close.
        """
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            tenant_id=None,
            headers={"x-tenant-id": "Bearer boom"},  # pragma: allowlist secret
        )
        ws.app = self._app(notification_service)

        async def _exploding_introspect(token: str) -> dict[str, Any]:
            raise RuntimeError("licenses upstream exploded")

        with patch("src.api.ws._introspect_license", _exploding_introspect):
            # Must NOT raise: the endpoint is expected to absorb this and reject.
            await websocket_job_notifications(_as_ws(ws))

        assert ws._closed is True
        assert ws.close_code == 1008
        assert ws._accepted is False
        notification_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_introspection_failure_does_not_log_the_token(self):
        """The fail-closed path must not leak the license token into logs.

        Mutation this kills: change the ``logger.exception`` call in
        ``_resolve_identity`` to interpolate the token (e.g.
        ``logger.exception("... token=%s", token)``) and this test fails.
        """
        notification_service = MagicMock()
        secret = "super-secret-token-value"  # pragma: allowlist secret
        ws = _FakeWebSocket(
            messages=[],
            tenant_id=None,
            headers={"x-tenant-id": f"Bearer {secret}"},
        )
        ws.app = self._app(notification_service)

        async def _exploding_introspect(token: str) -> dict[str, Any]:
            raise RuntimeError("licenses upstream exploded")

        with (
            patch("src.api.ws._introspect_license", _exploding_introspect),
            patch("src.api.ws.logger") as mock_logger,
        ):
            await websocket_job_notifications(_as_ws(ws))

        assert ws.close_code == 1008
        rendered = " ".join(
            repr(call)
            for call in mock_logger.exception.call_args_list + mock_logger.warning.call_args_list
        )
        assert secret not in rendered


class TestWebSocketHandshakeLicenceLane:
    """``_handshake_license_token`` must mirror the CANONICAL w2v_common rule.

    ``src/api/ws.py`` cannot delegate to ``AuthMiddleware``: every fleet auth
    middleware is a ``BaseHTTPMiddleware``, and starlette's
    ``BaseHTTPMiddleware.__call__`` returns early for any non-``http`` scope, so
    ``dispatch`` never runs on a ``websocket`` scope and
    ``websocket.state.license_token`` is never populated. The handshake reads the
    headers itself — which is exactly why it silently drifted: it froze on the
    PRE-0.74.0 rule (``X-Tenant-Id: Bearer`` only) while w2v_common widened to
    also accept an opaque ``Authorization: Bearer``. Behind the ``license-authz``
    ext-authz gate — which reads the token ONLY from ``Authorization`` — that
    drift rejects every socket. No test failed for it.
    """

    @staticmethod
    def _app(notification_service: MagicMock) -> MagicMock:
        app = MagicMock()
        connection_manager = MagicMock()
        connection_manager.connect.return_value = True
        app.state.notification_service = notification_service
        app.state.connection_manager = connection_manager
        return app

    async def _run(
        self, headers: dict[str, str], introspect_tenant: str
    ) -> tuple[MagicMock, list[str], Any]:
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            tenant_id=None,
            headers=headers,
        )
        ws.app = self._app(notification_service)
        seen: list[str] = []

        async def _fake_introspect(token: str) -> dict[str, Any]:
            seen.append(token)
            return {"active": True, "tenant_id": introspect_tenant}

        with patch("src.api.ws._introspect_license", _fake_introspect):
            await websocket_job_notifications(_as_ws(ws))
        return notification_service, seen, ws

    @pytest.mark.asyncio
    async def test_opaque_authorization_bearer_resolves_a_tenant(self):
        """THE REGRESSION. Pre-fix this closed 1008 with an empty introspect list.

        Mutation this kills: drop the ``Authorization`` branch from
        ``_handshake_license_token`` (i.e. restore the pre-0.74.0 copy) and the
        socket resolves no tenant at all.
        """
        licensed = tenant_uuid("authz-lane-tenant")
        notification_service, seen, ws = await self._run(
            {"authorization": "Bearer opaque-licence-tok"},  # pragma: allowlist secret
            licensed,
        )

        assert seen == ["opaque-licence-tok"]  # pragma: allowlist secret
        assert ws.close_code != 1008
        notification_service.subscribe.assert_called_once()
        assert notification_service.subscribe.call_args[0][0] == f"tenant:{licensed}"

    @pytest.mark.asyncio
    async def test_jwt_shaped_authorization_bearer_never_reaches_introspection(self):
        """A caller's own JWT is NOT a licence token — it must not be introspected.

        Mutation this kills: drop the ``not _looks_like_jwt(...)`` guard and a
        three-segment JWT is submitted to licence introspection, reviving the
        2026-04-26 prod 401 class.
        """
        notification_service = MagicMock()
        jwt_like = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1MSJ9.sig"  # pragma: allowlist secret
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            tenant_id=None,
            headers={"authorization": f"Bearer {jwt_like}"},
        )
        ws.app = self._app(notification_service)
        seen: list[str] = []

        async def _fake_introspect(token: str) -> dict[str, Any]:
            seen.append(token)
            return {"active": True, "tenant_id": tenant_uuid("must-not-happen")}

        with patch("src.api.ws._introspect_license", _fake_introspect):
            await websocket_job_notifications(_as_ws(ws))

        assert seen == []
        assert ws.close_code == 1008
        assert ws._accepted is False
        notification_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_x_tenant_id_bearer_still_resolves(self):
        """The deprecated lane stays load-bearing — every current w2v caller uses it."""
        licensed = tenant_uuid("legacy-lane-tenant")
        notification_service, seen, _ws = await self._run(
            {"x-tenant-id": "Bearer legacy-tok"},  # pragma: allowlist secret
            licensed,
        )

        assert seen == ["legacy-tok"]  # pragma: allowlist secret
        notification_service.subscribe.assert_called_once()
        assert notification_service.subscribe.call_args[0][0] == f"tenant:{licensed}"

    @pytest.mark.asyncio
    async def test_x_tenant_id_wins_when_both_headers_are_present(self):
        """PRECEDENCE. ``BaseServiceClient`` puts its SERVICE-ACCOUNT token on
        ``Authorization`` and the CALLER's delegated token on ``X-Tenant-Id``.

        Mutation this kills: check ``Authorization`` first and every delegated
        call is re-attributed from the customer's tenant to the service
        account's — a silent cross-tenant mis-billing, not an error.
        """
        licensed = tenant_uuid("delegated-tenant")
        notification_service, seen, _ws = await self._run(
            {
                "x-tenant-id": "Bearer delegated-tok",  # pragma: allowlist secret
                "authorization": "Bearer service-account-tok",  # pragma: allowlist secret
            },
            licensed,
        )

        assert seen == ["delegated-tok"]  # pragma: allowlist secret
        assert "service-account-tok" not in seen  # pragma: allowlist secret
        notification_service.subscribe.assert_called_once()
        assert notification_service.subscribe.call_args[0][0] == f"tenant:{licensed}"

    @pytest.mark.asyncio
    async def test_bare_authorization_without_bearer_is_not_a_credential(self):
        notification_service = MagicMock()
        ws = _FakeWebSocket(
            messages=[json.dumps({"action": "subscribe_tenant"})],
            tenant_id=None,
            headers={"authorization": "Basic dXNlcjpwdw=="},  # pragma: allowlist secret
        )
        ws.app = self._app(notification_service)
        seen: list[str] = []

        async def _fake_introspect(token: str) -> dict[str, Any]:
            seen.append(token)
            return {"active": True, "tenant_id": tenant_uuid("must-not-happen")}

        with patch("src.api.ws._introspect_license", _fake_introspect):
            await websocket_job_notifications(_as_ws(ws))

        assert seen == []
        assert ws.close_code == 1008


class TestWebSocketRejectIsAHandshakeFailure:
    """Lock in what an unauthenticated client ACTUALLY observes on the wire.

    The endpoint closes before ``accept``, so there is no open WebSocket to
    carry a close frame and the 1008 never reaches the client. Measured against
    a real server (uvicorn + websockets) the handshake fails with **HTTP 403**;
    under Starlette's ``TestClient`` the same rejection surfaces as a bare
    ``WebSocketDisconnect`` carrying no code. Either way: no 1008.

    This test exists to keep the docstrings honest. It is the executable half of
    the ``WS_POLICY_VIOLATION`` comment.
    """

    @staticmethod
    def _app() -> Any:
        from fastapi import FastAPI

        from src.api.ws import router

        app = FastAPI()
        app.include_router(router)
        app.state.notification_service = MagicMock()
        app.state.connection_manager = None
        app.state.settings = None
        return app

    @pytest.mark.asyncio
    async def test_close_is_sent_while_still_connecting_never_after_accept(self):
        """The app must emit ``websocket.close`` with NO ``websocket.accept`` before it.

        Asserted at the raw ASGI layer rather than through ``TestClient``,
        because ``TestClient`` does not perform a real handshake — it just
        relays ASGI messages, so it *does* hand the caller the 1008 and would
        happily pass either way. The wire behaviour is decided by this one fact:
        the close is sent while the connection is CONNECTING.

        Mutation this kills: move ``await websocket.accept()`` above the
        ``_resolve_identity`` reject in ``websocket_job_notifications``
        (accept-then-close). ``websocket.accept`` then appears in ``sent`` and
        the assertion fails.
        """
        app = self._app()
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "path": "/api/tts/ws",
            "raw_path": b"/api/tts/ws",
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
            "subprotocols": [],
        }
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "websocket.connect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await app(scope, receive, send)

        types = [m["type"] for m in sent]
        assert "websocket.accept" not in types, f"connection was accepted before reject: {types}"
        assert types == ["websocket.close"], types

    def test_real_server_refuses_the_handshake_with_http_403(self):
        """The documented client-observable signal, verified against a real server.

        This is the executable proof of the ``WS_POLICY_VIOLATION`` comment:
        because the close is sent pre-accept, uvicorn fails the HTTP upgrade and
        the 1008 is never framed. Clients must detect a 403, not a close code.

        Mutation this kills: accept-then-close in
        ``websocket_job_notifications`` — the handshake would then SUCCEED
        (HTTP 101) and ``websockets.connect`` would not raise ``InvalidStatus``
        at all, so the ``pytest.raises`` fails.
        """
        import socket
        import threading

        import uvicorn
        import websockets
        from websockets.exceptions import InvalidStatus

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        config = uvicorn.Config(self._app(), host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(200):  # poll for readiness rather than sleeping blind
                if getattr(server, "started", False):
                    break
                time.sleep(0.05)
            assert getattr(server, "started", False), "uvicorn did not start"

            async def _connect() -> None:
                async with websockets.connect(f"ws://127.0.0.1:{port}/api/tts/ws"):
                    pass  # pragma: no cover - handshake must not succeed

            with pytest.raises(InvalidStatus) as excinfo:
                asyncio.run(_connect())
            assert excinfo.value.response.status_code == 403
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    def test_query_string_tenant_cannot_open_the_socket(self):
        """The old ``?tenant_id=`` self-service channel selector stays dead.

        Mutation this kills: reinstate any ``query_params.get("tenant_id", ...)``
        fallback in ``_resolve_identity`` — the connect would succeed and this
        ``pytest.raises`` fails.
        """
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        client = TestClient(self._app())
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/tts/ws?tenant_id=victim-tenant"):
                pass  # pragma: no cover - connect must not succeed


class TestWebSocketTaskCleanup:
    """Lock in the sender/heartbeat task cleanup on disconnect."""

    @pytest.mark.asyncio
    async def test_no_task_leak_across_many_connections(self):
        """Opening + closing 50 WS connections must not leave background tasks running.

        Regression test for the task-leak path where sender_task / heartbeat_loop
        were cancelled but never awaited, leaving them in the event loop and
        growing ``asyncio.all_tasks()`` under churn.
        """
        before = len(asyncio.all_tasks())

        for _ in range(50):
            ws = _FakeWebSocket(messages=[json.dumps({"action": "ping"})])
            app = MagicMock()
            notification_service = MagicMock()
            notification_service.unsubscribe_all = MagicMock()
            connection_manager = MagicMock()
            connection_manager.connect.return_value = True
            app.state.notification_service = notification_service
            app.state.connection_manager = connection_manager
            ws.app = app

            await websocket_job_notifications(_as_ws(ws))

        # Yield once to let any pending callbacks finish draining.
        await asyncio.sleep(0)
        after = len(asyncio.all_tasks())

        # Allow a small drift (current task itself, test fixtures) but not
        # the 2 x 50 = 100 leaked tasks the old code would have left behind.
        assert after - before <= 2, (
            f"WebSocket task leak detected: {after - before} extra tasks "
            f"after 50 connect/disconnect cycles"
        )
