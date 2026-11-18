"""The entitlement gate on the /api/tts/ws WebSocket actually closes an
unlicensed tenant's socket.

``src.api.tts``/``src.api.usage`` have carried ``LICENSE_GATE`` since #113, but
that dependency is typed for a starlette ``Request`` and cannot be mounted on a
websocket route — so until w2v_common 0.76.0 shipped ``require_license_ws``
there was no way to ask "may you" on this surface at all. With the flag armed
that meant a tenant whose w2v licence was expired or revoked was refused on
every HTTP route and SERVED on the socket: the job-notification stream for its
whole tenant channel, indefinitely, on a connection opened once. The socket
authenticated IDENTITY (``_introspect_license``) and stopped there.

Every test here drives a REAL handshake through the REAL router at the raw ASGI
layer, never route introspection. Three reasons that distinction is not
pedantry here:

* the gate is NOT a route dependency and could not be seen by introspection at
  all — it is called from inside the handler, after the handshake resolves the
  tenant (nothing stamps identity on a websocket scope; see the call site);
* ``websocket.close`` carries the verdict, so the code and reason ARE the
  observable contract, and only a driven connection produces them;
* the refusal happens pre-``accept``, and the ordering of accept-vs-close is a
  property of the message stream, not of the route object.

The paired successes (entitled tenant, flag off) are what stop this file from
passing vacuously: if the handshake were broken for an unrelated reason, every
test would "pass" on a closed socket and prove nothing.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from w2v_common.auth import (
    LICENSE_WS_CLOSE_POLICY,
    LICENSE_WS_CLOSE_UNAVAILABLE,
    clear_license_status_cache,
)
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.ws import router as ws_router
from tests.conftest import tenant_uuid

TENANT = tenant_uuid("ws-license-gate-tenant")
SUBJECT = "ws-user-under-test"
TOKEN = "opaque-licence-token-for-the-ws-handshake"  # pragma: allowlist secret

WS_PATH = "/api/tts/ws"


# ---------------------------------------------------------------------------
# Fakes — the licences upstream the gate resolves through.
#
# Deliberately the same shapes as tests/unit/test_license_gate.py: the whole
# point of the WS gate is that it derives its verdict from the SAME
# ``get_license_status`` as the HTTP gate, so the two suites must be able to
# feed it identical upstream states.
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must close, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TENANT,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _authenticated_handshake(monkeypatch):
    """Make the handshake's IDENTITY step succeed for every test in this file.

    The gate under test is the step AFTER identity. Leaving introspection real
    would reject every socket at the earlier 1008 and every assertion below
    would pass for the wrong reason — the classic way a gate test goes vacuous.
    """

    async def _introspect(token: str) -> dict[str, Any]:
        assert token == TOKEN
        return {"active": True, "tenant_id": TENANT, "subject_id": SUBJECT}

    monkeypatch.setattr("src.api.ws._introspect_license", _introspect)


@pytest.fixture
def app() -> FastAPI:
    """The REAL ws router. No middleware stamps identity — that is the point:
    a websocket scope never gets one, so the handler must stamp it itself."""
    application = FastAPI()
    application.include_router(ws_router)
    # Reached only on the entitled path, where the handler proceeds past the
    # gate. A MagicMock notification service is enough for ``_accept_or_reject``
    # to accept; this file never asserts on anything past ``accept``.
    application.state.notification_service = MagicMock()
    application.state.connection_manager = None
    application.state.settings = None
    return application


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


async def _handshake(app: FastAPI) -> list[MutableMapping[str, Any]]:
    """Drive one real websocket handshake and return the ASGI messages sent.

    Raw ASGI rather than ``TestClient`` on purpose: this is the only layer that
    shows BOTH the close code/reason AND whether an ``websocket.accept`` was
    emitted first. ``TestClient`` would collapse a refusal into a
    ``WebSocketDisconnect`` and hide the ordering.

    The socket sends nothing after connecting, so on the ALLOW path the handler
    accepts and then sees the stream end — which is exactly the shape a normal
    client disconnect has.
    """
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "root_path": "",
        "path": WS_PATH,
        "raw_path": WS_PATH.encode(),
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"x-tenant-id", f"Bearer {TOKEN}".encode()),
        ],
        "subprotocols": [],
        "state": {},
    }
    sent: list[MutableMapping[str, Any]] = []
    events = [{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000}]

    async def receive() -> dict[str, Any]:
        return events.pop(0) if events else {"type": "websocket.disconnect", "code": 1000}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _closes(sent: list[MutableMapping[str, Any]]) -> MutableMapping[str, Any]:
    """The single close frame, asserting the socket was never accepted first."""
    types = [m["type"] for m in sent]
    assert "websocket.accept" not in types, f"socket was accepted before refusal: {types}"
    closes = [m for m in sent if m["type"] == "websocket.close"]
    assert len(closes) == 1, sent
    return closes[0]


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant's socket is CLOSED, not served
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_closes_an_unentitled_socket(app, monkeypatch, inventory, code):
    """1008 with the state-specific code — the SAME vocabulary the HTTP 402 body
    carries, so one client keys off one set of strings across both transports."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    close = _closes(await _handshake(app))

    assert close["code"] == LICENSE_WS_CLOSE_POLICY == 1008
    assert close["reason"].startswith(f"{code}:"), close["reason"]


@pytest.mark.asyncio
async def test_armed_gate_closes_1013_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 1008 either: we do not know.

    1013 ("try again later") is the socket's twin of the HTTP 503 + Retry-After.
    A client can tell "come back later" from "your licence is dead" and behave
    differently — reconnect vs. stop and tell the user.
    """
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    close = _closes(await _handshake(app))

    assert close["code"] == LICENSE_WS_CLOSE_UNAVAILABLE == 1013
    assert close["reason"].startswith("license_unavailable:"), close["reason"]


@pytest.mark.asyncio
async def test_close_reason_fits_the_rfc6455_cap(app, monkeypatch):
    """A reason over 123 bytes is a protocol violation the peer may drop the
    connection over — which would turn a precise verdict into a bare disconnect."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    close = _closes(await _handshake(app))

    assert len(close["reason"].encode("utf-8")) <= 123


# ---------------------------------------------------------------------------
# The non-vacuity pairs: an entitled tenant, and the dormant flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_gate_accepts_an_entitled_tenant(app, monkeypatch):
    """The gate must not be a blanket refusal.

    This is the test that fails if the tenant is NOT stamped on
    ``websocket.state`` before the gate runs: with no tenant the gate derives
    ``none`` and closes EVERY socket, including this entitled one.
    """
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    sent = await _handshake(app)

    assert sent[0]["type"] == "websocket.accept", sent


@pytest.mark.asyncio
async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    still gets a socket — deliberately — so the rollout is a flag flip, not a
    deploy, and matches how the HTTP gate shipped."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    sent = await _handshake(app)

    assert sent[0]["type"] == "websocket.accept", sent


# ---------------------------------------------------------------------------
# The identity the gate judges is the AUTHENTICATED one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_judges_the_introspected_tenant_not_the_raw_header(app, monkeypatch):
    """A raw ``x-tenant-id`` value must never become the gated identity.

    The handshake header is a Bearer LICENCE TOKEN that is introspected; the
    tenant comes back from that call. If the gate ever read the header's text as
    a tenant instead, a caller could name an entitled tenant and be let in — a
    spoofable lane the HTTP gate does not have. Here the header text is not a
    tenant at all, and the tenant the gate asks about must be the introspected
    one.
    """
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    asked: list[tuple[str, str]] = []

    class _Recording(_FakeLicenses):
        async def get_active_license(self, *, tenant_id: str, app_key: str):
            asked.append((tenant_id, app_key))
            return _entitled()

    _licenses(monkeypatch, _Recording(active=_entitled()))

    await _handshake(app)

    assert asked and asked[0][0] == TENANT, asked
    assert TOKEN not in [t for t, _ in asked]


@pytest.mark.asyncio
async def test_unauthenticated_socket_is_still_refused_before_the_gate(app, monkeypatch):
    """Identity is still the FIRST question. No licence token ⇒ 1008
    "Unauthenticated", and the licences upstream is never consulted at all —
    the gate is an addition to that check, not a replacement for it."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    consulted: list[str] = []

    class _Loud(_FakeLicenses):
        async def get_active_license(self, *, tenant_id: str, app_key: str):
            consulted.append(tenant_id)
            return _entitled()

    _licenses(monkeypatch, _Loud(active=_entitled()))

    scope_sent: list[MutableMapping[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        scope_sent.append(message)

    await app(
        {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "path": WS_PATH,
            "raw_path": WS_PATH.encode(),
            "query_string": b"",
            "headers": [(b"host", b"testserver")],  # no licence token
            "subprotocols": [],
            "state": {},
        },
        receive,
        send,
    )

    close = _closes(scope_sent)
    assert close["code"] == 1008
    assert close["reason"] == "Unauthenticated"
    assert consulted == [], "the gate ran for an unauthenticated socket"
