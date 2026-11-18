"""WebSocket endpoint for real-time job notifications."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid as _uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# The entitlement gate is imported straight from w2v_common, like
# ``LICENSE_GATE`` in src/api/tts.py: it is shared POLICY, and a local
# re-export would be one more place for the socket's verdict to drift from the
# API's.
from w2v_common.auth import require_license_ws

# Identity, by contrast, goes via the repo's own re-export layer rather than
# reaching into w2v_common directly — src.auth.middleware is this service's
# single seam onto the shared auth implementation.
from src.auth.middleware import _introspect_license, _looks_like_jwt

logger = logging.getLogger(__name__)

# Close code for an unauthenticated / unresolvable-tenant socket (RFC 6455
# 1008 "policy violation").
#
# ⚠️ This code does NOT reach the client, by design. We close BEFORE ``accept``,
# so the ASGI server is still in the CONNECTING state and there is no open
# WebSocket to carry a close frame; it fails the HTTP handshake instead.
# Measured, not assumed (uvicorn 0.3x / starlette 1.3.1 / websockets 16.0):
#
#   * real wire  -> ``websockets.exceptions.InvalidStatus: server rejected
#     WebSocket connection: HTTP 403`` (``exc.response.status_code == 403``)
#   * TestClient -> ``starlette.websockets.WebSocketDisconnect`` with no code
#
# So 1008 is only ever an internal marker: it is what the handler passes to
# ``close()`` and what the unit tests assert on the fake socket. Client-side,
# the observable signal is a **403 handshake failure**, and that is what
# consumers must be told to detect. Accepting first would make the 1008
# deliverable, at the cost of accepting an unauthenticated connection — not
# worth it for a diagnostic (see the endpoint docstring).
WS_POLICY_VIOLATION = 1008

router = APIRouter(prefix="/api/tts", tags=["tts-ws"])


# ---------------------------------------------------------------------------
# Helpers (each isolated and individually testable). Splitting the previously
# monolithic ``websocket_job_notifications`` (C901: 26) keeps the orchestrator
# under the C901 budget of 10 and untangles the happy-path from teardown.
# ---------------------------------------------------------------------------


def _canonical_tenant(value: object) -> str | None:
    """Return ``value`` as canonical lowercase-dashed uuid text, else ``None``.

    Since revision ``005_tenant_id_uuid`` a job's ``tenant_id`` is a postgres
    ``uuid``, which postgres renders back in exactly one spelling. The Redis
    notification channel is built as ``f"tenant:{tenant_id}"`` from THAT value
    on the publisher side, so a socket that subscribed under an uppercase or
    braced spelling of the same tenant would sit on a channel nobody publishes
    to and silently receive nothing — a "notifications are broken" bug with no
    error anywhere. Folding here is what keeps subscriber and publisher on one
    key.

    A non-uuid (a legacy slug, a leaked ``Bearer …`` value) returns ``None`` and
    the caller rejects the handshake, rather than opening a socket on a channel
    that can never correspond to a stored job.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return str(_uuid.UUID(value.strip()))
    except ValueError:
        return None


def _state_tenant_id(websocket: WebSocket) -> str | None:
    """Return ``websocket.state.tenant_id`` as canonical uuid text, else ``None``.

    Only a real, non-empty, uuid-shaped string counts. An empty/whitespace
    tenant is always a bug (see the fleet three-piece auth invariant), never a
    valid identity — subscribing under it would put the socket on a channel it
    must not read.
    """
    state = getattr(websocket, "state", None)
    if state is None:
        return None
    return _canonical_tenant(getattr(state, "tenant_id", None))


def _state_subject_id(websocket: WebSocket) -> str | None:
    """Return a non-empty ``websocket.state.subject_id``, else ``None``.

    Subject is NOT identity here and is never used to pick a channel — it only
    refines the entitlement gate's verdict (an authoritative "no active licence"
    404 becomes none/expired/revoked from that subject's inventory). So unlike
    the tenant it is resolved tolerantly and is allowed to be absent; it is also
    not uuid-folded, because a subject is an opaque IdP identifier, not a uuid.

    The ``isinstance`` guard is load-bearing, not defensive noise: the unit
    fakes give ``state`` a ``MagicMock``, whose every attribute is a truthy
    ``MagicMock``. Without it, ``str(...)`` of that mock would be handed to the
    licences upstream as a subject id.
    """
    state = getattr(websocket, "state", None)
    if state is None:
        return None
    subject = getattr(state, "subject_id", None)
    if isinstance(subject, str) and subject.strip():
        return subject.strip()
    return None


def _bearer(headers: Mapping[str, str], name: str) -> str | None:
    """Return the ``Bearer`` value of header ``name`` (lowercase), else ``None``.

    Real Starlette ``Headers`` are case-insensitive, but the unit fakes pass a
    plain ``dict``; both spellings are probed so the two behave identically.
    """
    raw = headers.get(name) or headers.get(name.title())
    if raw and raw.lower().startswith("bearer "):
        return raw[7:].strip() or None
    return None


def _handshake_license_token(websocket: WebSocket) -> str | None:
    """Extract an opaque licence token from this handshake, canonical rule.

    ⚠️ This MIRRORS ``AuthMiddleware._extract_license_token`` (w2v_common
    >= 0.74.0) and must not drift from it. It cannot simply delegate: every
    middleware in the fleet auth stack is a ``BaseHTTPMiddleware`` subclass, and
    starlette's ``BaseHTTPMiddleware.__call__`` returns early for any scope whose
    ``type`` is not ``"http"`` — so ``dispatch`` never runs on a ``websocket``
    scope and ``websocket.state.license_token`` is never populated here. The
    handshake has to read the headers itself.

    TWO shapes are accepted, in this order:

    1. ``X-Tenant-Id: Bearer <token>`` — w2v's own legacy convention. DEPRECATED
       but load-bearing, and checked FIRST on purpose: ``BaseServiceClient`` puts
       the CALLER's delegated token here while its constructor-baked
       SERVICE-ACCOUNT token sits on ``Authorization``. Preferring
       ``Authorization`` would silently re-attribute every delegated call from
       the customer's tenant to the service account's tenant.
    2. ``Authorization: Bearer <opaque>`` — the PLATFORM convention, and the only
       shape the ``license-authz`` ext-authz gate can read. Required for w2v to
       sit behind that gate at all: once gated, the licence rides
       ``Authorization`` and the old ``X-Tenant-Id``-only rule resolved no tenant,
       so the socket was rejected — failing closed, but for the wrong reason.

       Reached ONLY when the bearer is not JWT-shaped. A JWT on ``Authorization``
       is an upstream caller's own identity token and must NEVER be submitted to
       licence introspection (the 2026-04-26 prod 401s); ``_looks_like_jwt`` is
       imported from w2v_common rather than re-implemented so that test cannot
       drift either.
    """
    headers = getattr(websocket, "headers", None) or {}
    token = _bearer(headers, "x-tenant-id")
    if token:
        return token
    bearer = _bearer(headers, "authorization")
    if bearer and not _looks_like_jwt(bearer):
        return bearer
    return None


async def _resolve_identity(websocket: WebSocket) -> tuple[str | None, str | None]:
    """Resolve ``(tenant_id, subject_id)`` for this socket; tenant ``None`` ⇒ reject.

    There is no query-string fallback and no ``DEFAULT_TENANT_ID`` fallback on
    this path — both were exploitable. The fleet auth stack is built entirely
    from ``BaseHTTPMiddleware`` subclasses, which pass a ``websocket`` scope
    straight through to the app without running ``dispatch``; so
    ``websocket.state.tenant_id`` is never populated by middleware here, and the
    old ``query_params.get("tenant_id", DEFAULT_TENANT_ID)`` meant an
    unauthenticated client picked its own Redis notification channel. Redis has
    no RLS backstop, so that read another tenant's job events.

    Resolution order:

    1. ``websocket.state.tenant_id`` — set by an ASGI-level authenticator if one
       is ever added in front of this route (or by a test).
    2. An opaque licence token from the handshake headers — ``X-Tenant-Id:
       Bearer <token>`` first, then a non-JWT ``Authorization: Bearer <token>``
       (see ``_handshake_license_token``) — validated through the same license
       introspection the REST path uses.

    Neither ⇒ ``(None, None)`` ⇒ the caller closes with 1008 without accepting.

    The SUBJECT rides along because the entitlement gate mounted on this route
    wants the same pair ``AuthMiddleware`` stamps on an HTTP request, and it is
    only knowable on the same introspection call that yields the tenant —
    resolving it separately would introspect twice. Its fallback to the tenant
    id mirrors ``w2v_common.auth.middleware`` exactly (a tenant-scoped service
    account token legitimately carries no subject), so the socket and the API
    derive the SAME licence state for the same caller instead of two.
    """
    tenant_id = _state_tenant_id(websocket)
    if tenant_id is not None:
        return tenant_id, _state_subject_id(websocket) or tenant_id

    token = _handshake_license_token(websocket)
    if not token:
        return None, None
    try:
        result = await _introspect_license(token)
    except Exception:
        # FAIL CLOSED on ANY introspection error. The breadth is the point, not
        # laziness: this function's whole contract is "no verified tenant ⇒ no
        # socket", and enumerating exception types would make an unforeseen one
        # (a misconfigured LICENSE_INTROSPECT_URL raising httpx.InvalidURL, a
        # TypeError from a changed w2v_common signature, an AttributeError in
        # the cache layer) propagate out of the endpoint instead. That escape is
        # not merely uglier — it makes the reject behaviour depend on the ASGI
        # server's unhandled-exception policy rather than on this module.
        #
        # `except Exception` does NOT catch asyncio.CancelledError (a
        # BaseException since 3.8), so shutdown and task-cancellation semantics
        # are untouched. The token is never logged — `logger.exception` renders
        # the traceback only, and `_introspect_license` takes care never to put
        # the token in a message.
        #
        # No lint suppression is needed here: ruff's blind-except rule (BLE001)
        # exempts a handler whose body calls `logger.exception`. Keep that call
        # if this block is ever edited, or the lint comes back.
        logger.exception("WebSocket rejected: license introspection raised")
        return None, None
    if not result or not result.get("active"):
        return None, None
    # An active license MUST carry a tenant; an empty one would re-open the
    # exact hole this function exists to close. A non-uuid one is rejected the
    # same way -- see ``_canonical_tenant``.
    tenant_id = _canonical_tenant(result.get("tenant_id"))
    if tenant_id is None:
        return None, None
    subject = result.get("subject_id")
    subject_id = subject.strip() if isinstance(subject, str) and subject.strip() else tenant_id
    return tenant_id, subject_id


def _get_ws_settings(app: Any) -> tuple[int, int]:
    """Return ``(queue_size, heartbeat_interval)`` from app settings or defaults."""
    ws_settings = getattr(app.state, "settings", None)
    queue_size = ws_settings.websocket.event_queue_size if ws_settings else 100
    heartbeat_interval = ws_settings.websocket.heartbeat_interval if ws_settings else 30
    return queue_size, heartbeat_interval


async def _sender_loop(websocket: WebSocket, event_queue: asyncio.Queue[str]) -> None:
    """Forward queued events to the WebSocket until cancelled or disconnected."""
    try:
        while True:
            msg = await event_queue.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect, asyncio.CancelledError:
        pass
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
        logger.debug("Sender task error: %s", exc)


async def _heartbeat_loop(websocket: WebSocket, interval: int) -> None:
    """Send periodic heartbeat messages to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(interval)
            await websocket.send_json(
                {"event": "heartbeat", "timestamp": datetime.now(UTC).isoformat()}
            )
    except WebSocketDisconnect, asyncio.CancelledError:
        pass
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
        logger.debug("Heartbeat task error: %s", exc)


def _key_from_subscribe_msg(msg: dict[str, Any]) -> str | None:
    """Translate a subscribe/unsubscribe payload into a notification key.

    Returns ``None`` if the payload does not name a job_id or batch_id.
    """
    job_id = msg.get("job_id")
    if job_id:
        return f"job:{job_id}"
    batch_id = msg.get("batch_id")
    if batch_id:
        return f"batch:{batch_id}"
    return None


async def _handle_subscribe(
    websocket: WebSocket,
    msg: dict[str, Any],
    notification_service: Any,
    event_queue: asyncio.Queue[str],
    subscribed_keys: set[str],
) -> None:
    key = _key_from_subscribe_msg(msg)
    if key is None:
        await websocket.send_json({"event": "error", "message": "Missing job_id or batch_id"})
        return
    notification_service.subscribe(key, event_queue)
    subscribed_keys.add(key)
    await websocket.send_json({"event": "subscribed", "key": key})


async def _handle_subscribe_tenant(
    websocket: WebSocket,
    tenant_id: str,
    notification_service: Any,
    event_queue: asyncio.Queue[str],
    subscribed_keys: set[str],
) -> None:
    key = f"tenant:{tenant_id}"
    notification_service.subscribe(key, event_queue)
    subscribed_keys.add(key)
    await websocket.send_json({"event": "subscribed", "key": key})


async def _handle_unsubscribe(
    websocket: WebSocket,
    msg: dict[str, Any],
    notification_service: Any,
    event_queue: asyncio.Queue[str],
    subscribed_keys: set[str],
) -> None:
    key = _key_from_subscribe_msg(msg)
    if key is None:
        return
    notification_service.unsubscribe(key, event_queue)
    subscribed_keys.discard(key)
    await websocket.send_json({"event": "unsubscribed", "key": key})


async def _handle_ping(websocket: WebSocket) -> None:
    await websocket.send_json({"event": "pong", "timestamp": datetime.now(UTC).isoformat()})


async def _dispatch_message(
    websocket: WebSocket,
    raw: str,
    *,
    tenant_id: str,
    notification_service: Any,
    event_queue: asyncio.Queue[str],
    subscribed_keys: set[str],
) -> None:
    """Parse a single incoming WebSocket frame and dispatch it.

    Errors decoding or routing the frame are reported back to the client as a
    structured error event; the calling loop continues to the next frame.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json({"event": "error", "message": "Invalid JSON"})
        return

    # ⛔ VALID JSON IS NOT NECESSARILY AN OBJECT, and this function's contract
    # (docstring above) is that a bad frame is answered with an error event and
    # the loop CONTINUES. ``json.loads`` accepts every JSON value, so ``[]``,
    # ``"hi"``, ``3`` and ``null`` all parse cleanly and then reach
    # ``msg.get(...)``, which raises ``AttributeError`` on each of them.
    #
    # ``AttributeError`` is not in the receive loop's
    # ``(OSError, ConnectionError, TimeoutError, RuntimeError)``, so it escaped
    # ``websocket_job_notifications`` entirely: one ``[]`` frame from any
    # authenticated client KILLED THE SOCKET — the opposite of the documented
    # behaviour, and a trivially reachable self-inflicted disconnect. (Cleanup
    # still ran via the handler's ``finally``, so nothing leaked; the connection
    # simply died.)
    #
    # Handled HERE rather than by widening that tuple: a malformed frame is a
    # client error this function already knows how to answer, and catching
    # ``AttributeError`` at the loop would also swallow genuine bugs.
    if not isinstance(msg, dict):
        await websocket.send_json({"event": "error", "message": "Message must be a JSON object"})
        return

    action = msg.get("action")
    if action == "ping":
        await _handle_ping(websocket)
    elif action == "subscribe":
        await _handle_subscribe(websocket, msg, notification_service, event_queue, subscribed_keys)
    elif action == "subscribe_tenant":
        await _handle_subscribe_tenant(
            websocket, tenant_id, notification_service, event_queue, subscribed_keys
        )
    elif action == "unsubscribe":
        await _handle_unsubscribe(
            websocket, msg, notification_service, event_queue, subscribed_keys
        )
    else:
        await websocket.send_json({"event": "error", "message": f"Unknown action: {action}"})


async def _accept_or_reject(
    websocket: WebSocket,
    notification_service: Any,
    connection_manager: Any,
    tenant_id: str,
) -> bool:
    """Accept the connection; enforce per-tenant connection limits.

    Returns True iff the connection was accepted and the caller should proceed
    with the receive loop.
    """
    if notification_service is None:
        await websocket.close(code=4500, reason="Notification service unavailable")
        return False

    await websocket.accept()
    logger.debug("WebSocket connection accepted")

    if connection_manager and not connection_manager.connect(tenant_id, websocket):
        logger.warning("WebSocket connection limit exceeded for tenant=%s", tenant_id)
        await websocket.send_json({"event": "error", "message": "Connection limit exceeded"})
        await websocket.close(code=4008, reason="Connection limit exceeded")
        return False
    return True


async def _cleanup(
    *,
    sender: asyncio.Task[None],
    heartbeat: asyncio.Task[None],
    notification_service: Any,
    event_queue: asyncio.Queue[str],
    connection_manager: Any,
    tenant_id: str,
    websocket: WebSocket,
) -> None:
    """Cancel background tasks and release notification/connection resources.

    Awaiting the cancelled tasks here is REQUIRED — without ``asyncio.gather``
    the cancelled tasks can linger and leak memory under WebSocket churn.
    """
    sender.cancel()
    heartbeat.cancel()
    await asyncio.gather(sender, heartbeat, return_exceptions=True)
    try:
        notification_service.unsubscribe_all(event_queue)
    except OSError, ConnectionError, TimeoutError, RuntimeError:
        logger.warning("Failed to unsubscribe from notifications", exc_info=True)
    if connection_manager:
        connection_manager.disconnect(tenant_id, websocket)


@router.websocket("/ws", name="tts_job_notifications")
async def websocket_job_notifications(websocket: WebSocket) -> None:
    """WebSocket for real-time job status updates. **Server-side clients only.**

    Authentication is REQUIRED: the socket must carry
    ``X-Tenant-Id: Bearer <license-token>`` on the handshake (or arrive with
    ``state.tenant_id`` already set by an ASGI authenticator). Anything else is
    rejected before ``accept``. A ``?tenant_id=`` query parameter is ignored
    entirely — it used to select the subscription channel.

    ENTITLEMENT is enforced too, not just identity: once the tenant is known it
    is stamped on ``websocket.state`` and put through ``require_license_ws``,
    the WebSocket twin of the ``LICENSE_GATE`` the HTTP routers carry. A tenant
    whose w2v licence is none/expired/revoked is closed with 1008, and an
    unverifiable one (licences upstream down) with 1013 — never allowed through.
    Dormant unless ``W2V_LICENSE_GATE_ENABLED`` is true.

    **Why a handshake header is an acceptable sole credential.** The browser
    ``WebSocket`` constructor cannot set request headers, so header auth makes
    this endpoint unreachable from a browser by construction. That is a
    deliberate design constraint, not an oversight, because there is no browser
    consumer to break — verified by search, not assumed:

    * ``w2v_frontend`` contains no ``new WebSocket(`` against this service. Its
      only socket is the pipeline/job-progress stream, which targets the
      **orchestrator** (``server.js`` proxies the upgrade to w2v-orchestrator;
      ``svelte.config.js`` CSP ``connect-src`` allows exactly one origin,
      ``wss://w2v.nonagentic.ai``). Its ``StreamMode`` union in
      ``src/lib/stores/job-progress/types.ts`` lists ``'ws'``, but the shipped
      transports are ``sse-transport.svelte.ts`` and
      ``polling-transport.svelte.ts`` — there is no ws-transport module.
    * A repo-wide search for ``api/tts/ws`` across the w2v tree matches only
      this service's own source/tests/docs (``README.md``, ``CLAUDE.md``,
      ``src/asgi.py``, ``src/api/schemas.py``, ``docs/capabilities/tts.md``) —
      no caller in any other service.

    Consequently NO browser-usable channel is offered: no ``Sec-WebSocket-
    Protocol`` token subprotocol and no ticket endpoint, and deliberately never
    a query-string token (it would land in access logs and proxy history). If a
    browser consumer is ever genuinely needed, add a short-lived single-use
    ticket minted by an authenticated REST call — do not relax this to a query
    parameter.

    **Rejection is observed as an HTTP 403 handshake failure**, not as a 1008
    close frame; see :data:`WS_POLICY_VIOLATION` for the measurement. Clients
    must detect the failed handshake, not a close code.

    Client messages:
        {"action": "subscribe", "job_id": "..."}
        {"action": "subscribe", "batch_id": "..."}
        {"action": "subscribe_tenant"}
        {"action": "unsubscribe", "job_id": "..."}
        {"action": "ping"}

    Server messages:
        {"event": "job_started"|"job_completed"|"job_failed"|"batch_progress", ...}
        {"event": "pong", "timestamp": "..."}
        {"event": "subscribed", "key": "..."}
        {"event": "error", "message": "..."}
    """
    # Authenticate BEFORE anything else: an unauthenticated socket is never
    # accepted and never learns whether the notification service is up.
    #
    # Closing pre-``accept`` means the client sees a 403 handshake failure, NOT
    # the 1008 (see WS_POLICY_VIOLATION). That is the deliberate trade: rejecting
    # at the handshake leaks nothing and allocates nothing, whereas accepting
    # first — purely to deliver a nicer close code — would hand an unauthenticated
    # caller an open connection. A 403 is a perfectly legible signal.
    tenant_id, subject_id = await _resolve_identity(websocket)
    if tenant_id is None:
        logger.warning("WebSocket rejected: no authenticated tenant on handshake")
        await websocket.close(code=WS_POLICY_VIOLATION, reason="Unauthenticated")
        return

    # ENTITLEMENT. Identity ("who are you") is settled above; this asks the
    # SECOND question ("may you"), which this socket never asked before. The
    # HTTP routers have carried ``LICENSE_GATE`` since #113, so with the flag
    # armed an expired or revoked tenant was refused on /api/tts and SERVED on
    # this socket — the regression class w2v_common 0.76.0 added
    # ``require_license_ws`` for.
    #
    # ⚠️ WHY THIS IS CALLED HERE AND NOT MOUNTED AS ``LICENSE_GATE_WS`` ON THE
    # ROUTE. A route dependency runs BEFORE the handler, and on a websocket
    # scope NOTHING has stamped an identity by then: every middleware in the
    # fleet auth stack is a ``BaseHTTPMiddleware``, whose ``__call__`` returns
    # early for a non-``http`` scope, so ``dispatch`` never runs and
    # ``websocket.state.tenant_id`` is empty (the same reason
    # ``_handshake_license_token`` has to read the headers itself). A gate
    # mounted there would resolve NO tenant, derive ``none``, and close EVERY
    # socket — including an entitled one. The tenant is first known one line
    # above, so the gate is called one line below it, on the state we stamp
    # ourselves from the introspected licence. Stamping is what makes the
    # dependency's own ``websocket.state`` precedence resolve to a REAL,
    # verified tenant rather than the raw ``x-tenant-id`` header value, which
    # would be a spoofable lane the HTTP gate does not have.
    #
    # It is a no-op while ``W2V_LICENSE_GATE_ENABLED`` is unset, and it refuses
    # by raising ``WebSocketException`` (1008 licence / 1013 upstream
    # unavailable), which starlette turns into a close. Like the reject above,
    # that happens pre-``accept``, so the wire signal is a 403 handshake
    # failure — see :data:`WS_POLICY_VIOLATION`.
    websocket.state.tenant_id = tenant_id
    websocket.state.subject_id = subject_id
    await require_license_ws(websocket)

    app = websocket.app
    notification_service = getattr(app.state, "notification_service", None)
    connection_manager = getattr(app.state, "connection_manager", None)

    if not await _accept_or_reject(websocket, notification_service, connection_manager, tenant_id):
        return

    queue_size, heartbeat_interval = _get_ws_settings(app)
    event_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
    subscribed_keys: set[str] = set()

    sender = asyncio.create_task(_sender_loop(websocket, event_queue))
    heartbeat = asyncio.create_task(_heartbeat_loop(websocket, heartbeat_interval))

    try:
        while True:
            data = await websocket.receive_text()
            await _dispatch_message(
                websocket,
                data,
                tenant_id=tenant_id,
                notification_service=notification_service,
                event_queue=event_queue,
                subscribed_keys=subscribed_keys,
            )
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected (tenant=%s)", tenant_id)
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        await _cleanup(
            sender=sender,
            heartbeat=heartbeat,
            notification_service=notification_service,
            event_queue=event_queue,
            connection_manager=connection_manager,
            tenant_id=tenant_id,
            websocket=websocket,
        )
