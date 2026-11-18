"""Request-id correlation is now the shared middleware + structlog processor.

The bespoke ``src.middleware.request_id.RequestIdMiddleware`` (which bound
structlog contextvars itself) is gone. request_id now flows from the w2v
request-id ContextVar — set by the factory's shared ``RequestIdMiddleware`` —
into structlog logs via ``w2v_common.observability.structlog_bridge.add_request_id``,
wired into the chain by ``src.logging_config.configure_logging``. These tests
lock (a) the app echoes ``X-Request-Id`` and (b) the processor injects it into
structlog output with nothing bound in structlog contextvars.
"""

from __future__ import annotations

import structlog
from fastapi.testclient import TestClient
from w2v_common.observability.request_context import clear_request_id, set_request_id
from w2v_common.observability.structlog_bridge import add_request_id

from src.asgi import app
from src.config import get_settings
from src.logging_config import configure_logging


def test_app_echoes_request_id_header() -> None:
    """The shared RequestIdMiddleware (via create_app) preserves an inbound id."""
    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-Request-Id": "trace-abc-123"})
    assert resp.headers.get("X-Request-Id") == "trace-abc-123"


def test_app_generates_request_id_when_missing() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    rid = resp.headers.get("X-Request-Id")
    assert rid
    assert len(rid) >= 32


def test_processor_injects_request_id_into_structlog_output() -> None:
    """With nothing bound in structlog contextvars, request_id still reaches the
    log purely via the ContextVar the shared middleware sets."""
    settings = get_settings()
    configure_logging(settings)
    set_request_id("rid-structlog-xyz")
    try:
        cap = structlog.testing.CapturingLogger()
        log = structlog.wrap_logger(
            cap,
            processors=[
                structlog.contextvars.merge_contextvars,
                add_request_id,
                structlog.processors.KeyValueRenderer(key_order=["event", "request_id"]),
            ],
        )
        log.info("served")
    finally:
        clear_request_id()
    assert cap.calls
    assert "request_id='rid-structlog-xyz'" in cap.calls[0].args[0]
