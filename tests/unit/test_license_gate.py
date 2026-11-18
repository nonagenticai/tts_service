"""The entitlement gate on /api/tts and /api/tts/usage actually refuses an
unlicensed tenant.

Until w2v_common 0.75.0 this service authenticated the licence token for
tenant IDENTITY and stopped there: ``AuthMiddleware`` (via
``src.auth.middleware``) answers "who are you", never "may you". A tenant
whose licence was ``expired`` or ``revoked`` still presented a syntactically
valid token, so every route under ``/api/tts`` (generation, jobs, streaming,
usage/cost/quota analytics) served them — billing-bearing provider calls and
all. ``LICENSE_GATE`` is mounted router-wide on both ``src.api.tts.router``
and ``src.api.usage.router`` to close that.

Every test here drives a REAL HTTP request through the REAL routers, because
the interesting question is not whether ``require_license`` works —
w2v_common owns that — but whether THIS service's routers actually carry it.
This FastAPI build includes sub-routers lazily (``_IncludedRouter``), so a
parent router's ``dependencies=`` are applied when a request is MATCHED, not
baked into the route objects at import time: static introspection of
``route.dependant.dependencies`` would report the routes as ungated and be a
FALSE NEGATIVE (this is exactly the trap a sibling service paid for). A
request is the only honest probe.

The paired 200s (entitled, flag-off) are what stop this file from passing
vacuously: if the gate were mounted wrongly (or the fixture broken) those
cases would not be 200s either.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.health import router as health_router
from src.api.tts import router as tts_router
from src.api.usage import router as usage_router
from src.providers.registry import ProviderRegistry
from tests.conftest import tenant_uuid

TENANT = tenant_uuid("license-gate-tenant")
SUBJECT = "user-under-test"

#: The cheapest genuinely gated route on ``tts_router``: it takes no tenant
#: dependency and no DB session, and with an EMPTY provider registry its only
#: real work is an empty loop. So a non-200 here can only be the gate.
GATED_TTS_PATH = "/api/tts/providers"

#: Same property, on the SEPARATE ``usage_router`` — proves that router
#: carries the gate too, not just ``tts_router``.
GATED_USAGE_PATH = "/api/tts/usage/rates"

#: A probe. Must answer 200 for a revoked tenant: a 402/503 on a liveness
#: probe CrashLoops the pod, and an unlicensed tenant cannot be billed by
#: killing our own deployment.
PROBE_PATH = "/health/live"


# ---------------------------------------------------------------------------
# Fakes
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
    ``error`` models an upstream failure, which must 503, never allow.
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


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way ``AuthMiddleware`` stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TENANT
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(health_router)
    application.include_router(tts_router)
    application.include_router(usage_router)

    # Both GATED_* handlers read app.state.provider_registry; an empty real
    # registry makes their own work free (an empty loop) so only the gate can
    # fail the request. No DB, no redis, no settings needed for either route.
    application.state.provider_registry = ProviderRegistry()
    return application


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused (tts_router)
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
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(app, GATED_TTS_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


@pytest.mark.asyncio
async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(app, GATED_TTS_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(app, GATED_TTS_PATH)

    assert response.status_code == 200, response.text
    assert response.json() == []


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_TTS_PATH)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_TTS_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# usage_router carries the SAME gate (separate router object from tts_router)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_router_is_also_gated(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_USAGE_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "license_revoked"


@pytest.mark.asyncio
async def test_usage_router_serves_an_entitled_tenant(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(app, GATED_USAGE_PATH)

    assert response.status_code == 200, response.text
    assert response.json() == []


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_is_served_for_a_revoked_tenant(app, monkeypatch):
    """A gated probe path CrashLoops the pod. /health/live must stay 200."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, PROBE_PATH)

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_TTS_PATH)
    assert not is_license_exempt_path(GATED_USAGE_PATH)
