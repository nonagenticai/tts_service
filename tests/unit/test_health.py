"""Unit tests for health check endpoints."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.exceptions
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import OperationalError

from src.api.health import _check_storage, health_check, health_live, health_ready, readyz


class TestHealthLive:
    @pytest.mark.asyncio
    async def test_liveness_returns_ok(self):
        result = await health_live()
        assert result == {"status": "ok"}


class TestHealthCheck:
    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_healthy_status(self, mock_get_engine: MagicMock, mock_get_redis: MagicMock):
        # Mock database
        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_get_engine.return_value = mock_engine

        # Mock redis
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_redis

        raw = await health_check()
        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        assert result["status"] == "healthy"
        components_raw = result["components"]
        assert isinstance(components_raw, dict)
        components: dict[str, Any] = components_raw
        assert components["api"] == "healthy"

    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_degraded_when_db_fails(
        self, mock_get_engine: MagicMock, mock_get_redis: MagicMock
    ):
        mock_get_engine.side_effect = OSError("db connection failed")

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_redis

        raw = await health_check()
        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        assert result["status"] == "degraded"
        components_raw = result["components"]
        assert isinstance(components_raw, dict)
        components: dict[str, Any] = components_raw
        assert "unhealthy" in components["database"]

    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_degraded_when_redis_fails(
        self, mock_get_engine: MagicMock, mock_get_redis: MagicMock
    ):
        # Mock database as healthy
        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_get_engine.return_value = mock_engine

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("redis error"))
        mock_get_redis.return_value = mock_redis

        raw = await health_check()
        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_redis_not_configured(
        self, mock_get_engine: MagicMock, mock_get_redis: MagicMock
    ):
        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_get_engine.return_value = mock_engine

        mock_get_redis.return_value = None

        raw = await health_check()
        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        components_raw = result["components"]
        assert isinstance(components_raw, dict)
        components: dict[str, Any] = components_raw
        assert components["redis"] == "not configured"


class TestCheckStorage:
    """Regression tests for ``_check_storage`` — catches the exact class of
    deployment bug where STORAGE_LOCAL_PATH points at a read-only parent."""

    def test_writable_path_returns_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
        from src.config import clear_settings_cache

        clear_settings_cache()
        assert _check_storage() == "healthy"

    def test_missing_path_returns_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(missing))
        from src.config import clear_settings_cache

        clear_settings_cache()
        assert _check_storage() == "path not found"

    def test_unwritable_path_returns_not_writable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real root cause of the app_inspector /health bug, generalized.

        Strips write permission from a real directory and asserts the
        check reports ``not writable``. The deployment analogue is a
        readOnlyRootFilesystem container mounting the PVC at a different
        path than ``STORAGE_LOCAL_PATH``.
        """
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o555)
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(locked))
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            assert _check_storage() == "not writable"
        finally:
            locked.chmod(0o755)  # allow tmp_path cleanup

    @pytest.mark.asyncio
    async def test_health_endpoint_reports_storage_component(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration: /health must expose storage under components."""
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
        from src.config import clear_settings_cache

        clear_settings_cache()

        with (
            patch("src.api.health.get_engine") as mock_get_engine,
            patch("src.api.health.get_redis_client") as mock_get_redis,
        ):
            mock_conn = AsyncMock()
            mock_engine = AsyncMock()
            mock_engine.connect = MagicMock(return_value=mock_conn)
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            mock_get_engine.return_value = mock_engine

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_get_redis.return_value = mock_redis

            raw = await health_check()

        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        components = cast(dict[str, Any], result["components"])
        assert components["storage"] == "healthy"
        assert result["status"] == "healthy"


class TestHealthReady:
    @pytest.mark.asyncio
    @patch("src.api.health.get_engine")
    async def test_ready_success(self, mock_get_engine: MagicMock):
        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_get_engine.return_value = mock_engine

        raw = await health_ready()
        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        assert result["ready"] is True
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    @patch("src.api.health.get_engine")
    async def test_ready_failure(self, mock_get_engine: MagicMock):
        mock_get_engine.side_effect = OSError("db error")

        result = await health_ready()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


class _IntModel(BaseModel):
    """Minimal model used only to mint a real pydantic ``ValidationError``."""

    value: int


def _db_operational_error() -> OperationalError:
    """The exact exception SQLAlchemy raises when psycopg3 loses the server."""
    return OperationalError(
        "SELECT 1",
        {},
        Exception("server closed the connection unexpectedly"),
    )


def _engine(execute_error: BaseException | None = None) -> AsyncMock:
    """An engine whose ``connect()`` context manager optionally fails on execute."""
    mock_conn = AsyncMock()
    mock_engine = AsyncMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(side_effect=execute_error)
    return mock_engine


def _healthy_redis() -> AsyncMock:
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    return mock_redis


def _body(response: JSONResponse) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(cast(bytes, response.body).decode("utf-8")))


class TestProbeExceptionHierarchy:
    """A DB/Redis fault must be a 503 (dependency down), never a 500.

    ``sqlalchemy.exc.OperationalError`` — into which psycopg3 wraps a lost
    server — is ``DatabaseError -> DBAPIError -> StatementError ->
    SQLAlchemyError -> Exception``. NOTHING in that chain is ``OSError`` /
    ``ConnectionError`` / ``TimeoutError`` / ``RuntimeError``, so the old
    builtins-only except tuples let it escape and FastAPI answered 500. On
    ``/health/ready`` (and its ``/readyz`` alias) that breaks the Kubernetes
    readiness contract: 500 means "the app is broken", 503 means "not ready".
    Same premise for ``redis.exceptions.RedisError`` and pydantic's
    ``ValidationError`` (a ``ValueError``).

    One test per probe site changed in ``src/api/health.py``.
    """

    def test_probe_exceptions_are_not_the_builtins_they_shadow(self) -> None:
        """Canary: if these ever get re-parented, this file's premise moves."""
        builtin_tuple = (OSError, ConnectionError, TimeoutError, RuntimeError)
        assert not issubclass(OperationalError, builtin_tuple)
        assert not issubclass(redis.exceptions.RedisError, builtin_tuple)
        assert not issubclass(ValidationError, builtin_tuple)

    # -- probe: health_ready() database (also the /readyz alias) -------------

    @pytest.mark.asyncio
    @patch("src.api.health.get_engine")
    async def test_ready_returns_503_on_sqlalchemy_error(self, mock_get_engine: MagicMock) -> None:
        mock_get_engine.side_effect = _db_operational_error()

        result = await health_ready()

        assert isinstance(result, JSONResponse), "DB fault escaped the handler -> 500"
        assert result.status_code == 503
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["ready"] is False
        checks = cast(dict[str, Any], body["checks"])
        database = cast(dict[str, Any], checks["database"])
        assert database["status"] == "down"
        assert "server closed the connection unexpectedly" in database["error"]

    @pytest.mark.asyncio
    @patch("src.api.health.get_engine")
    async def test_ready_returns_503_when_execute_raises(self, mock_get_engine: MagicMock) -> None:
        """The fault surfaces on ``conn.execute``, not on ``get_engine()``."""
        mock_get_engine.return_value = _engine(_db_operational_error())

        result = await health_ready()

        assert isinstance(result, JSONResponse), "DB fault escaped the handler -> 500"
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("src.api.health.get_engine")
    async def test_readyz_alias_returns_503_on_sqlalchemy_error(
        self, mock_get_engine: MagicMock
    ) -> None:
        mock_get_engine.side_effect = _db_operational_error()

        result = await readyz()

        assert isinstance(result, JSONResponse), "DB fault escaped the handler -> 500"
        assert result.status_code == 503
        assert _body(result)["ready"] is False

    # -- probe: health_check() database --------------------------------------

    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_health_degraded_on_sqlalchemy_error(
        self, mock_get_engine: MagicMock, mock_get_redis: MagicMock
    ) -> None:
        mock_get_engine.side_effect = _db_operational_error()
        mock_get_redis.return_value = _healthy_redis()

        raw = await health_check()

        assert isinstance(raw, dict), "DB fault escaped the handler -> 500"
        result = cast(dict[str, Any], raw)
        assert result["status"] == "degraded"
        components = cast(dict[str, Any], result["components"])
        assert "unhealthy" in components["database"]
        assert components["api"] == "healthy"

    # -- probe: health_check() redis -----------------------------------------

    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_health_degraded_on_redis_error(
        self, mock_get_engine: MagicMock, mock_get_redis: MagicMock
    ) -> None:
        mock_get_engine.return_value = _engine()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(
            side_effect=redis.exceptions.ConnectionError(
                "Error 111 connecting to redis:6379. Connection refused."
            )
        )
        mock_get_redis.return_value = mock_redis

        raw = await health_check()

        assert isinstance(raw, dict), "Redis fault escaped the handler -> 500"
        result = cast(dict[str, Any], raw)
        assert result["status"] == "degraded"
        components = cast(dict[str, Any], result["components"])
        assert "unhealthy" in components["redis"]

    @pytest.mark.asyncio
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_health_degraded_when_ping_returns_false(
        self, mock_get_engine: MagicMock, mock_get_redis: MagicMock
    ) -> None:
        """A DEAD Redis reports ``ping() -> False``, it does not raise.

        ``w2v_common``'s ``RedisClient.ping()`` is fail-soft: it catches
        ``RuntimeError``/``RedisError`` and returns ``False``. So the probe
        above cannot learn about a dead backend from an exception — discarding
        the bool reported a totally dead Redis as ``"healthy"`` and the overall
        status as ``"healthy"``. This is the realistic production shape of the
        outage; the raising variant above only happens with a substituted
        client.
        """
        mock_get_engine.return_value = _engine()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=False)
        mock_get_redis.return_value = mock_redis

        raw = await health_check()

        assert isinstance(raw, dict)
        result = cast(dict[str, Any], raw)
        components = cast(dict[str, Any], result["components"])
        assert components["redis"].startswith("unhealthy"), (
            f"dead Redis reported as {components['redis']!r}"
        )
        assert result["status"] == "degraded"

    # -- probe: health_check() outer safety net ------------------------------

    @pytest.mark.asyncio
    @patch("src.api.health._check_storage")
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_health_returns_503_when_outer_net_sees_library_error(
        self,
        mock_get_engine: MagicMock,
        mock_get_redis: MagicMock,
        mock_check_storage: MagicMock,
    ) -> None:
        mock_get_engine.return_value = _engine()
        mock_get_redis.return_value = _healthy_redis()
        mock_check_storage.side_effect = _db_operational_error()

        raw = await health_check()

        assert isinstance(raw, JSONResponse), "library error escaped the outer net -> 500"
        assert raw.status_code == 503
        assert _body(raw)["status"] == "unhealthy"

    # -- the HTTP contract, end to end ---------------------------------------

    @pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
    def test_readiness_http_status_is_503_not_500(self, path: str) -> None:
        """Through the real ASGI stack: the wire status must be 503, never 500."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.health import router

        app = FastAPI()
        app.include_router(router)

        with (
            patch("src.api.health.get_engine", side_effect=_db_operational_error()),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.get(path)

        assert response.status_code != 500, "DB fault surfaced as an app bug, not 'not ready'"
        assert response.status_code == 503
        assert response.json()["ready"] is False

    # -- probe: _check_storage() settings ------------------------------------

    def test_check_storage_reports_settings_validation_error(self) -> None:
        """``get_settings()`` raises pydantic ``ValidationError``, not ``OSError``."""
        try:
            _IntModel.model_validate({"value": "not-an-int"})
        except ValidationError as exc:
            validation_error = exc
        else:  # pragma: no cover - defensive
            pytest.fail("expected a ValidationError")

        with patch("src.api.health.get_settings", side_effect=validation_error):
            assert _check_storage().startswith("unhealthy: ")
