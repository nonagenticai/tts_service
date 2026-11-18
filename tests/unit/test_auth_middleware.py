"""Unit tests for the auth middleware."""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from w2v_common.tenant import MissingTenantError

from src.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    AuthMiddleware,
    _introspect_license,
    _is_trusted_network,
)
from src.auth.models import User
from tests.conftest import TENANT_A

# ============================================================================
# User model tests
# ============================================================================


class TestUser:
    def test_user_creation(self):
        user = User(sub="user-1", preferred_username="alice", tenant_id=uuid.UUID(TENANT_A))
        assert user.sub == "user-1"
        assert user.preferred_username == "alice"
        assert user.tenant_id == uuid.UUID(TENANT_A)

    def test_effective_tenant_id_with_tenant(self):
        user = User(sub="user-1", tenant_id=uuid.UUID(TENANT_A))
        assert user.effective_tenant_id == uuid.UUID(TENANT_A)

    def test_effective_tenant_id_without_tenant(self):
        """A tenant-less User must REFUSE, not self-name a tenant from ``sub``.

        This used to assert ``effective_tenant_id == "user-1"``. That fallback
        minted a tenant that was never registered; w2v_common 0.67.0 deleted it
        and raises instead. Asserting the refusal is the contract now.
        """
        user = User(sub="user-1", tenant_id=None)
        with pytest.raises(MissingTenantError):
            _ = user.effective_tenant_id

    def test_a_non_uuid_tenant_is_rejected_at_construction(self):
        """A legacy slug tenant must not survive as a plausible-looking string."""
        with pytest.raises(ValidationError):
            User.model_validate({"sub": "user-1", "tenant_id": "tenant-1"})


# ============================================================================
# _is_trusted_network
# ============================================================================


class TestTrustDefaultIsEmpty:
    """w2v-common >= 0.89.0: with TRUSTED_CIDRS unset, NO peer is trusted.

    Every other trusted-network test here opts in via the
    ``explicit_trusted_cidrs`` fixture; this one pins the default itself, so a
    regression back to an implicit loopback/private-range default goes red.
    """

    def test_unset_env_trusts_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        import w2v_common.auth.middleware as auth_middleware

        saved = os.environ.get("TRUSTED_CIDRS")
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
        try:
            mod = importlib.reload(auth_middleware)
            assert mod.TRUSTED_NETWORKS == []
            for host in ("127.0.0.1", "::1", "10.0.0.1", "10.42.5.10", "172.16.0.1", "192.168.1.1"):
                assert mod._is_trusted_network(host) is False, host
        finally:
            if saved is not None:
                os.environ["TRUSTED_CIDRS"] = saved
            importlib.reload(auth_middleware)


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestIsTrustedNetwork:
    def test_localhost_v4_trusted(self):
        assert _is_trusted_network("127.0.0.1") is True

    def test_localhost_v6_trusted(self):
        assert _is_trusted_network("::1") is True

    def test_private_192_168_not_trusted_by_default(self):
        # 192.168.0.0/16 was deliberately removed from the default TRUSTED_CIDRS
        # in w2v_common to fix the SSR-proxy source-IP bypass (see
        # project_orchestrator_trusted_networks_bypass_2026-04-18). Cluster pod
        # IPs and node IPs live in 10.0.0.0/8 / 172.16.0.0/12, not 192.168/16.
        assert _is_trusted_network("192.168.1.100") is False

    def test_private_10_trusted(self):
        assert _is_trusted_network("10.0.0.1") is True

    def test_private_172_trusted(self):
        assert _is_trusted_network("172.16.0.1") is True

    def test_public_ip_not_trusted(self):
        assert _is_trusted_network("8.8.8.8") is False

    def test_none_not_trusted(self):
        assert _is_trusted_network(None) is False

    def test_empty_string_not_trusted(self):
        assert _is_trusted_network("") is False

    def test_invalid_ip_not_trusted(self):
        assert _is_trusted_network("not-an-ip") is False


# ============================================================================
# Public endpoints configuration
# ============================================================================


class TestPublicEndpoints:
    def test_health_is_public(self):
        assert "/health" in PUBLIC_ENDPOINTS

    def test_health_ready_is_public(self):
        assert "/health/ready" in PUBLIC_ENDPOINTS

    def test_health_live_is_public(self):
        assert "/health/live" in PUBLIC_ENDPOINTS

    def test_docs_is_public(self):
        assert "/docs" in PUBLIC_ENDPOINTS

    def test_public_prefixes(self):
        assert "/docs" in PUBLIC_PREFIXES
        assert "/health/" in PUBLIC_PREFIXES


# ============================================================================
# _introspect_license
# ============================================================================


class TestIntrospectLicense:
    @pytest.mark.asyncio
    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_introspect_active_license(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"active": True, "tenant_id": TENANT_A}
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        # Clear cache before test
        from w2v_common.auth.middleware import _get_license_cache

        _get_license_cache().clear()

        result = await _introspect_license("valid-token")
        assert result is not None
        assert result["active"] is True

    @pytest.mark.asyncio
    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_introspect_failed_request(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection error"))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        from w2v_common.auth.middleware import _get_license_cache

        _get_license_cache().clear()

        result = await _introspect_license("bad-token-unique-1234")
        assert result is None


# ============================================================================
# AuthMiddleware
# ============================================================================


class TestAuthMiddleware:
    def _make_middleware(self):
        app = MagicMock()
        return AuthMiddleware(app)

    def _make_request(self, path="/api/tts/generate", headers=None, client_host="127.0.0.1"):
        request = MagicMock()
        request.url.path = path
        request.headers = headers or {}

        class MockClient:
            host = client_host

        request.client = MockClient()
        request.state = MagicMock(spec=[])
        return request

    @pytest.mark.asyncio
    async def test_public_endpoint_passes_through(self):
        middleware = self._make_middleware()
        request = self._make_request(path="/health")
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_public_endpoint_trailing_slash(self):
        middleware = self._make_middleware()
        request = self._make_request(path="/health/")
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_public_prefix_passes_through(self):
        middleware = self._make_middleware()
        request = self._make_request(path="/docs/something")
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("explicit_trusted_cidrs")
    async def test_trusted_network_no_token_returns_401(self):
        """w2v_common 0.6.9 flips AUTH_TRUST_INCLUSTER_IDENTITY default to false.

        Trusted-network identity inference is fail-closed by default. A request
        from a trusted CIDR with neither a license token nor an X-Tenant-Id
        header must now return 401 instead of falling through.
        """
        middleware = self._make_middleware()
        request = self._make_request(
            path="/api/tts/generate",
            client_host="127.0.0.1",
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        call_next.assert_not_called()
        assert response.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("explicit_trusted_cidrs")
    async def test_trusted_network_with_plain_x_tenant_id_returns_401(self):
        """Bare `X-Tenant-Id: <id>` (no Bearer) is no longer a valid auth path.

        With AUTH_TRUST_INCLUSTER_IDENTITY=false (w2v_common 0.6.9 default),
        only `X-Tenant-Id: Bearer <license-token>` authenticates a request.
        """
        middleware = self._make_middleware()
        request = self._make_request(
            path="/api/tts/generate",
            headers={"X-Tenant-Id": "custom-tenant"},
            client_host="127.0.0.1",
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        call_next.assert_not_called()
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_untrusted_network_no_token_returns_401(self):
        middleware = self._make_middleware()
        request = self._make_request(
            path="/api/tts/generate",
            client_host="8.8.8.8",
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        call_next.assert_not_called()
        assert response.status_code == 401

    def test_extract_license_token_reads_an_opaque_authorization_bearer(self):
        """An OPAQUE Authorization Bearer IS a license token source.

        Changed in w2v_common 0.74.0, which adopted the platform licensing
        convention: a bearer that is not JWT-shaped is an opaque licence token.
        Before 0.74.0 Authorization was never read here (FR-401, 0.6.5).
        """
        middleware = self._make_middleware()
        request = self._make_request(headers={"Authorization": "Bearer my-token-123"})
        assert middleware._extract_license_token(request) == "my-token-123"

    def test_extract_license_token_still_ignores_a_jwt_shaped_authorization(self):
        """A JWT-shaped bearer is NEVER a license token — the surviving half of FR-401.

        Authorization stays reserved for upstream JWT auth via
        `_extract_jwt_bearer` whenever the bearer looks like a JWT. Regressing
        this reproduces the 2026-04-26 prod 401s on /api/assets/*.
        """
        middleware = self._make_middleware()
        request = self._make_request(headers={"Authorization": "Bearer header.payload.signature"})
        assert middleware._extract_license_token(request) is None

    def test_extract_jwt_bearer_from_authorization(self):
        """Authorization Bearer feeds JWT validation only, never license introspection."""
        middleware = self._make_middleware()
        request = self._make_request(headers={"Authorization": "Bearer my-token-123"})
        assert middleware._extract_jwt_bearer(request) == "my-token-123"

    def test_extract_license_token_from_x_tenant_id_bearer(self):
        middleware = self._make_middleware()
        request = self._make_request(headers={"X-Tenant-Id": "Bearer my-token-456"})
        token = middleware._extract_license_token(request)
        assert token == "my-token-456"

    def test_extract_license_token_no_token(self):
        middleware = self._make_middleware()
        request = self._make_request(headers={})
        token = middleware._extract_license_token(request)
        assert token is None

    def test_extract_license_token_plain_x_tenant_id(self):
        middleware = self._make_middleware()
        request = self._make_request(headers={"X-Tenant-Id": "plain-tenant-id"})
        token = middleware._extract_license_token(request)
        assert token is None

    @pytest.mark.asyncio
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_valid_license_token(self, mock_introspect):
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": TENANT_A,
            "subject_id": "user-1",
        }
        middleware = self._make_middleware()
        request = self._make_request(
            path="/api/tts/generate",
            headers={"X-Tenant-Id": "Bearer valid-token"},
            client_host="8.8.8.8",
        )
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)
        call_next.assert_called_once()

    @pytest.mark.asyncio
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_invalid_license_token(self, mock_introspect):
        mock_introspect.return_value = {"active": False, "reason": "expired"}
        middleware = self._make_middleware()
        request = self._make_request(
            path="/api/tts/generate",
            headers={"X-Tenant-Id": "Bearer expired-token"},
            client_host="8.8.8.8",
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        call_next.assert_not_called()
        assert response.status_code == 401

    @pytest.mark.asyncio
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_introspect_returns_none(self, mock_introspect):
        mock_introspect.return_value = None
        middleware = self._make_middleware()
        request = self._make_request(
            path="/api/tts/generate",
            headers={"X-Tenant-Id": "Bearer unknown-token"},
            client_host="8.8.8.8",
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        call_next.assert_not_called()
        assert response.status_code == 401
