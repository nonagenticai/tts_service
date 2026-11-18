"""Integration tests for API endpoints (uses TestClient, no real DB)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.providers.base import GenerationResult, VoiceMetadata
from tests.conftest import tenant_uuid


class _AuthenticatedTenant:
    """Stamp an authenticated tenant onto the ASGI scope.

    ``AUTH_ENABLED=false`` in tests, so ``AuthMiddleware`` never populates
    ``request.state.tenant_id``. Routes now 401 without one (product-owner
    ruling: a tenant_id may never hold a sentinel), so the test client has to
    stand in for the authenticated caller instead of relying on a fallback.
    """

    def __init__(self, app, tenant_id: str = tenant_uuid("tenant-test")) -> None:
        self.app = app
        self.tenant_id = tenant_id

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["tenant_id"] = self.tenant_id
        await self.app(scope, receive, send)


@pytest.fixture
def mock_app():
    """Create app with mocked dependencies."""
    # Must import after env vars are set in conftest
    from src.providers.registry import ProviderRegistry

    # Create mock provider
    mock_provider = AsyncMock()
    mock_provider.name = "kokoro"
    mock_provider.display_name = "Kokoro-82M"
    mock_provider.provider_type = "local"
    mock_provider.max_text_length = 10000
    mock_provider.requires_api_key = False
    mock_provider.is_available = AsyncMock(return_value=True)
    mock_provider.get_default_voice = MagicMock(return_value="af_heart")
    mock_provider.list_voices = AsyncMock(
        return_value=[
            VoiceMetadata(id="af_heart", display_name="Heart"),
        ]
    )
    mock_provider.generate = AsyncMock(
        return_value=GenerationResult(
            audio_data=b"fake_mp3_data",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.5,
        )
    )

    registry = ProviderRegistry()
    registry.register(mock_provider)

    mock_storage = AsyncMock()
    mock_tts_service = MagicMock()
    mock_tts_service.generate_sync = AsyncMock(
        return_value=(
            GenerationResult(
                audio_data=b"fake_mp3_data",
                format="mp3",
                sample_rate=24000,
                duration_seconds=1.5,
            ),
            {
                "resolved_provider": "kokoro",
                "cached": False,
                "cache_hit": False,
                "latency_ms": 10.0,
                "estimated_cost": 0.0,
            },
        )
    )

    # Patch the app creation to inject mocks
    with patch(
        "src.database.connection.DATABASE_URL",
        "postgresql+psycopg://test:test@localhost/test",  # pragma: allowlist secret
    ):
        from src.asgi import app
        from src.database.session import get_db

        app.state.provider_registry = registry
        app.state.storage_service = mock_storage
        app.state.tts_service = mock_tts_service

        # Override get_db dependency to avoid real DB connection
        mock_db = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        # Disable lifespan for testing
        client = TestClient(_AuthenticatedTenant(app), raise_server_exceptions=False)
        yield client, mock_tts_service, registry

        app.dependency_overrides.clear()


class TestRootEndpoint:
    def test_root(self, mock_app):
        client, _, _ = mock_app
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TTS Service"
        assert data["status"] == "running"


class TestHealthEndpoints:
    def test_health_live(self, mock_app):
        client, _, _ = mock_app
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestProviderEndpoints:
    def test_list_providers(self, mock_app):
        client, _, _ = mock_app
        response = client.get("/api/tts/providers")
        assert response.status_code == 200
        providers = response.json()
        assert len(providers) >= 1
        assert providers[0]["name"] == "kokoro"

    def test_list_voices(self, mock_app):
        client, _, _ = mock_app
        response = client.get("/api/tts/providers/kokoro/voices")
        assert response.status_code == 200
        voices = response.json()
        assert len(voices) >= 1

    def test_list_voices_unknown_provider(self, mock_app):
        client, _, _ = mock_app
        response = client.get("/api/tts/providers/nonexistent/voices")
        assert response.status_code == 404


class TestGenerateEndpoint:
    def test_generate_sync(self, mock_app):
        client, _mock_tts, _ = mock_app
        response = client.post(
            "/api/tts/generate",
            json={"text": "Hello world", "provider": "kokoro"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0

    def test_generate_unknown_provider(self, mock_app):
        client, _, _ = mock_app
        response = client.post(
            "/api/tts/generate",
            json={"text": "Hello", "provider": "nonexistent"},
        )
        assert response.status_code == 404
