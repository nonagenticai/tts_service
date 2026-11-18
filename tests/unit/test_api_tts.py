"""Unit tests for TTS API endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import Response

from src.api.schemas import TTSGenerateRequest
from src.api.tts import (
    _get_registry,
    _get_storage,
    _get_tenant_id,
    _get_tts_service,
    generate,
)
from src.providers.base import GenerationResult
from tests.conftest import tenant_uuid


class TestHelperFunctions:
    def test_get_tenant_id_with_state(self):
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("my-tenant")
        result = _get_tenant_id(request)
        assert result == tenant_uuid("my-tenant")

    @pytest.mark.parametrize("state", [SimpleNamespace(), SimpleNamespace(tenant_id="")])
    def test_get_tenant_id_unresolved_raises_401_never_substitutes(self, state):
        """Inverted contract: this used to return the nil-UUID sentinel.

        PRODUCT-OWNER RULING — an absent tenant is an ERROR, never a default.
        The nil UUID passed every uuid cast, and billing bucketed it as
        unattributed and NEVER CHARGED it: real usage, silently unbilled.
        """
        request = MagicMock(spec=[])
        request.state = state
        with pytest.raises(HTTPException) as exc_info:
            _get_tenant_id(request)
        assert exc_info.value.status_code == 401
        assert "no sentinel" in str(exc_info.value.detail)

    def test_get_tenant_id_strips_and_returns_the_real_tenant(self):
        request = MagicMock(spec=[])
        request.state = SimpleNamespace(tenant_id=f"  {tenant_uuid('real-tenant')}  ")
        assert _get_tenant_id(request) == tenant_uuid("real-tenant")

    @pytest.mark.parametrize("slug", ["n8n-dev", "tenant-1", "default"])
    def test_get_tenant_id_non_uuid_raises_400_not_401_and_not_500(self, slug):
        """A slug is a bad ARGUMENT, and must never reach postgres.

        ``tenant_id`` is a ``uuid`` column (revision ``005_tenant_id_uuid``).
        Without this branch the slug travels all the way to psycopg and comes
        back as a ``DataError`` — a 500 raised part-way through a job write,
        after the caller has already been charged for the synthesis. 400 here,
        before anything is recorded. 401 would be a lie: something DID
        authenticate; the value it carried is simply unstorable.
        """
        request = MagicMock(spec=[])
        request.state = SimpleNamespace(tenant_id=slug)
        with pytest.raises(HTTPException) as exc_info:
            _get_tenant_id(request)
        assert exc_info.value.status_code == 400
        assert "must be a UUID" in str(exc_info.value.detail)

    def test_get_tenant_id_folds_an_uppercase_uuid_to_the_canonical_form(self):
        """Postgres renders the column lowercase; the helper must match it."""
        request = MagicMock(spec=[])
        request.state = SimpleNamespace(tenant_id=tenant_uuid("real-tenant").upper())
        assert _get_tenant_id(request) == tenant_uuid("real-tenant")

    def test_get_tts_service(self):
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_service = MagicMock()
        request.app.state.tts_service = mock_service
        result = _get_tts_service(request)
        assert result is mock_service

    def test_get_registry(self):
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry
        result = _get_registry(request)
        assert result is mock_registry

    def test_get_storage(self):
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = MagicMock()
        request.app.state.storage_service = mock_storage
        result = _get_storage(request)
        assert result is mock_storage


class TestProcessJobBackground:
    @pytest.mark.asyncio
    async def test_process_job_background_success(self):
        from src.api.tts import _process_job_background

        tts = AsyncMock()
        mock_db = AsyncMock()

        with patch("src.api.tts.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_background(tts, "job-1", tenant_uuid("tenant-1"))

        tts.process_job.assert_called_once_with(
            mock_db, "job-1", tenant_uuid("tenant-1"), post_processing=None
        )

    @pytest.mark.asyncio
    async def test_process_job_background_exception(self):
        from src.api.tts import _process_job_background

        tts = AsyncMock()
        tts.process_job = AsyncMock(side_effect=RuntimeError("generation failed"))
        mock_db = AsyncMock()

        with patch("src.api.tts.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            # Should not raise (logs instead)
            await _process_job_background(tts, "job-1", tenant_uuid("tenant-1"))


class TestGenerateEndpoint:
    @pytest.mark.asyncio
    async def test_generate_sync_returns_response(self):
        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate
        from src.providers.base import GenerationResult

        body = TTSGenerateRequest(text="Hello world", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("tenant-1")
        request.headers = {}

        mock_tts = AsyncMock()
        mock_result = GenerationResult(
            audio_data=b"audio data",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.5,
        )
        mock_tts.generate_sync = AsyncMock(
            return_value=(mock_result, {"resolved_provider": "kokoro", "estimated_cost": 0.0})
        )
        request.app.state.tts_service = mock_tts

        mock_registry = MagicMock()
        mock_registry.get_provider = MagicMock()
        request.app.state.provider_registry = mock_registry

        db = AsyncMock()

        response = await generate(body, request, db)
        assert isinstance(response, Response)
        assert response.body == b"audio data"
        assert response.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_generate_unknown_provider_returns_404(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate

        body = TTSGenerateRequest(text="Hello", provider="nonexistent")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")

        mock_registry = MagicMock()
        mock_registry.get_provider.side_effect = KeyError("not registered")
        request.app.state.provider_registry = mock_registry

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate(body, request, db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_provider_unavailable_returns_503(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate

        body = TTSGenerateRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.headers = {}

        mock_tts = AsyncMock()
        mock_tts.generate_sync = AsyncMock(side_effect=RuntimeError("Provider not available"))
        request.app.state.tts_service = mock_tts

        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate(body, request, db)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_generate_value_error_returns_400(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate

        body = TTSGenerateRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.headers = {}

        mock_tts = AsyncMock()
        mock_tts.generate_sync = AsyncMock(side_effect=ValueError("bad input"))
        request.app.state.tts_service = mock_tts

        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate(body, request, db)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_generate_wav_format(self):
        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate
        from src.providers.base import GenerationResult

        body = TTSGenerateRequest(text="Hello", output_format="wav")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.headers = {}

        mock_tts = AsyncMock()
        result = GenerationResult(
            audio_data=b"wav data",
            format="wav",
            sample_rate=24000,
            duration_seconds=1.0,
        )
        mock_tts.generate_sync = AsyncMock(
            return_value=(result, {"resolved_provider": "kokoro", "estimated_cost": 0.0})
        )
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()
        db = AsyncMock()

        response = await generate(body, request, db)
        assert isinstance(response, Response)
        assert response.media_type == "audio/wav"

    @pytest.mark.asyncio
    async def test_generate_cached_response_headers(self):
        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate
        from src.providers.base import GenerationResult

        body = TTSGenerateRequest(text="Hello")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.headers = {}

        mock_tts = AsyncMock()
        result = GenerationResult(
            audio_data=b"audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.0,
        )
        mock_tts.generate_sync = AsyncMock(
            return_value=(
                result,
                {
                    "cached": True,
                    "resolved_provider": "kokoro",
                    "detected_language": "en",
                    "estimated_cost": 0.0,
                },
            )
        )
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()
        db = AsyncMock()

        response = await generate(body, request, db)
        assert isinstance(response, Response)
        assert response.headers.get("X-Cache") == "HIT"
        assert response.headers.get("X-Provider") == "kokoro"
        assert response.headers.get("X-Detected-Language") == "en"


class TestListProviders:
    @pytest.mark.asyncio
    async def test_list_providers(self):
        from src.api.tts import list_providers

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_provider = AsyncMock()
        mock_provider.name = "kokoro"
        mock_provider.display_name = "Kokoro"
        mock_provider.provider_type = "local"
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.list_voices = AsyncMock(return_value=[MagicMock()])
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.max_text_length = 10000
        mock_provider.requires_api_key = False
        mock_provider.supports_streaming = False
        mock_provider.supports_ssml = False

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = [mock_provider]
        request.app.state.provider_registry = mock_registry
        request.app.state.settings = MagicMock()
        request.app.state.settings.routing.cost_per_1k_chars = {"kokoro": 0.0}

        result = await list_providers(request)
        assert len(result) == 1
        assert result[0].name == "kokoro"
        assert result[0].available is True


class TestListVoices:
    @pytest.mark.asyncio
    async def test_list_voices_success(self):
        from src.api.tts import list_voices
        from src.providers.base import VoiceMetadata

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_provider = AsyncMock()
        voice = VoiceMetadata(id="af_heart", display_name="AF Heart", language="en")
        mock_provider.list_voices = AsyncMock(return_value=[voice])

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider
        request.app.state.provider_registry = mock_registry

        result = await list_voices("kokoro", request)
        assert len(result) == 1
        assert result[0].id == "af_heart"

    @pytest.mark.asyncio
    async def test_list_voices_unknown_provider(self):
        from fastapi import HTTPException

        from src.api.tts import list_voices

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        mock_registry.get_provider.side_effect = KeyError("not registered")
        request.app.state.provider_registry = mock_registry

        with pytest.raises(HTTPException) as exc_info:
            await list_voices("nonexistent", request)
        assert exc_info.value.status_code == 404


class TestGetJobStatus:
    @pytest.mark.asyncio
    async def test_job_not_found(self):
        from fastapi import HTTPException

        from src.api.tts import get_job_status

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_job_status("nonexistent", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_job_found(self):
        from src.api.tts import get_job_status

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.get_presigned_url = AsyncMock(return_value=None)
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "j1"
        mock_job.status = "pending"
        mock_job.provider = "kokoro"
        mock_job.voice = "af_heart"
        mock_job.created_at = None
        mock_job.completed_at = None
        mock_job.error_message = None
        mock_job.audio_path = None
        mock_job.audio_duration_seconds = None
        mock_job.audio_size_bytes = None
        mock_job.cached = False
        mock_job.detected_language = None
        mock_job.routing_strategy = None
        mock_job.estimated_cost_usd = None

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            result = await get_job_status("j1", request, db)
            assert result.job_id == "j1"
            assert result.status == "pending"


class TestDownloadAudio:
    @pytest.mark.asyncio
    async def test_audio_not_found(self):
        from fastapi import HTTPException

        from src.api.tts import download_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await download_audio("j1", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_audio_not_completed(self):
        from fastapi import HTTPException

        from src.api.tts import download_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "pending"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            with pytest.raises(HTTPException) as exc_info:
                await download_audio("j1", request, db)
            assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_audio_success(self):
        from src.api.tts import download_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.retrieve = AsyncMock(return_value=b"audio bytes")
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/path/audio.mp3"
        mock_job.output_format = "mp3"
        mock_job.audio_duration_seconds = 1.5

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            result = await download_audio("j1", request, db)
            assert result.body == b"audio bytes"
            assert result.media_type == "audio/mpeg"


class TestGetAudioPresignedUrl:
    @pytest.mark.asyncio
    async def test_job_not_found(self):
        from fastapi import HTTPException

        from src.api.tts import get_audio_presigned_url

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_audio_presigned_url("j1", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_presigned_url_returns_download_fallback_for_local(self):
        from src.api.tts import get_audio_presigned_url

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.base_url = "http://localhost:6194/"
        mock_storage = AsyncMock()
        mock_storage.get_presigned_url = AsyncMock(return_value=None)
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/path/audio.mp3"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            result = await get_audio_presigned_url("j1", request, db)
            assert result["url"] == "http://localhost:6194/api/tts/jobs/j1/audio"
            assert result["expires_in"] is None
            assert result["backend"] == "local"

    @pytest.mark.asyncio
    async def test_presigned_url_success(self):
        from src.api.tts import get_audio_presigned_url

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.get_presigned_url = AsyncMock(return_value="https://example.com/audio")
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/path/audio.mp3"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            result = await get_audio_presigned_url("j1", request, db)
            assert result["url"] == "https://example.com/audio"
            assert result["expires_in"] == 3600


@pytest.mark.asyncio
class TestGenerateRejectsMissingTenant:
    """A tenant-less request must 401 AND emit NO cost event.

    Both halves matter: a 401 alone would still pass if the route emitted a
    sentinel-attributed cost event on the way out — which is exactly the
    silently-unbilled traffic the ruling forbids.
    """

    async def test_no_tenant_is_401_and_emits_no_cost_event(self):
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")

        async def _generate_and_emit(**kwargs):
            """Stand in for the real generate_sync, which DOES emit a cost event.

            The emit is modelled here on purpose: if the route ever stops
            rejecting a tenant-less request, generation proceeds and this XADD
            lands — which is precisely the silently-unbilled traffic the ruling
            forbids, and what ``assert_not_awaited`` below catches.
            """
            await redis_mock.xadd(
                "billing:costs",
                {"tenant_id": kwargs.get("tenant_id", ""), "cost_source": "w2v-tts"},
            )
            return (
                GenerationResult(
                    audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
                ),
                {"resolved_provider": "stub", "cached": False, "latency_ms": 1.0},
            )

        tts = MagicMock()
        tts._metering_redis = redis_mock
        tts.generate_sync = AsyncMock(side_effect=_generate_and_emit)

        request = MagicMock()
        request.state = SimpleNamespace()  # auth resolved no tenant
        request.app.state.tts_service = tts
        request.app.state.provider_registry = MagicMock()
        request.headers = {}

        body = TTSGenerateRequest(text="hello world", provider="stub")

        raised: HTTPException | None = None
        try:
            await generate(body, request, db=AsyncMock())
        except HTTPException as exc:
            raised = exc

        # Half 1 FIRST, and independent of half 2: whatever the route decided,
        # a tenant-less request must never put a cost event on the bus. Asserting
        # this before the status code keeps it load-bearing — a 4xx alone would
        # pass even if the route emitted a sentinel-attributed event on the way out.
        redis_mock.xadd.assert_not_awaited()
        tts.generate_sync.assert_not_awaited()
        # Half 2: and the rejection is a 401.
        assert raised is not None, "tenant-less request was not rejected"
        assert raised.status_code == 401
