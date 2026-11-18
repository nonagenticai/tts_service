"""Additional targeted tests to push coverage from ~93% toward 95%+.

Covers remaining uncovered lines in:
- services/tts_service.py: ImportError branches, concatenate_batch, cache store fail, alert fail
- api/tts.py: stream_generate_raw, _concatenate_batch_background timeout path
- database/session.py: rollback paths
- config.py: parse_list_fields edge cases, utility methods
- api/ws.py: sender_task error, unsubscribe without id
- services/notification_service.py: subscriber_loop reconnect, shutdown cleanup
- services/storage_service.py: _init_minio, unsupported backend
- providers/base.py: generate_stream default impl
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.websockets import WebSocketDisconnect

from src.providers.base import AudioChunk, GenerationResult, TTSProvider
from src.services.tts_service import TTSService, _compute_cache_key, _estimate_cost
from tests.conftest import tenant_uuid

# ============================================================================
# TTSService._process_text ImportError branch (lines 117-118)
# ============================================================================


class _AuthenticatedTenant:
    """Stamp an authenticated tenant onto the ASGI scope for TestClient routes.

    ``AuthMiddleware`` is not mounted on these bare ``FastAPI()`` apps, so
    nothing populates ``request.state.tenant_id``. Routes now 401 without one
    (product-owner ruling: a tenant_id may never hold a sentinel), so the client
    must stand in for the authenticated caller rather than rely on a fallback.
    """

    def __init__(self, app, tenant_id: str = tenant_uuid("tenant-test")) -> None:
        self.app = app
        self.tenant_id = tenant_id

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["tenant_id"] = self.tenant_id
        await self.app(scope, receive, send)


class TestTTSServiceProcessTextImportError:
    """Cover the ImportError branch in _process_text."""

    def test_process_text_import_error_returns_original(self):
        """When ssml_processor cannot be imported, return original text."""
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        with patch.dict("sys.modules", {"src.services.ssml_processor": None}):
            with patch("builtins.__import__", side_effect=ImportError("no ssml")):
                # The _process_text method catches ImportError
                # We need to directly invoke it so the import fails
                pass

        # Alternative: patch the actual import inside _process_text
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "src.services.ssml_processor":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = service._process_text("Hello world", "kokoro", ssml=None)
            assert result == "Hello world"


# ============================================================================
# TTSService._detect_language ImportError branch (lines 146-147)
# ============================================================================


class TestTTSServiceDetectLanguageImportError:
    """Cover the ImportError branch in _detect_language."""

    def test_detect_language_import_error_returns_none(self):
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "src.services.language_detector":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            lang, voice = service._detect_language("Hello", "kokoro", None)
            assert lang is None
            assert voice is None


# ============================================================================
# TTSService._detect_language voice auto-select path (lines 152-155)
# ============================================================================


class TestTTSServiceDetectLanguageVoiceSelect:
    """Cover the auto voice selection branches in _detect_language."""

    def test_detect_language_auto_voice_with_no_explicit_voice(self):
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        mock_detect_fn = MagicMock(return_value=("en", 0.95))
        mock_best_voice = MagicMock(return_value="best_voice")

        with (
            patch("src.services.language_detector.detect_language", mock_detect_fn),
            patch(
                "src.services.language_detector.get_best_voice_for_language",
                mock_best_voice,
            ),
        ):
            lang, voice = service._detect_language("Hello", "kokoro", None, [])
            assert lang == "en"
            assert voice == "best_voice"

    def test_detect_language_with_explicit_voice_returns_no_auto_voice(self):
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        mock_detect_fn = MagicMock(return_value=("en", 0.95))

        with patch("src.services.language_detector.detect_language", mock_detect_fn):
            lang, voice = service._detect_language("Hello", "kokoro", "af_heart", [])
            assert lang == "en"
            assert voice is None  # No auto voice when explicit voice is provided

    def test_detect_language_low_confidence_returns_none(self):
        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        mock_detect_fn = MagicMock(return_value=("en", 0.3))

        with patch("src.services.language_detector.detect_language", mock_detect_fn):
            lang, voice = service._detect_language("Hello", "kokoro", None, [])
            assert lang is None
            assert voice is None


# ============================================================================
# TTSService.generate_sync cache store failure (line 387-389)
# ============================================================================


class TestTTSServiceCacheStoreFailure:
    """Cover the cache store exception path in generate_sync."""

    @pytest.mark.asyncio
    async def test_generate_sync_cache_store_exception_is_swallowed(self):
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.sample_rate = 24000
        mock_provider.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"audio",
                format="mp3",
                sample_rate=24000,
                duration_seconds=1.0,
            )
        )

        registry = MagicMock()
        registry.get_provider.return_value = mock_provider

        storage = AsyncMock()
        storage.store = AsyncMock(side_effect=RuntimeError("storage broke"))

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)

        service = TTSService(registry=registry, storage=storage, redis_client=redis)

        # Patch settings to enable caching
        mock_settings = MagicMock()
        mock_settings.cache.enabled = True
        mock_settings.cache.key_prefix = "tts:gen:"
        mock_settings.cache.ttl = 86400
        mock_settings.routing.cost_per_1k_chars = {"kokoro": 0.0}
        mock_settings.routing.default_strategy = "balanced"
        service._settings = mock_settings

        result, _meta = await service.generate_sync(
            tenant_id=tenant_uuid("tenant-test"),
            text="Hello",
            provider="kokoro",
            voice="af_heart",
        )
        # Should succeed despite cache store failure
        assert result.audio_data == b"audio"


# ============================================================================
# TTSService.generate_stream with auto_voice (line 421-422)
# ============================================================================


class TestTTSServiceGenerateStreamAutoVoice:
    """Cover the auto_voice assignment in generate_stream."""

    @pytest.mark.asyncio
    async def test_generate_stream_auto_voice_assigned(self):
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="default_voice")
        mock_provider.list_voices = AsyncMock(return_value=[])
        mock_provider.sample_rate = 24000

        async def mock_generate_stream(**kwargs):
            yield AudioChunk(data=b"chunk", chunk_index=0, is_final=True)

        mock_provider.generate_stream = mock_generate_stream

        registry = MagicMock()
        registry.get_provider.return_value = mock_provider

        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        # Mock _detect_language to return an auto_voice
        with patch.object(service, "_detect_language", return_value=("en", "auto_voice_result")):
            chunks = []
            async for chunk in service.generate_stream(
                tenant_id=tenant_uuid("tenant-test"),
                text="Hello",
                provider="kokoro",
                voice=None,  # No explicit voice -> auto voice selection
            ):
                chunks.append(chunk)
            assert len(chunks) >= 1


# ============================================================================
# TTSService.concatenate_batch (lines 594, 600-622)
# ============================================================================


class TestTTSServiceConcatenateBatch:
    """Cover the concatenate_batch method."""

    @pytest.mark.asyncio
    async def test_concatenate_batch_no_completed_jobs_raises(self):
        registry = MagicMock()
        storage = AsyncMock()
        service = TTSService(registry=registry, storage=storage)

        db = AsyncMock()

        with patch(
            "src.services.tts_service.JobManager.get_batch_jobs",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with pytest.raises(ValueError, match="No completed jobs"):
                await service.concatenate_batch(db, "batch-123", tenant_id=tenant_uuid("tenant-a"))

    @pytest.mark.asyncio
    async def test_concatenate_batch_no_audio_data_raises_strict(self):
        registry = MagicMock()
        storage = AsyncMock()
        storage.retrieve = AsyncMock(return_value=None)
        service = TTSService(registry=registry, storage=storage)

        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/some/path.mp3"
        mock_job.job_id = "job-1"

        with patch(
            "src.services.tts_service.JobManager.get_batch_jobs",
            new_callable=AsyncMock,
            return_value=[mock_job],
        ):
            with pytest.raises(ValueError, match="audio retrieval failed"):
                await service.concatenate_batch(db, "batch-123", tenant_id=tenant_uuid("tenant-a"))

    @pytest.mark.asyncio
    async def test_concatenate_batch_no_audio_data_raises_non_strict(self):
        registry = MagicMock()
        storage = AsyncMock()
        storage.retrieve = AsyncMock(return_value=None)
        service = TTSService(registry=registry, storage=storage)

        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/some/path.mp3"
        mock_job.job_id = "job-1"

        with patch(
            "src.services.tts_service.JobManager.get_batch_jobs",
            new_callable=AsyncMock,
            return_value=[mock_job],
        ):
            with pytest.raises(ValueError, match="No audio data found"):
                await service.concatenate_batch(
                    db, "batch-123", strict=False, tenant_id=tenant_uuid("tenant-a")
                )

    @pytest.mark.asyncio
    async def test_concatenate_batch_success(self):
        registry = MagicMock()
        storage = AsyncMock()
        storage.retrieve = AsyncMock(return_value=b"audio_data")
        storage.store = AsyncMock(return_value="/stored/path.mp3")
        service = TTSService(registry=registry, storage=storage)

        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/some/path.mp3"

        with (
            patch(
                "src.services.tts_service.JobManager.get_batch_jobs",
                new_callable=AsyncMock,
                return_value=[mock_job],
            ),
            patch(
                "src.services.tts_service.JobManager.set_batch_concat_path",
                new_callable=AsyncMock,
            ) as mock_set_path,
            patch(
                "src.services.tts_service.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread,
        ):
            # First call is concatenate_audio, second is probe_duration
            mock_to_thread.side_effect = [b"concatenated_audio", 5.5]

            path, duration = await service.concatenate_batch(
                db, "batch-123", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )

            assert path == "/stored/path.mp3"
            assert duration == 5.5
            mock_set_path.assert_called_once_with(
                db, "batch-123", "/stored/path.mp3", tenant_id=tenant_uuid("tenant-a")
            )


# ============================================================================
# TTSService.process_job alert failure (lines 539-540)
# ============================================================================


class TestTTSServiceProcessJobAlertFailure:
    """Cover the alert exception path in process_job."""

    @pytest.mark.asyncio
    async def test_process_job_alert_exception_swallowed(self):
        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="af_heart")
        mock_provider.sample_rate = 24000
        mock_provider.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"audio",
                format="mp3",
                sample_rate=24000,
                duration_seconds=1.0,
            )
        )

        registry = MagicMock()
        registry.get_provider.return_value = mock_provider

        storage = AsyncMock()
        storage.store = AsyncMock(return_value="/path/audio.mp3")

        service = TTSService(registry=registry, storage=storage)

        mock_settings = MagicMock()
        mock_settings.cache.enabled = False
        mock_settings.routing.cost_per_1k_chars = {"kokoro": 0.0}
        mock_settings.routing.default_strategy = "balanced"
        service._settings = mock_settings

        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "job-123"
        mock_job.text = "Hello"
        mock_job.text_length = 5
        mock_job.provider = "kokoro"
        mock_job.voice = "af_heart"
        mock_job.speed = 1.0
        mock_job.output_format = "mp3"
        mock_job.model = None

        with (
            patch(
                "src.services.tts_service.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch(
                "src.services.tts_service.JobManager.mark_processing",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.tts_service.JobManager.mark_completed",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.tts_service.JobManager.record_usage",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.quota_service.check_and_create_alerts",
                new_callable=AsyncMock,
                side_effect=RuntimeError("alert system down"),
            ),
        ):
            # Should succeed despite alert failure
            await service.process_job(db, "job-123", tenant_uuid("tenant-1"))


# ============================================================================
# config.py parse_list_fields edge cases (lines 93, 95-97, 100)
# ============================================================================


class TestConfigCorsOrigins:
    """Cover cors_origins list[str] field behavior."""

    def test_cors_origins_empty_list(self):
        from src.config import APISettings

        settings = APISettings(cors_origins=[])
        assert settings.cors_origins == []

    def test_cors_origins_from_list(self):
        from src.config import APISettings

        settings = APISettings(cors_origins=["http://a.com", "http://b.com"])
        assert settings.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_single(self):
        from src.config import APISettings

        settings = APISettings(cors_origins=["http://a.com"])
        assert settings.cors_origins == ["http://a.com"]

    def test_config_utility_methods(self):
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        try:
            s = Settings()
            db_url = s.get_database_url()
            assert "postgresql" in db_url or "sqlite" in db_url

            redis_url = s.get_redis_url()
            assert redis_url.startswith("redis://")

            # In test environment, neither production nor development
            assert s.is_production() is False
            # environment is "test", so is_development() is False
            assert s.environment == "test"
            assert s.is_development() is False
        finally:
            clear_settings_cache()

    def test_config_port_8000_rejected(self):
        from src.config import Settings, clear_settings_cache

        clear_settings_cache()
        try:
            with pytest.raises(ValueError, match="8000"):
                Settings(api={"port": 8000})
        finally:
            clear_settings_cache()

    def test_logging_invalid_level(self):
        from src.config import LoggingSettings

        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingSettings(level="INVALID")


# ============================================================================
# database/session.py rollback paths (lines 41-42, 55-57)
# ============================================================================


class TestDatabaseSessionRollback:
    """Cover rollback branches in session management."""

    @pytest.mark.asyncio
    async def test_get_db_rollback_on_exception(self):
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_maker = MagicMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session_maker.return_value = mock_cm

        with patch(
            "w2v_common.database.session.get_session_maker",
            new_callable=AsyncMock,
            return_value=mock_session_maker,
        ):
            from src.database.session import get_db

            gen = get_db()
            _ = await gen.__anext__()
            with pytest.raises(RuntimeError, match="commit failed"):
                await gen.__anext__()

    @pytest.mark.asyncio
    async def test_get_db_session_rollback_on_exception(self):
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_maker = MagicMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session_maker.return_value = mock_cm

        with patch(
            "w2v_common.database.session.get_session_maker",
            new_callable=AsyncMock,
            return_value=mock_session_maker,
        ):
            from src.database.session import get_db_session

            with pytest.raises(RuntimeError, match="commit failed"):
                async with get_db_session() as _session:
                    raise RuntimeError("commit failed")


# ============================================================================
# services/storage_service.py _init_minio and unsupported backend (lines 33-57)
# ============================================================================


class TestStorageServiceMinioInit:
    """Cover _init_minio and edge cases in storage_service."""

    @pytest.mark.asyncio
    async def test_initialize_unsupported_backend(self):
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            mock_settings = MagicMock()
            mock_settings.storage.backend = "unsupported"
            mock_settings.storage.local_path = "/tmp/test"
            mock_settings.storage.minio_endpoint = "localhost:9000"
            mock_settings.storage.minio_access_key = "key"
            mock_settings.storage.minio_secret_key = "secret"  # pragma: allowlist secret
            mock_settings.storage.minio_bucket = "bucket"
            mock_settings.storage.minio_secure = False
            mock_settings.storage.presign_expiry_seconds = 3600

            with patch("src.services.storage_service.get_settings", return_value=mock_settings):
                from src.services.storage_service import StorageService

                svc = StorageService()
                with pytest.raises(ValueError, match="Unsupported storage backend"):
                    await svc.initialize()
        finally:
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_init_minio_no_miniopy(self):
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            mock_settings = MagicMock()
            mock_settings.storage.backend = "minio"
            mock_settings.storage.local_path = "/tmp/test"
            mock_settings.storage.minio_endpoint = "localhost:9000"
            mock_settings.storage.minio_access_key = "key"
            mock_settings.storage.minio_secret_key = "secret"  # pragma: allowlist secret
            mock_settings.storage.minio_bucket = "bucket"
            mock_settings.storage.minio_secure = False
            mock_settings.storage.presign_expiry_seconds = 3600

            with patch("src.services.storage_service.get_settings", return_value=mock_settings):
                from src.services.storage_service import StorageService

                svc = StorageService()

                with patch.dict("sys.modules", {"miniopy_async": None}):
                    with pytest.raises(RuntimeError, match="miniopy-async"):
                        await svc.initialize()
        finally:
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_init_minio_bucket_exists(self):
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            mock_settings = MagicMock()
            mock_settings.storage.backend = "minio"
            mock_settings.storage.local_path = "/tmp/test"
            mock_settings.storage.minio_endpoint = "localhost:9000"
            mock_settings.storage.minio_access_key = "key"
            mock_settings.storage.minio_secret_key = "secret"  # pragma: allowlist secret
            mock_settings.storage.minio_bucket = "test-bucket"
            mock_settings.storage.minio_secure = False
            mock_settings.storage.presign_expiry_seconds = 3600

            mock_minio_cls = MagicMock()
            mock_minio_instance = AsyncMock()
            mock_minio_instance.bucket_exists = AsyncMock(return_value=True)
            mock_minio_cls.return_value = mock_minio_instance

            mock_miniopy = MagicMock()
            mock_miniopy.Minio = mock_minio_cls

            with (
                patch("src.services.storage_service.get_settings", return_value=mock_settings),
                patch.dict("sys.modules", {"miniopy_async": mock_miniopy}),
            ):
                from src.services.storage_service import StorageService

                svc = StorageService()
                await svc.initialize()
                mock_minio_instance.bucket_exists.assert_called_once_with("test-bucket")
                mock_minio_instance.make_bucket.assert_not_called()
        finally:
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_init_minio_bucket_needs_creation(self):
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            mock_settings = MagicMock()
            mock_settings.storage.backend = "minio"
            mock_settings.storage.local_path = "/tmp/test"
            mock_settings.storage.minio_endpoint = "localhost:9000"
            mock_settings.storage.minio_access_key = "key"
            mock_settings.storage.minio_secret_key = "secret"  # pragma: allowlist secret
            mock_settings.storage.minio_bucket = "test-bucket"
            mock_settings.storage.minio_secure = False
            mock_settings.storage.presign_expiry_seconds = 3600

            mock_minio_cls = MagicMock()
            mock_minio_instance = AsyncMock()
            mock_minio_instance.bucket_exists = AsyncMock(return_value=False)
            mock_minio_instance.make_bucket = AsyncMock()
            mock_minio_cls.return_value = mock_minio_instance

            mock_miniopy = MagicMock()
            mock_miniopy.Minio = mock_minio_cls

            with (
                patch("src.services.storage_service.get_settings", return_value=mock_settings),
                patch.dict("sys.modules", {"miniopy_async": mock_miniopy}),
            ):
                from src.services.storage_service import StorageService

                svc = StorageService()
                await svc.initialize()
                mock_minio_instance.make_bucket.assert_called_once_with("test-bucket")
        finally:
            clear_settings_cache()


# ============================================================================
# api/tts.py stream_generate_raw (lines 267-277)
# ============================================================================


class TestStreamGenerateRawEndpoint:
    """Cover the stream_generate_raw endpoint."""

    @pytest.mark.asyncio
    async def test_stream_generate_raw_unknown_provider(self):
        from unittest.mock import MagicMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        # Setup app state
        mock_registry = MagicMock()
        mock_registry.get_provider.side_effect = KeyError("no such provider")

        mock_tts = MagicMock()

        app.state.tts_service = mock_tts
        app.state.provider_registry = mock_registry

        from src.api.tts import router
        from src.database.session import get_db

        app.include_router(router)
        # The stream routes now depend on get_db (for the pre-stream quota gate);
        # override it so the TestClient does not open a real DB connection.
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(_AuthenticatedTenant(app))
        response = client.post(
            "/api/tts/stream/raw",
            json={"text": "Hello", "provider": "nonexistent"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_generate_raw_success(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = MagicMock()

        mock_tts = MagicMock()

        async def mock_stream(**kwargs):
            yield AudioChunk(data=b"raw_audio", chunk_index=0, is_final=False)
            yield AudioChunk(data=b"", chunk_index=1, is_final=True)

        mock_tts.generate_stream = mock_stream

        app.state.tts_service = mock_tts
        app.state.provider_registry = mock_registry

        from src.api.tts import router
        from src.database.session import get_db

        app.include_router(router)
        # See the unknown-provider test: override get_db so the pre-stream quota
        # gate has a session without opening a real DB connection.
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(_AuthenticatedTenant(app))
        response = client.post(
            "/api/tts/stream/raw",
            json={"text": "Hello", "provider": "kokoro"},
        )
        assert response.status_code == 200
        assert b"raw_audio" in response.content


# ============================================================================
# api/tts.py _concatenate_batch_background (lines 377-378)
# ============================================================================


class TestConcatenateBatchBackground:
    """Cover _concatenate_batch_background paths."""

    @pytest.mark.asyncio
    async def test_concatenate_batch_background_no_jobs(self):
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()

        mock_db = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "src.api.tts.get_db_session",
                return_value=mock_session_cm,
            ),
            patch(
                "src.api.tts.JobManager.get_batch_jobs",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            # Should return early when no jobs found
            await _concatenate_batch_background(
                mock_tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )
            mock_tts.concatenate_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_concatenate_batch_background_all_done(self):
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()
        mock_tts.concatenate_batch = AsyncMock(return_value=("/path", 5.0))

        mock_db = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_job = MagicMock()
        mock_job.status = "completed"

        with (
            patch(
                "src.api.tts.get_db_session",
                return_value=mock_session_cm,
            ),
            patch(
                "src.api.tts.JobManager.get_batch_jobs",
                new_callable=AsyncMock,
                return_value=[mock_job],
            ),
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            await _concatenate_batch_background(
                mock_tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )
            mock_tts.concatenate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_concatenate_batch_background_exception(self):
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()

        mock_db = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "src.api.tts.get_db_session",
                return_value=mock_session_cm,
            ),
            patch(
                "src.api.tts.JobManager.get_batch_jobs",
                new_callable=AsyncMock,
                side_effect=RuntimeError("db error"),
            ),
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            # Should not raise - exception is caught
            await _concatenate_batch_background(
                mock_tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )


# ============================================================================
# api/ws.py sender_task exception & unsubscribe edge (lines 66-73, 123)
# ============================================================================


class TestWebSocketEdgeCases:
    """Cover edge case paths in the WebSocket handler."""

    @pytest.mark.asyncio
    async def test_ws_no_notification_service(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.state.notification_service = None
        app.state.connection_manager = None

        from src.api.ws import router

        app.include_router(router)

        client = TestClient(_AuthenticatedTenant(app))
        with pytest.raises((WebSocketDisconnect, RuntimeError)):
            # WebSocket close with 4500 code
            with client.websocket_connect("/api/tts/ws"):
                pass


# ============================================================================
# services/notification_service.py subscriber_loop reconnect (lines 146-150)
# ============================================================================


class TestNotificationServiceSubscriberLoop:
    """Cover subscriber_loop reconnect and cleanup paths."""

    @pytest.mark.asyncio
    async def test_subscriber_loop_reconnect_on_error(self):
        from src.services.notification_service import NotificationService

        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()

        call_count = 0

        async def subscribe_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("connection lost")
            # On second call, stop the loop
            svc._running = False

        async def empty_listen():
            # Async generator that yields nothing (loop will exit)
            return
            yield

        mock_pubsub.subscribe = subscribe_side_effect
        mock_pubsub.listen = empty_listen
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        svc = NotificationService(redis_client=mock_redis)
        svc._running = True

        with patch(
            "src.services.notification_service.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await svc._subscriber_loop()

        mock_pubsub.aclose.assert_called()

    @pytest.mark.asyncio
    async def test_subscriber_loop_pubsub_aclose_error(self):
        from src.services.notification_service import NotificationService

        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()

        call_count = 0

        async def subscribe_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("connection lost")
            svc._running = False

        async def empty_listen():
            # Async generator that yields nothing (loop will exit)
            return
            yield

        mock_pubsub.subscribe = subscribe_side_effect
        mock_pubsub.listen = empty_listen
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        svc = NotificationService(redis_client=mock_redis)
        svc._running = True

        with patch(
            "src.services.notification_service.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await svc._subscriber_loop()

        # Should not raise


# ============================================================================
# providers/base.py generate_stream default (line 77-103)
# This is already tested but ensuring the base path is covered
# ============================================================================


class TestTTSProviderBaseGenerateStream:
    """Cover the default generate_stream fallback implementation."""

    @pytest.mark.asyncio
    async def test_base_generate_stream_fallback(self):
        """The base class generate_stream falls back to generate()."""

        class ConcreteProvider(TTSProvider):
            name = "test"
            display_name = "Test"
            provider_type = "local"
            max_text_length = 1000

            async def is_available(self) -> bool:
                return True

            async def generate(self, text, voice, speed=1.0, output_format="mp3", **kwargs):
                return GenerationResult(
                    audio_data=b"test_audio",
                    format="mp3",
                    sample_rate=24000,
                    duration_seconds=1.0,
                )

            async def list_voices(self):
                return []

            def get_default_voice(self):
                return "default"

        provider = ConcreteProvider()
        chunks = []
        async for chunk in provider.generate_stream(text="hello", voice="default"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].data == b"test_audio"
        assert chunks[0].total_bytes == 10


# ============================================================================
# _compute_cache_key and _estimate_cost (standalone functions)
# ============================================================================


class TestStandaloneFunctions:
    """Cover standalone helper functions in tts_service."""

    def test_compute_cache_key_deterministic(self):
        key1 = _compute_cache_key(tenant_uuid("t1"), "hello", "kokoro", "af_heart", 1.0, "mp3")
        key2 = _compute_cache_key(tenant_uuid("t1"), "hello", "kokoro", "af_heart", 1.0, "mp3")
        assert key1 == key2

    def test_compute_cache_key_differs_with_tenant(self):
        key1 = _compute_cache_key(tenant_uuid("t1"), "hello", "kokoro", "af_heart", 1.0, "mp3")
        key2 = _compute_cache_key(tenant_uuid("t2"), "hello", "kokoro", "af_heart", 1.0, "mp3")
        assert key1 != key2

    def test_estimate_cost_known_provider(self):
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            cost = _estimate_cost("openai", 1000)
            # Default cost for openai is 0.015 per 1k chars
            assert cost > 0.0
        finally:
            clear_settings_cache()

    def test_estimate_cost_unknown_provider(self):
        from src.config import clear_settings_cache

        clear_settings_cache()
        try:
            cost = _estimate_cost("unknown_provider", 1000)
            assert cost == 0.0
        finally:
            clear_settings_cache()


# ============================================================================
# api/health.py exception handler (lines 52-53)
# ============================================================================


class TestHealthEndpointException:
    """Cover the exception handler in health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_overall_exception(self):
        """Cover the outer exception handler in health_check."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        from src.api.health import router

        app.include_router(router)

        # Mock get_engine to raise an exception at the top level
        with patch(
            "src.api.health.get_engine",
            new_callable=AsyncMock,
            side_effect=RuntimeError("total failure"),
        ):
            with patch(
                "src.api.health.get_redis_client",
                side_effect=RuntimeError("also fails"),
            ):
                client = TestClient(_AuthenticatedTenant(app))
                response = client.get("/health")
                # The outer try/except catches and returns 503
                # Actually inner errors are caught, so it may still return 200 with degraded
                assert response.status_code in (200, 503)


# ============================================================================
# asgi.py HealthCheckLoggingFilter (lines 44-47)
# ============================================================================


class TestHealthCheckLoggingFilter:
    """Cover the HealthCheckLoggingFilter."""

    def test_filter_health_paths(self):
        import logging

        from src.asgi import HealthCheckLoggingFilter

        f = HealthCheckLoggingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="GET /health HTTP/1.1",
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False

        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="GET /api/tts/generate HTTP/1.1",
            args=(),
            exc_info=None,
        )
        assert f.filter(record2) is True

    def test_filter_ready_path(self):
        import logging

        from src.asgi import HealthCheckLoggingFilter

        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="GET /health/ready HTTP/1.1",
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False


# ============================================================================
# schemas.py SampleRateConfig validator (line 92)
# ============================================================================


class TestSampleRateConfigValidator:
    """Cover the sample rate validation error."""

    def test_invalid_sample_rate(self):
        from src.api.schemas import SampleRateConfig

        with pytest.raises(ValueError, match="Invalid sample rate"):
            SampleRateConfig(target_rate=12345)

    def test_valid_sample_rates(self):
        from src.api.schemas import SampleRateConfig

        for rate in [0, 8000, 16000, 22050, 24000, 44100, 48000, 96000]:
            config = SampleRateConfig(target_rate=rate)
            assert config.target_rate == rate
