"""Unit tests for TTS API streaming and batch endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import tenant_uuid


class TestStreamGenerate:
    @pytest.mark.asyncio
    async def test_stream_unknown_provider_404(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate

        body = TTSStreamRequest(text="Hello", provider="nonexistent")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        mock_registry.get_provider.side_effect = KeyError("not registered")
        request.app.state.provider_registry = mock_registry
        request.app.state.tts_service = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await stream_generate(body, request)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_returns_streaming_response(self):
        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry

        mock_tts = AsyncMock()
        request.app.state.tts_service = mock_tts

        response = await stream_generate(body, request)
        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"


class TestStreamGenerateRaw:
    @pytest.mark.asyncio
    async def test_raw_stream_unknown_provider_404(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate_raw

        body = TTSStreamRequest(text="Hello", provider="nonexistent")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        mock_registry.get_provider.side_effect = KeyError("not registered")
        request.app.state.provider_registry = mock_registry
        request.app.state.tts_service = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await stream_generate_raw(body, request)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raw_stream_mp3_content_type(self):
        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate_raw

        body = TTSStreamRequest(text="Hello", provider="kokoro", output_format="mp3")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry
        request.app.state.tts_service = MagicMock()

        response = await stream_generate_raw(body, request)
        assert response.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_raw_stream_wav_content_type(self):
        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate_raw

        body = TTSStreamRequest(text="Hello", provider="kokoro", output_format="wav")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry
        request.app.state.tts_service = MagicMock()

        response = await stream_generate_raw(body, request)
        assert response.media_type == "audio/wav"


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_batch_unknown_provider_404(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(
            items=[TTSBatchItem(id="1", text="Hello")],
            provider="nonexistent",
        )
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        mock_registry = MagicMock()
        mock_registry.get_provider.side_effect = KeyError("not registered")
        request.app.state.provider_registry = mock_registry
        request.app.state.tts_service = MagicMock()
        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate_batch(body, request, db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_creates_jobs(self):
        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(
            items=[
                TTSBatchItem(id="1", text="Hello"),
                TTSBatchItem(id="2", text="World"),
            ],
            provider="kokoro",
        )
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(side_effect=["job-1", "job-2"])
        request.app.state.tts_service = mock_tts
        db = AsyncMock()

        response = await generate_batch(body, request, db)
        assert response.status == "pending"
        assert len(response.job_ids) == 2
        assert response.batch_id is not None

    @pytest.mark.asyncio
    async def test_batch_auto_provider(self):
        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(
            items=[TTSBatchItem(id="1", text="Hello")],
            provider="auto",
        )
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(return_value="job-1")
        request.app.state.tts_service = mock_tts
        db = AsyncMock()

        response = await generate_batch(body, request, db)
        assert response.status == "pending"


class TestDownloadBatchAudio:
    @pytest.mark.asyncio
    async def test_batch_not_found(self):
        from fastapi import HTTPException

        from src.api.tts import download_batch_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[])
            with pytest.raises(HTTPException) as exc_info:
                await download_batch_audio("nonexistent", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_no_concat(self):
        from fastapi import HTTPException

        from src.api.tts import download_batch_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()
        mock_job = MagicMock()
        mock_job.concatenated_audio_path = None

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
            with pytest.raises(HTTPException) as exc_info:
                await download_batch_audio("batch-1", request, db)
            assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_batch_audio_not_found(self):
        from fastapi import HTTPException

        from src.api.tts import download_batch_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.retrieve = AsyncMock(return_value=None)
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.concatenated_audio_path = "/path/concat.mp3"
        mock_job.output_format = "mp3"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
            with pytest.raises(HTTPException) as exc_info:
                await download_batch_audio("batch-1", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_audio_success(self):
        from src.api.tts import download_batch_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.retrieve = AsyncMock(return_value=b"concat audio")
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.concatenated_audio_path = "/path/concat.mp3"
        mock_job.output_format = "mp3"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
            result = await download_batch_audio("batch-1", request, db)
            assert result.body == b"concat audio"
            assert result.media_type == "audio/mpeg"


class TestConcatenateBatchBackground:
    @pytest.mark.asyncio
    async def test_concatenate_no_jobs(self):
        from src.api.tts import _concatenate_batch_background

        tts = AsyncMock()
        mock_db = AsyncMock()

        with patch("src.api.tts.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.api.tts.JobManager") as mock_jm:
                mock_jm.get_batch_jobs = AsyncMock(return_value=[])
                await _concatenate_batch_background(
                    tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
                )

        # Should return without calling concatenate
        tts.concatenate_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_concatenate_all_completed(self):
        from src.api.tts import _concatenate_batch_background

        tts = AsyncMock()
        mock_db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"

        with patch("src.api.tts.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.api.tts.JobManager") as mock_jm:
                mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await _concatenate_batch_background(
                        tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
                    )

        tts.concatenate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_concatenate_error_handled(self):
        from src.api.tts import _concatenate_batch_background

        tts = AsyncMock()
        tts.concatenate_batch = AsyncMock(side_effect=RuntimeError("concat error"))
        mock_db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"

        with patch("src.api.tts.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.api.tts.JobManager") as mock_jm:
                mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    # Should not raise
                    await _concatenate_batch_background(
                        tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
                    )
