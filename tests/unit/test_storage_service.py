"""Unit tests for the storage service domain wrapper.

``StorageService`` is now a thin wrapper over
:class:`w2v_common.storage.StorageService`; these tests mock the facade seam
(``svc._storage``) rather than the raw ``miniopy-async`` client. The wrapper's
own responsibilities — key convention, the local-backend traversal guard, and
delegation to ``delete_object`` / ``presign_get`` (incl. host-rewrite) — are
exercised directly.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from w2v_common.config import StorageSettings as CommonStorageSettings
from w2v_common.storage import StorageService as FacadeStorageService

from src.services.storage_service import StorageService


def _local_service(local_path: str | Path) -> StorageService:
    """A StorageService whose facade is a local backend rooted at ``local_path``."""
    svc = StorageService()
    svc._local_path = Path(local_path)
    svc._storage = FacadeStorageService(
        CommonStorageSettings(backend="local", local_path=str(local_path))
    )
    return svc


def _minio_service(**overrides) -> StorageService:
    """A StorageService whose facade is a minio backend (client injected by caller)."""
    settings = CommonStorageSettings(
        backend="minio",
        minio_endpoint="minio.test:9000",
        minio_bucket="test-bucket",
        **overrides,
    )
    svc = StorageService()
    svc._storage = FacadeStorageService(settings)
    return svc


class TestStorageServiceInit:
    def test_init_reads_settings(self):
        service = StorageService()
        assert service._backend == "local"
        assert isinstance(service._local_path, Path)

    def test_backend_delegates_to_facade(self):
        svc = _minio_service()
        assert svc._backend == "minio"


class TestStorageServiceLocal:
    @pytest.mark.asyncio
    async def test_initialize_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "audio"
            service = _local_service(target)
            await service.initialize()
            assert target.exists()

    @pytest.mark.asyncio
    async def test_store_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            path = await service.store("test-job-123", b"audio data", "mp3")
            # Key convention: audio/{job}.{fmt}, returned as an absolute path.
            assert "audio/test-job-123.mp3" in path
            assert Path(path).exists()
            assert Path(path).read_bytes() == b"audio data"

    @pytest.mark.asyncio
    async def test_retrieve_local_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            file_path = Path(tmpdir) / "test.mp3"
            file_path.write_bytes(b"audio")
            data = await service.retrieve(str(file_path))
            assert data == b"audio"

    @pytest.mark.asyncio
    async def test_retrieve_local_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            data = await service.retrieve(str(Path(tmpdir) / "nope.mp3"))
            assert data is None

    @pytest.mark.asyncio
    async def test_delete_local_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            file_path = Path(tmpdir) / "test.mp3"
            file_path.write_bytes(b"audio")
            result = await service.delete(str(file_path))
            assert result is True
            assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_delete_local_missing_is_idempotent(self):
        # delete_object treats a missing object as success (idempotent).
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            result = await service.delete(str(Path(tmpdir) / "nope.mp3"))
            assert result is True

    @pytest.mark.asyncio
    async def test_check_health_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            assert await service.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_local_missing_path(self):
        service = _local_service("/nonexistent/path/that/doesnt/exist")
        assert await service.check_health() is False

    @pytest.mark.asyncio
    async def test_presigned_url_local_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            assert await service.get_presigned_url("some/path.mp3") is None


class TestStorageServiceUnsupportedBackend:
    """An unknown backend degrades through the facade exactly as before."""

    @pytest.mark.asyncio
    async def test_store_unsupported_backend(self):
        service = StorageService()
        service._storage._backend = "unsupported"
        with pytest.raises(ValueError, match="Unsupported backend"):
            await service.store("test", b"data", "mp3")

    @pytest.mark.asyncio
    async def test_retrieve_unsupported_backend(self):
        service = StorageService()
        service._storage._backend = "unsupported"
        assert await service.retrieve("path") is None

    @pytest.mark.asyncio
    async def test_delete_unsupported_backend(self):
        service = StorageService()
        service._storage._backend = "unsupported"
        assert await service.delete("path") is False

    @pytest.mark.asyncio
    async def test_check_health_unsupported_backend(self):
        service = StorageService()
        service._storage._backend = "unsupported"
        assert await service.check_health() is False

    @pytest.mark.asyncio
    async def test_initialize_unsupported_backend(self):
        service = StorageService()
        service._storage._backend = "unsupported"
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            await service.initialize()


class TestStorageServiceMinio:
    @pytest.mark.asyncio
    async def test_store_minio(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        service._storage._minio_client = mock_minio

        path = await service.store("job-123", b"audio data", "mp3")
        assert path == "audio/job-123.mp3"
        mock_minio.put_object.assert_called_once()
        # mp3 → audio/mpeg content-type
        assert mock_minio.put_object.call_args.kwargs["content_type"] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_store_minio_non_mp3_content_type(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        service._storage._minio_client = mock_minio

        await service.store("job-9", b"data", "wav")
        assert mock_minio.put_object.call_args.kwargs["content_type"] == "audio/wav"

    @pytest.mark.asyncio
    async def test_retrieve_minio(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_response = AsyncMock()
        mock_response.read = AsyncMock(return_value=b"audio data")
        mock_response.close = MagicMock()  # close() is called synchronously
        mock_minio.get_object = AsyncMock(return_value=mock_response)
        service._storage._minio_client = mock_minio

        data = await service.retrieve("audio/job-123.mp3")
        assert data == b"audio data"
        mock_response.close.assert_called_once()
        mock_response.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_minio_error(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_minio.get_object = AsyncMock(side_effect=OSError("not found"))
        service._storage._minio_client = mock_minio

        assert await service.retrieve("missing.mp3") is None

    @pytest.mark.asyncio
    async def test_retrieve_minio_read_error_releases_connection(self):
        """If response.read() throws, the connection must still be closed/released."""
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_response = AsyncMock()
        mock_response.read = AsyncMock(side_effect=OSError("read failed"))
        mock_response.close = MagicMock()
        mock_minio.get_object = AsyncMock(return_value=mock_response)
        service._storage._minio_client = mock_minio

        assert await service.retrieve("audio/job-123.mp3") is None
        mock_response.close.assert_called_once()
        mock_response.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_minio(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        service._storage._minio_client = mock_minio

        result = await service.delete("audio/job.mp3")
        assert result is True
        mock_minio.remove_object.assert_called_once_with("test-bucket", "audio/job.mp3")

    @pytest.mark.asyncio
    async def test_delete_minio_error(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_minio.remove_object = AsyncMock(side_effect=OSError("delete error"))
        service._storage._minio_client = mock_minio

        assert await service.delete("audio/job.mp3") is False

    @pytest.mark.asyncio
    async def test_check_health_minio(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_minio.bucket_exists = AsyncMock(return_value=True)
        service._storage._minio_client = mock_minio

        assert await service.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_minio_error(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_minio.bucket_exists = AsyncMock(side_effect=ConnectionError("conn error"))
        service._storage._minio_client = mock_minio

        assert await service.check_health() is False

    @pytest.mark.asyncio
    async def test_presigned_url_minio(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_minio.presigned_get_object = AsyncMock(return_value="https://minio/test")
        service._storage._minio_client = mock_minio

        result = await service.get_presigned_url("audio/test.mp3")
        assert result == "https://minio/test"

    @pytest.mark.asyncio
    async def test_presigned_url_minio_error(self):
        service = _minio_service()
        mock_minio = AsyncMock()
        mock_minio.presigned_get_object = AsyncMock(side_effect=OSError("error"))
        service._storage._minio_client = mock_minio

        assert await service.get_presigned_url("audio/test.mp3") is None

    @pytest.mark.asyncio
    async def test_presigned_url_host_rewrite(self):
        """External endpoint must rewrite the presigned URL host (facade-provided)."""
        service = _minio_service(minio_external_endpoint="https://minio.example.com")
        mock_minio = AsyncMock()
        mock_minio.presigned_get_object = AsyncMock(
            return_value="http://minio.test:9000/test-bucket/audio/x.mp3?sig=abc"
        )
        service._storage._minio_client = mock_minio

        url = await service.get_presigned_url("audio/x.mp3")
        assert url is not None
        assert url.startswith("https://minio.example.com/")
        assert "minio.test" not in url
        assert "sig=abc" in url


class TestDualBackendConstruction:
    """Regression: a minio-backed service must construct + initialize without a
    live MinIO (the facade's lazy client, mocked here)."""

    @pytest.mark.asyncio
    async def test_minio_initialize_does_not_raise(self, monkeypatch):
        import miniopy_async

        fake_client = AsyncMock()
        fake_client.bucket_exists = AsyncMock(return_value=True)
        monkeypatch.setattr(miniopy_async, "Minio", MagicMock(return_value=fake_client))

        service = _minio_service(
            minio_access_key="ak",
            minio_secret_key="sk",  # pragma: allowlist secret
        )
        assert service._backend == "minio"
        # Must resolve the lazy miniopy client and not raise.
        await service.initialize()
        fake_client.bucket_exists.assert_awaited_once()


class TestStorageServicePathConfinement:
    """Defense-in-depth: a tampered DB row pointing outside the storage root
    must not be honoured by ``retrieve``/``delete`` (local backend)."""

    @pytest.mark.asyncio
    async def test_retrieve_rejects_absolute_outside_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            with tempfile.NamedTemporaryFile(delete=False) as target:
                target.write(b"sensitive")
                target_path = target.name
            try:
                data = await service.retrieve(target_path)
                assert data is None
                assert Path(target_path).read_bytes() == b"sensitive"
            finally:
                Path(target_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_retrieve_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "audio"
            root.mkdir()
            service = _local_service(root)
            outside = Path(tmpdir) / "secret.txt"
            outside.write_bytes(b"sensitive")
            data = await service.retrieve("../secret.txt")
            assert data is None
            assert outside.read_bytes() == b"sensitive"

    @pytest.mark.asyncio
    async def test_delete_rejects_absolute_outside_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            with tempfile.NamedTemporaryFile(delete=False) as target:
                target.write(b"sensitive")
                target_path = target.name
            try:
                result = await service.delete(target_path)
                assert result is False
                assert Path(target_path).exists()
            finally:
                Path(target_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_delete_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "audio"
            root.mkdir()
            service = _local_service(root)
            outside = Path(tmpdir) / "secret.txt"
            outside.write_bytes(b"sensitive")
            result = await service.delete("../secret.txt")
            assert result is False
            assert outside.exists()

    @pytest.mark.asyncio
    async def test_store_then_retrieve_roundtrip(self):
        """The path emitted by store() must still be retrievable —
        confinement must not break the legitimate happy path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = _local_service(tmpdir)
            stored_path = await service.store("job-1", b"audio data", "mp3")
            data = await service.retrieve(stored_path)
            assert data == b"audio data"
            assert await service.delete(stored_path) is True
            assert not Path(stored_path).exists()
