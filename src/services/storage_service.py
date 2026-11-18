"""Audio file storage — thin domain wrapper over ``w2v_common.storage``.

The object I/O (local filesystem / MinIO client, presign + host-rewrite,
bucket bootstrap) is delegated to the fleet-shared
:class:`w2v_common.storage.StorageService` facade. This wrapper keeps only the
tts_service-specific bits the facade deliberately does not carry:

* the audio key convention ``audio/{job_id}.{fmt}`` and ``audio/mpeg``
  content-type;
* a local-backend path-traversal guard (``_safe_local_path``) — the facade
  reads local objects by key/path without confining them to the storage root,
  so a tampered/stale DB row pointing outside the root must be rejected here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from w2v_common.storage import StorageService as _StorageFacade

from src.config import get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Manages audio file storage (local or S3/MinIO) via the shared facade."""

    def __init__(self) -> None:
        settings = get_settings()
        # Kept locally for the traversal guard; the facade owns the live client.
        self._local_path = Path(settings.storage.local_path)
        self._storage = _StorageFacade(settings.storage)

    @property
    def _backend(self) -> str:
        return self._storage.backend

    async def initialize(self) -> None:
        await self._storage.initialize()

    async def store(self, job_id: str, audio_data: bytes, fmt: str) -> str:
        """Store audio data and return the storage reference.

        The reference (MinIO key or absolute local path) round-trips back
        through :meth:`retrieve` / :meth:`delete` / :meth:`get_presigned_url`.
        """
        object_key = f"audio/{job_id}.{fmt}"
        content_type = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
        return await self._storage.put_bytes(object_key, audio_data, content_type=content_type)

    async def retrieve(self, path: str) -> bytes | None:
        """Retrieve audio data for a storage reference."""
        if self._backend == "local":
            safe = self._safe_local_path(path)
            if safe is None:
                return None
            return await self._storage.get_bytes(str(safe))
        return await self._storage.get_bytes(path)

    async def delete(self, path: str) -> bool:
        """Delete a single audio object from the configured backend.

        Delegates to the facade's idempotent ``delete_object`` (a missing
        object is treated as success). For the local backend the traversal
        guard is applied first — a reference outside the storage root is
        rejected (``False``) rather than honoured.
        """
        if self._backend == "local":
            safe = self._safe_local_path(path)
            if safe is None:
                return False
            return await self._storage.delete_object(str(safe))
        return await self._storage.delete_object(path)

    async def get_presigned_url(self, path: str, expires: int | None = None) -> str | None:
        """Generate a presigned URL for direct client download (MinIO only).

        When ``STORAGE_MINIO_EXTERNAL_ENDPOINT`` is configured the facade
        rewrites the internal host so clients outside Docker/k8s can reach
        MinIO directly.
        """
        return await self._storage.presign_get(path, expires=expires)

    async def check_health(self) -> bool:
        """Check storage backend health by verifying the path or bucket exists."""
        return await self._storage.check_health()

    def _safe_local_path(self, raw: str) -> Path | None:
        """Resolve ``raw`` and reject anything outside ``self._local_path``.

        Defense in depth: ``store`` only emits paths under the configured root,
        so the DB row should always round-trip cleanly. A rejection here means
        either a stale row from a different configuration or a tampered DB entry
        — return ``None`` (treated as "not found" by callers) rather than
        honouring the traversal.
        """
        root = self._local_path.resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._local_path / raw
        try:
            resolved = candidate.resolve()
        except OSError, RuntimeError:
            logger.warning("Could not resolve storage path: %s", raw)
            return None
        if not resolved.is_relative_to(root):
            logger.warning("Rejected path outside storage root: %s", raw)
            return None
        return resolved
