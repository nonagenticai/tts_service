"""Protocols for dependency injection seams in tts_service.

These Protocols expose narrow, public DI seams so callers (including tests)
can provide alternate implementations without poking at private attributes
or relying on `# type: ignore` suppressions.
"""

from __future__ import annotations

from typing import Protocol


class CacheChecker(Protocol):
    """Callable seam for checking the audio generation cache.

    Returns ``(audio_data, cache_key, cached_duration_seconds)``. ``audio_data``
    is ``None`` on a miss; ``cache_key`` may be empty when caching is disabled;
    ``cached_duration_seconds`` is ``None`` when the entry carries no measured
    duration (see ``TTSService._store_cache``) — never a substituted ``0.0``.
    """

    async def __call__(
        self,
        tenant_id: str,
        text: str,
        provider: str,
        voice: str,
        speed: float,
        output_format: str,
    ) -> tuple[bytes | None, str, float | None]: ...
