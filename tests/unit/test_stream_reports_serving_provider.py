"""``/stream`` must report the provider that SERVED, not the one requested.

The SSE ``metadata`` event is emitted BEFORE generation starts and used to read
``body.provider`` straight off the request — normally the literal ``"auto"``.
``TTSService.generate_stream`` then runs its own fallback loop, so a cloud
provider can fail before its first chunk and a local one serve every byte, and
the terminal ``done`` event never corrected the record.

The buffered ``/generate`` path already does this right: ``_build_audio_response``
emits ``X-Provider`` from ``meta["resolved_provider"]`` — the provider that
actually produced audio — and OMITS the header when it is unknown. These tests
bring ``/stream`` onto that same convention rather than inventing a third one:
the pre-generation event stops claiming to name the serving provider (it is
``requested_provider``, which is what it always was), and the ``done`` event
carries the resolved one.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.providers.base import AudioChunk
from tests.conftest import tenant_uuid


def _sse_events(payload: str) -> list[tuple[str, dict]]:
    """Parse an SSE payload into ``[(event_name, data_dict), ...]``."""
    events: list[tuple[str, dict]] = []
    for block in payload.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((name, json.loads(data)))
    return events


async def _drive_stream(body, chunks: list[AudioChunk]) -> list[tuple[str, dict]]:
    """Call ``stream_generate`` with a fake service yielding *chunks*; parse the SSE."""
    from src.api.tts import stream_generate

    request = MagicMock()
    request.state.tenant_id = tenant_uuid("tenant-test")
    request.app.state.provider_registry = MagicMock()

    async def fake_stream(**_kwargs):
        for chunk in chunks:
            yield chunk

    tts = MagicMock()
    tts.generate_stream = fake_stream
    request.app.state.tts_service = tts

    db = AsyncMock()
    response = await stream_generate(body, request, db)
    parts: list[str] = [
        part if isinstance(part, str) else bytes(part).decode()
        async for part in response.body_iterator
    ]
    payload = "".join(parts)
    return _sse_events(payload)


class TestStreamReportsTheServingProvider:
    @pytest.mark.asyncio
    async def test_done_event_names_the_provider_that_actually_served(self):
        """``auto`` requested, kokoro served after a fallback — ``done`` says kokoro."""
        from src.api.schemas import TTSStreamRequest

        body = TTSStreamRequest(text="Hello", provider="auto")
        events = await _drive_stream(
            body,
            [
                AudioChunk(data=b"\xff\xfb\x90aa", chunk_index=0),
                AudioChunk(
                    data=b"", chunk_index=1, is_final=True, total_bytes=4, provider="kokoro"
                ),
            ],
        )

        done = dict(events)["done"]
        assert done["provider"] == "kokoro", (
            "the done event must name the SERVING provider — /generate already "
            "reports it as X-Provider after the same failover"
        )

    @pytest.mark.asyncio
    async def test_metadata_event_does_not_claim_auto_is_the_serving_provider(self):
        """The pre-generation event cannot know who will serve, so it must not claim to."""
        from src.api.schemas import TTSStreamRequest

        body = TTSStreamRequest(text="Hello", provider="auto")
        events = await _drive_stream(
            body,
            [
                AudioChunk(data=b"\xff\xfb\x90aa", chunk_index=0),
                AudioChunk(
                    data=b"", chunk_index=1, is_final=True, total_bytes=4, provider="kokoro"
                ),
            ],
        )

        meta = dict(events)["metadata"]
        assert meta.get("provider") != "auto"
        assert meta["requested_provider"] == "auto"

    @pytest.mark.asyncio
    async def test_unknown_serving_provider_omits_the_claim(self):
        """No resolved provider on the final chunk ⇒ omit, never substitute the request."""
        from src.api.schemas import TTSStreamRequest

        body = TTSStreamRequest(text="Hello", provider="auto")
        events = await _drive_stream(
            body,
            [
                AudioChunk(data=b"\xff\xfb\x90aa", chunk_index=0),
                AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=4),
            ],
        )

        done = dict(events)["done"]
        assert "provider" not in done
        # The audio itself is unaffected.
        assert done["total_bytes"] == 4

    @pytest.mark.asyncio
    async def test_pinned_provider_is_still_reported(self):
        """Positive control: a pinned provider that serves is reported as itself."""
        from src.api.schemas import TTSStreamRequest

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        events = await _drive_stream(
            body,
            [
                AudioChunk(data=b"\xff\xfb\x90aa", chunk_index=0),
                AudioChunk(
                    data=b"", chunk_index=1, is_final=True, total_bytes=4, provider="kokoro"
                ),
            ],
        )

        assert dict(events)["metadata"]["requested_provider"] == "kokoro"
        assert dict(events)["done"]["provider"] == "kokoro"


class TestGenerateStreamStampsTheServingProvider:
    @pytest.mark.asyncio
    async def test_final_chunk_carries_the_provider_that_served(self):
        """The service layer is what knows who served; it must stamp the sentinel.

        Without this the route has nothing truthful to report — the whole
        fallback decision lives inside ``generate_stream``.
        """
        from src.providers.base import TTSProvider
        from src.services.tts_service import TTSService

        svc = TTSService.__new__(TTSService)

        class _Provider(TTSProvider):
            name = "kokoro"
            display_name = "Kokoro"
            provider_type = "local"
            max_text_length = 5000

            async def generate(self, text, voice, speed=1.0, output_format="mp3", **kwargs):
                raise NotImplementedError

            async def generate_stream(self, text, voice, speed=1.0, output_format="mp3", **kwargs):
                yield AudioChunk(data=b"\xff\xfb\x90aa", chunk_index=0)
                yield AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=4)

            async def list_voices(self):
                return []

            def get_default_voice(self):
                return "af_bella"

            async def is_available(self):
                return True

        provider = _Provider()
        svc._resolve_provider_chain = AsyncMock(return_value=(provider, "openai", ["kokoro"]))
        svc._stream_attempt_order = MagicMock(return_value=["kokoro"])
        svc._open_stream_candidate = AsyncMock(return_value=(provider, "af_bella", "Hello"))
        svc._record_health = MagicMock()
        svc._meter_attempt_once = AsyncMock()
        svc._meter_abandoned_stream = AsyncMock()
        svc._generation_semaphore = _NullSem()
        svc._provider_semaphores = {}

        chunks = [
            c
            async for c in svc.generate_stream(
                text="Hello", provider="auto", tenant_id="t", db=None, record_usage=False
            )
        ]
        final = [c for c in chunks if c.is_final]
        assert final, "the final sentinel must reach the caller"
        assert final[0].provider == "kokoro"


class _NullSem:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False
