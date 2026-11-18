"""SSML char-count parity: quota bookkeeping must bill the SAME stripped text
the cost bus bills.

Bug (origin/main): on an SSML request ``_process_text`` returns the
tag-stripped text for non-SSML providers, so the cost bus bills the STRIPPED
length (``used_text`` / ``processed_text``) while the quota bookkeeping billed
the RAW ``len(text)`` (SSML tags included). The two char counts diverge on
every SSML request.

These tests drive the two real billing call paths (sync ``generate_sync`` and
the streaming ``_meter_completed_stream``) with spies on the quota recorder and
the cost-bus emitter, then assert the quota char count EQUALS the cost-bus char
count. RED on origin/main (raw 48 != stripped 12); GREEN after the fix.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from src.providers.base import GenerationResult
from src.services.ssml_processor import process_ssml_for_provider
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid

# 48 raw chars; 12 chars once SSML tags are stripped for a non-SSML provider.
SSML_INPUT = '<speak>Hello <break time="500ms"/> world</speak>'
STRIPPED = process_ssml_for_provider(SSML_INPUT, "openai")


def _bare_service() -> TTSService:
    """A TTSService whose __init__ plumbing is bypassed (see test_metering.py)."""
    return TTSService.__new__(TTSService)


def test_stripped_fixture_actually_diverges() -> None:
    """Guard: the fixture must exercise a real raw-vs-stripped gap."""
    assert len(SSML_INPUT) == 48
    assert len(STRIPPED) == 12
    assert len(SSML_INPUT) != len(STRIPPED)


@pytest.mark.asyncio
async def test_stream_quota_charcount_matches_cost_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """_meter_completed_stream: quota chars must equal cost-bus chars."""
    svc = _bare_service()
    record_spy = AsyncMock()
    emit_spy = Mock()
    # The recorder is spied, so it is called regardless of ``db`` — no sentinel
    # session needed (the real recorder's db-None early-return is bypassed).
    monkeypatch.setattr(svc, "_record_stream_usage", record_spy)
    monkeypatch.setattr(svc, "_emit_cost_event", emit_spy)

    await svc._meter_completed_stream(
        None,
        tenant_id=tenant_uuid("t"),
        provider="openai",
        text=SSML_INPUT,  # raw, tags included
        processed_text=STRIPPED,  # what was actually synthesized + billed
        model=None,
        output_format="mp3",
        started_at=0.0,
        record_usage=True,
    )

    quota_chars = record_spy.call_args.kwargs["characters"]
    cost_chars = len(emit_spy.call_args.kwargs["text"])
    assert quota_chars == cost_chars, (
        f"stream quota billed {quota_chars} chars but cost bus billed {cost_chars}"
    )


@pytest.mark.asyncio
async def test_sync_quota_charcount_matches_cost_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_sync: quota chars must equal cost-bus chars on an SSML request."""
    svc = _bare_service()

    result = GenerationResult(
        audio_data=b"\x00\x00",
        format="mp3",
        sample_rate=24000,
        duration_seconds=1.0,
    )

    # Stub every heavy collaborator; leave the real _process_text +
    # _record_sync_usage/_emit_cost_event billing lines under test.
    provider_obj = Mock()
    monkeypatch.setattr(
        svc, "_resolve_provider_chain", AsyncMock(return_value=(provider_obj, "openai", []))
    )
    monkeypatch.setattr(svc, "_resolve_voice", Mock(return_value="voice"))
    monkeypatch.setattr(svc, "_detect_language_and_voice", AsyncMock(return_value="voice"))
    monkeypatch.setattr(svc, "_try_cache_hit", AsyncMock(return_value=(None, None)))
    # _generate_with_fallback returns used_text = the STRIPPED text the provider
    # actually synthesized (primary succeeds on the processed text).
    monkeypatch.setattr(
        svc,
        "_generate_with_fallback",
        AsyncMock(return_value=(result, "openai", "voice", STRIPPED)),
    )
    monkeypatch.setattr(svc, "_measure_audio_metrics", AsyncMock())
    monkeypatch.setattr(svc, "_persist_cache", AsyncMock())

    record_spy = AsyncMock()
    emit_spy = Mock()
    monkeypatch.setattr(svc, "_record_sync_usage", record_spy)
    monkeypatch.setattr(svc, "_emit_cost_event", emit_spy)

    await svc.generate_sync(
        tenant_id=tenant_uuid("tenant-test"),
        text=SSML_INPUT,
        provider="openai",
        ssml=True,
        db=None,  # skip quota pre-check; recorder is spied anyway
        record_usage=True,
        emit_cost=True,
    )

    quota_chars = record_spy.call_args.kwargs["characters"]
    cost_chars = len(emit_spy.call_args.kwargs["text"])
    assert quota_chars == cost_chars, (
        f"sync quota billed {quota_chars} chars but cost bus billed {cost_chars}"
    )


@pytest.mark.asyncio
async def test_sync_cache_hit_quota_charcount_is_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_sync cache-HIT branch: quota chars must equal the STRIPPED count.

    A cache hit still bills the tenant's character quota (comment at the hit
    branch), but does not re-emit a cost event. The billed count must match the
    cache-MISS / cost-bus count (STRIPPED = 12), NOT the raw SSML length (48).
    RED before the fix (records len(text) = 48); GREEN after (len(processed_text)
    = 12).
    """
    svc = _bare_service()

    cached = GenerationResult(
        audio_data=b"\x00\x00",
        format="mp3",
        sample_rate=24000,
        duration_seconds=1.0,
    )

    provider_obj = Mock()
    monkeypatch.setattr(
        svc, "_resolve_provider_chain", AsyncMock(return_value=(provider_obj, "openai", []))
    )
    monkeypatch.setattr(svc, "_resolve_voice", Mock(return_value="voice"))
    monkeypatch.setattr(svc, "_detect_language_and_voice", AsyncMock(return_value="voice"))
    # Prime the cache so generate_sync takes the real cache-HIT branch and never
    # reaches _generate_with_fallback.
    monkeypatch.setattr(svc, "_try_cache_hit", AsyncMock(return_value=(cached, "cache-key")))
    fallback_spy = AsyncMock()
    monkeypatch.setattr(svc, "_generate_with_fallback", fallback_spy)

    record_spy = AsyncMock()
    monkeypatch.setattr(svc, "_record_sync_usage", record_spy)

    await svc.generate_sync(
        tenant_id=tenant_uuid("tenant-test"),
        text=SSML_INPUT,
        provider="openai",
        ssml=True,
        db=None,  # skip quota pre-check; recorder is spied anyway
        record_usage=True,
        emit_cost=True,
    )

    fallback_spy.assert_not_awaited()  # proves the real cache-hit path was taken
    quota_chars = record_spy.call_args.kwargs["characters"]
    assert quota_chars == len(STRIPPED), (
        f"cache-hit quota billed {quota_chars} chars but the stripped/miss count "
        f"is {len(STRIPPED)} (raw SSML is {len(SSML_INPUT)})"
    )


@pytest.mark.asyncio
async def test_async_job_quota_and_cost_bus_bill_stripped_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_job (async job path): both counters must bill the STRIPPED text.

    Same defect class, third path. ``job.text_length`` is ``len(raw)`` captured
    at job-creation time — before any provider is resolved — so on an SSML job
    the async path billed the RAW length on BOTH the usage row and the cost
    bus. Those two agree with each other but disagree with what the identical
    synchronous request charges, so the same input costs a different amount
    depending on whether the caller asked for sync or async.

    RED on origin/main (48 raw); GREEN after (12 stripped, matching sync).
    """
    svc = _bare_service()

    result = GenerationResult(
        audio_data=b"\x00\x00",
        format="mp3",
        sample_rate=24000,
        duration_seconds=1.0,
    )

    job = Mock()
    job.text = SSML_INPUT
    job.text_length = len(SSML_INPUT)  # raw, as JobManager.create_job stores it
    job.provider = "auto"
    job.voice = "voice"
    job.speed = 1.0
    job.output_format = "mp3"
    job.model = None
    job.routing_strategy = None

    # generate_sync is exercised for real elsewhere; here it is stubbed to
    # return the meta contract it now publishes, so this test pins the
    # process_job side of the wiring.
    monkeypatch.setattr(
        svc,
        "generate_sync",
        AsyncMock(
            return_value=(
                result,
                {
                    "resolved_provider": "openai",
                    "billed_text": STRIPPED,
                    "estimated_cost": 0.0,
                    "latency_ms": 1.0,
                    "cache_hit": False,
                },
            )
        ),
    )
    storage = Mock()
    storage.store = AsyncMock(return_value="/audio.mp3")
    svc._storage = storage  # type: ignore[attr-defined]
    monkeypatch.setattr(svc, "_notify", AsyncMock())
    monkeypatch.setattr(svc, "_measure_audio_metrics", AsyncMock())

    emit_spy = Mock()
    monkeypatch.setattr(svc, "_emit_cost_event", emit_spy)

    db = AsyncMock()

    import src.services.tts_service as tts_mod

    jm = Mock()
    jm.get_job = AsyncMock(return_value=job)
    jm.mark_processing = AsyncMock()
    jm.mark_completed = AsyncMock()
    jm.record_usage = AsyncMock()
    monkeypatch.setattr(tts_mod, "JobManager", jm)

    await svc.process_job(db, "job-1", "t")

    quota_chars = jm.record_usage.call_args.kwargs["characters"]
    cost_chars = len(emit_spy.call_args.kwargs["text"])
    assert quota_chars == len(STRIPPED), (
        f"async-job quota billed {quota_chars} chars but the synthesized text is "
        f"{len(STRIPPED)} chars (raw SSML is {len(SSML_INPUT)})"
    )
    assert cost_chars == len(STRIPPED), (
        f"async-job cost bus billed {cost_chars} chars but the synthesized text "
        f"is {len(STRIPPED)} chars (raw SSML is {len(SSML_INPUT)})"
    )
