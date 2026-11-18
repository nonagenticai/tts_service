"""Regression tests for three defects found in the 2026-07-30 bug hunt.

B1 — ``subprocess.TimeoutExpired`` escaped every handler (it subclasses
     ``SubprocessError``, not ``OSError`` and not the builtin ``TimeoutError``),
     so one hung ffmpeg/ffprobe left a job non-terminal until a restart.
B2 — the synchronous ``/api/tts/generate`` path never recorded ``TTSUsage``, so
     ``check_quota`` (which aggregates those rows) could never see it: a tenant
     posting ``async_mode=false`` passed quota forever while cost events billed.
B3 — the served Content-Type was derived from the REQUESTED format, not the
     produced bytes, so ElevenLabs MP3 was served as ``audio/wav``; the batch
     concatenator wrote MP3 bytes into ``seg_0000.wav`` and stream-copied them.

Plus two billing defects found in the follow-up hunt:

B4 — the streaming paths (``generate_stream`` + ``/api/tts/stream[/raw]``)
     enforced no quota, wrote no ``TTSUsage`` row, and emitted no cost event, so
     a tenant 429'd on ``/generate`` could stream unlimited, unattributed,
     paid-provider audio for free.
B5 — the async job path emitted the billing cost event inside ``generate_sync``,
     BEFORE ``_storage.store`` / ``mark_completed``; a storage/DB failure after a
     paid synthesis then billed the tenant on the cost bus while the job row read
     ``failed`` with $0 usage — a silent overbill.
"""

from __future__ import annotations

import subprocess  # nosec B404 -- TimeoutExpired is constructed, no process spawned
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.providers.base import AudioChunk, GenerationResult
from src.services.tts_service import TTSService
from tests.conftest import tenant_uuid


def _timeout_expired() -> subprocess.TimeoutExpired:
    """The exact exception a hung ffprobe raises."""
    return subprocess.TimeoutExpired(cmd=["ffprobe", "-i", "input.mp3"], timeout=30)


def _make_service() -> TTSService:
    registry = MagicMock()
    storage = MagicMock()
    return TTSService(registry=registry, storage=storage)


def _stub_provider(fmt: str = "mp3") -> AsyncMock:
    provider = AsyncMock()
    provider.is_available = AsyncMock(return_value=True)
    provider.get_default_voice = MagicMock(return_value="af_heart")
    provider.sample_rate = 24000
    provider.generate = AsyncMock(
        return_value=GenerationResult(
            audio_data=b"audio", format=fmt, sample_rate=24000, duration_seconds=1.0
        )
    )
    return provider


# ---------------------------------------------------------------------------
# B1 — TimeoutExpired must not escape / must still reach a terminal state
# ---------------------------------------------------------------------------


class TestSubprocessTimeoutIsHandled:
    def test_timeout_expired_is_not_oserror_or_timeouterror(self):
        """The premise of the bug: the old except tuples could not catch it."""
        exc = _timeout_expired()
        assert not isinstance(exc, OSError)
        assert not isinstance(exc, TimeoutError)
        assert isinstance(exc, subprocess.SubprocessError)

    @pytest.mark.asyncio
    async def test_measure_audio_metrics_swallows_subprocess_timeout(self):
        """``_measure_audio_metrics`` documents "Never raises" — make it true."""
        service = _make_service()
        result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        meta: dict = {}

        processor = MagicMock()
        processor.measure_audio = MagicMock(side_effect=_timeout_expired())

        with patch("src.utils.post_processor.AudioPostProcessor", return_value=processor):
            await service._measure_audio_metrics(result, meta)

        # Swallowed, and no half-written metrics left behind.
        assert "audio_metrics" not in meta

    @pytest.mark.asyncio
    async def test_generate_sync_survives_a_hung_prober(self):
        """A hung ffprobe must not fail an otherwise successful generation."""
        service = _make_service()
        cast(Any, service._registry).get_provider.return_value = _stub_provider()

        processor = MagicMock()
        processor.measure_audio = MagicMock(side_effect=_timeout_expired())

        with patch("src.utils.post_processor.AudioPostProcessor", return_value=processor):
            result, meta = await service.generate_sync(
                tenant_id=tenant_uuid("tenant-test"),
                text="Hello",
                provider="kokoro",
                voice="af_heart",
            )

        assert result.audio_data == b"audio"
        assert meta["resolved_provider"] == "kokoro"

    @pytest.mark.asyncio
    async def test_process_job_marks_failed_on_subprocess_timeout(self):
        """A TimeoutExpired out of generation still terminalises the job row."""
        service = _make_service()
        db = AsyncMock()

        job = MagicMock()
        job.text = "Hello"
        job.provider = "kokoro"
        job.voice = "af_heart"
        job.speed = 1.0
        job.output_format = "mp3"
        job.model = None
        job.text_length = 5
        job.routing_strategy = "balanced"

        with (
            patch.object(service, "generate_sync", AsyncMock(side_effect=_timeout_expired())),
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            with pytest.raises(subprocess.TimeoutExpired):
                await service.process_job(db, "job-1", tenant_uuid("t1"))

        # The job is terminal (failed), not stranded in "processing".
        mock_jm.mark_failed.assert_awaited_once()
        assert mock_jm.mark_failed.call_args[0][1] == "job-1"
        mock_jm.record_usage.assert_awaited_once()
        assert mock_jm.record_usage.call_args[1]["is_error"] is True

    @pytest.mark.asyncio
    async def test_process_job_background_marks_failed_on_subprocess_timeout(self):
        """The API-layer background wrapper also catches it (no unhandled task)."""
        from src.api.tts import _process_job_background

        tts = AsyncMock()
        tts.process_job = AsyncMock(side_effect=_timeout_expired())
        mock_db = AsyncMock()

        with (
            patch("src.api.tts.get_db_session") as mock_get_db,
            patch("src.api.tts._mark_job_failed", AsyncMock()) as mock_mark,
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_background(tts, "job-1", tenant_uuid("tenant-1"))

        mock_mark.assert_awaited_once()
        assert mock_mark.call_args[0][0] == "job-1"


# ---------------------------------------------------------------------------
# B2 — the sync path must record usage (otherwise quota is unenforceable)
# ---------------------------------------------------------------------------


class TestSyncPathRecordsUsage:
    @pytest.mark.asyncio
    async def test_sync_generate_writes_a_tts_usage_row(self, db_session):
        """A real row must land in tts_usage — that table IS the quota ledger."""
        from sqlalchemy import select

        from src.database.models import TTSUsage

        service = _make_service()
        cast(Any, service._registry).get_provider.return_value = _stub_provider()

        await service.generate_sync(
            text="Hello world",
            provider="kokoro",
            voice="af_heart",
            tenant_id=tenant_uuid("tenant-quota"),
            db=db_session,
        )

        rows = (
            (
                await db_session.execute(
                    select(TTSUsage).where(TTSUsage.tenant_id == tenant_uuid("tenant-quota"))
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].provider == "kokoro"
        assert rows[0].total_characters == len("Hello world")
        assert rows[0].request_count == 1

    @pytest.mark.asyncio
    async def test_sync_generate_usage_accumulates_across_requests(self, db_session):
        """Two sync calls aggregate — so quota actually moves toward the limit."""
        from sqlalchemy import select

        from src.database.models import TTSUsage

        service = _make_service()
        cast(Any, service._registry).get_provider.return_value = _stub_provider()

        for _ in range(2):
            await service.generate_sync(
                text="abcde",
                provider="kokoro",
                voice="af_heart",
                tenant_id=tenant_uuid("tenant-accum"),
                db=db_session,
                no_cache=True,
            )

        row = (
            await db_session.execute(
                select(TTSUsage).where(TTSUsage.tenant_id == tenant_uuid("tenant-accum"))
            )
        ).scalar_one()
        assert row.request_count == 2
        assert row.total_characters == 10

    @pytest.mark.asyncio
    async def test_sync_generate_without_db_does_not_record(self):
        """No session ⇒ nothing to aggregate against; must not blow up."""
        service = _make_service()
        cast(Any, service._registry).get_provider.return_value = _stub_provider()

        with patch("src.services.tts_service.JobManager") as mock_jm:
            mock_jm.record_usage = AsyncMock()
            await service.generate_sync(
                tenant_id=tenant_uuid("tenant-test"), text="Hi", provider="kokoro", voice="af_heart"
            )

        mock_jm.record_usage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_async_path_does_not_double_count(self):
        """process_job records once; it must suppress generate_sync's own record."""
        service = _make_service()
        storage = AsyncMock()
        storage.store = AsyncMock(return_value="audio/job-1.mp3")
        service._storage = storage
        db = AsyncMock()

        job = MagicMock()
        job.text = "Hello world"
        job.provider = "kokoro"
        job.voice = "af_heart"
        job.speed = 1.0
        job.output_format = "mp3"
        job.model = None
        job.text_length = 11
        job.routing_strategy = "balanced"

        result = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        gen_meta = {"resolved_provider": "kokoro", "estimated_cost": 0.0, "latency_ms": 5}

        with (
            patch.object(
                service, "generate_sync", AsyncMock(return_value=(result, gen_meta))
            ) as mock_gen,
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()
            await service.process_job(db, "job-1", tenant_uuid("t1"))

        assert mock_gen.call_args[1]["record_usage"] is False
        mock_jm.record_usage.assert_awaited_once()


# ---------------------------------------------------------------------------
# B3 — the declared format must follow the bytes, never the request
# ---------------------------------------------------------------------------


class TestFormatHonesty:
    @pytest.mark.asyncio
    async def test_elevenlabs_sends_output_format_and_reports_produced_format(self):
        """A wav request cannot be honoured — the result must SAY mp3, not wav."""
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test-key")  # pragma: allowlist secret

        response = MagicMock()
        response.content = b"\xff\xfb-mp3-bytes"
        response.raise_for_status = MagicMock()
        cast(Any, provider)._client.post = AsyncMock(return_value=response)

        try:
            result = await provider.generate(
                "Hello", voice="rachel", speed=1.0, output_format="wav"
            )
        finally:
            await provider.close()

        # The API is now told what we want...
        posted = cast(Any, provider)._client.post
        assert posted.call_args[1]["params"]["output_format"] == "mp3_44100_128"
        # ...and the result never claims a format it did not produce.
        assert result.format == "mp3"
        assert result.audio_data == b"\xff\xfb-mp3-bytes"

    @pytest.mark.asyncio
    async def test_download_audio_content_type_follows_stored_bytes(self):
        """job.output_format says wav, the stored object is mp3 → audio/mpeg."""
        from src.api.tts import download_audio

        job = MagicMock()
        job.status = "completed"
        job.output_format = "wav"  # what the caller ASKED for
        job.audio_path = "audio/job-9.mp3"  # what was actually PRODUCED
        job.audio_duration_seconds = 2.0

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        storage = AsyncMock()
        storage.retrieve = AsyncMock(return_value=b"\xff\xfb-mp3-bytes")
        request.app.state.storage_service = storage
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=job)
            response = await download_audio("job-9", request, db)

        assert response.media_type == "audio/mpeg"
        assert response.headers["x-audio-format"] == "mp3"
        assert 'filename="job-9.mp3"' in response.headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_download_audio_falls_back_to_request_format_for_legacy_rows(self):
        """A stored path with no extension keeps the old behaviour."""
        from src.api.tts import download_audio

        job = MagicMock()
        job.status = "completed"
        job.output_format = "wav"
        job.audio_path = "audio/legacy-object"
        job.audio_duration_seconds = 1.0

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        storage = AsyncMock()
        storage.retrieve = AsyncMock(return_value=b"RIFF")
        request.app.state.storage_service = storage
        db = AsyncMock()

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=job)
            response = await download_audio("legacy", request, db)

        assert response.media_type == "audio/wav"

    def test_build_audio_response_uses_result_format(self):
        """The sync response follows the produced bytes too."""
        from src.api.tts import _build_audio_response

        result = GenerationResult(
            audio_data=b"\xff\xfb", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        response = _build_audio_response(result, {"resolved_provider": "elevenlabs"}, "elevenlabs")
        assert response.media_type == "audio/mpeg"

    def test_concatenate_names_segments_by_their_real_format(self):
        """MP3 bytes must never be written to seg_XXXX.wav and stream-copied."""
        from src.utils import audio as audio_utils

        seen: dict[str, object] = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            # The concat list file records the segment paths ffmpeg was handed.
            list_path = Path(cmd[cmd.index("-i") + 1])
            seen["filelist"] = list_path.read_text()
            Path(cmd[-1]).write_bytes(b"concatenated")
            return MagicMock(returncode=0, stderr="")

        with patch("src.utils.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("src.utils.audio.subprocess.run", side_effect=_fake_run):
                out = audio_utils.concatenate_audio(
                    [b"mp3-one", b"mp3-two"],
                    output_format="wav",
                    segment_formats=["mp3", "mp3"],
                )

        assert out == b"concatenated"
        # Segments carry their REAL extension...
        assert "seg_0000.mp3" in str(seen["filelist"])
        assert "seg_0001.mp3" in str(seen["filelist"])
        # ...and because they are not already wav, ffmpeg must re-encode.
        cmd = cast(list[str], seen["cmd"])
        assert "copy" not in cmd
        assert "pcm_s16le" in cmd

    def test_concatenate_stream_copies_when_formats_are_uniform(self):
        """The fast path is preserved when every segment already matches."""
        from src.utils import audio as audio_utils

        seen: dict[str, object] = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            list_path = Path(cmd[cmd.index("-i") + 1])
            seen["filelist"] = list_path.read_text()
            Path(cmd[-1]).write_bytes(b"concatenated")
            return MagicMock(returncode=0, stderr="")

        with patch("src.utils.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("src.utils.audio.subprocess.run", side_effect=_fake_run):
                audio_utils.concatenate_audio(
                    [b"a", b"b"], output_format="mp3", segment_formats=["mp3", "mp3"]
                )

        assert "seg_0000.mp3" in str(seen["filelist"])
        assert "copy" in cast(list[str], seen["cmd"])

    def test_concatenate_rejects_misaligned_segment_formats(self):
        from src.utils import audio as audio_utils

        with pytest.raises(ValueError, match="segment_formats"):
            audio_utils.concatenate_audio([b"a", b"b"], segment_formats=["mp3"])


# ---------------------------------------------------------------------------
# B4 — streaming must enforce quota and record usage + cost (no free bypass)
# ---------------------------------------------------------------------------


def _stream_provider() -> AsyncMock:
    """A provider whose generate_stream yields one chunk then a final marker."""
    provider = AsyncMock()
    provider.is_available = AsyncMock(return_value=True)
    provider.get_default_voice = MagicMock(return_value="af_heart")
    provider.list_voices = AsyncMock(return_value=[])
    provider.sample_rate = 24000

    async def _ok_stream(**_kwargs: object):
        yield AudioChunk(data=b"chunk", chunk_index=0)
        yield AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=5)

    provider.generate_stream = _ok_stream
    return provider


class TestStreamBillingAndQuota:
    @pytest.mark.asyncio
    async def test_stream_completion_records_usage_and_emits_cost(self):
        """A completed stream must write a usage row AND bill — same as sync."""
        service = _make_service()
        cast(Any, service._registry).get_provider.return_value = _stream_provider()
        db = AsyncMock()

        with (
            patch("src.services.tts_service.JobManager") as mock_jm,
            patch.object(service, "_emit_cost_event") as mock_emit,
        ):
            mock_jm.record_usage = AsyncMock()
            chunks = [
                chunk
                async for chunk in service.generate_stream(
                    text="Hello",
                    provider="kokoro",
                    tenant_id=tenant_uuid("tenant-stream"),
                    db=db,
                )
            ]

        assert [c.data for c in chunks] == [b"chunk", b""]
        # Usage row: attributed to the serving provider + billed per character.
        mock_jm.record_usage.assert_awaited_once()
        usage = mock_jm.record_usage.call_args[1]
        assert usage["tenant_id"] == tenant_uuid("tenant-stream")
        assert usage["provider"] == "kokoro"
        assert usage["characters"] == len("Hello")
        # Cost event fired for the streamed bytes.
        mock_emit.assert_called_once()
        assert mock_emit.call_args[1]["tenant_id"] == tenant_uuid("tenant-stream")
        assert mock_emit.call_args[1]["provider"] == "kokoro"

    @pytest.mark.asyncio
    async def test_stream_without_db_writes_no_usage_row(self):
        """No session (e.g. unauth dev call) ⇒ nothing to aggregate against."""
        service = _make_service()
        cast(Any, service._registry).get_provider.return_value = _stream_provider()

        with (
            patch("src.services.tts_service.JobManager") as mock_jm,
            patch.object(service, "_emit_cost_event"),
        ):
            mock_jm.record_usage = AsyncMock()
            _ = [
                chunk
                async for chunk in service.generate_stream(
                    tenant_id=tenant_uuid("tenant-test"), text="Hi", provider="kokoro"
                )
            ]

        mock_jm.record_usage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sse_route_rejects_over_quota_before_streaming(self):
        """An over-quota tenant gets a clean 429 and never starts streaming."""
        from fastapi import HTTPException

        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate
        from src.services.quota_service import QuotaExceededError

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.app.state.provider_registry = MagicMock()
        mock_tts = AsyncMock()
        request.app.state.tts_service = mock_tts
        db = AsyncMock()

        over = QuotaExceededError("over", "monthly_request_limit", 10, 10)
        with patch("src.api.tts.check_quota", AsyncMock(side_effect=over)):
            with pytest.raises(HTTPException) as exc_info:
                await stream_generate(body, request, db)

        assert exc_info.value.status_code == 429
        mock_tts.generate_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_raw_route_rejects_over_quota_before_streaming(self):
        """The raw binary route enforces the same pre-stream quota gate."""
        from fastapi import HTTPException

        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate_raw
        from src.services.quota_service import QuotaExceededError

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.app.state.provider_registry = MagicMock()
        mock_tts = AsyncMock()
        request.app.state.tts_service = mock_tts
        db = AsyncMock()

        over = QuotaExceededError("over", "monthly_request_limit", 10, 10)
        with patch("src.api.tts.check_quota", AsyncMock(side_effect=over)):
            with pytest.raises(HTTPException) as exc_info:
                await stream_generate_raw(body, request, db)

        assert exc_info.value.status_code == 429
        mock_tts.generate_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_sse_route_forwards_tenant_and_db_to_generate_stream(self):
        """The route must hand tenant_id + db down so metering can attribute."""
        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("tenant-42")
        request.app.state.provider_registry = MagicMock()

        seen: dict[str, object] = {}

        async def _capturing_stream(**kwargs: object):
            seen.update(kwargs)
            yield AudioChunk(data=b"x", chunk_index=0)
            yield AudioChunk(data=b"", chunk_index=1, is_final=True, total_bytes=1)

        mock_tts = MagicMock()
        mock_tts.generate_stream = _capturing_stream
        request.app.state.tts_service = mock_tts
        db = AsyncMock()

        response = await stream_generate(body, request, db)
        # Drain the SSE generator so the inner async-for actually runs.
        _ = [part async for part in response.body_iterator]

        assert seen["tenant_id"] == tenant_uuid("tenant-42")
        assert seen["db"] is db


# ---------------------------------------------------------------------------
# B5 — a late failure after a paid synthesis must not bill a failed job
# ---------------------------------------------------------------------------


def _async_job() -> MagicMock:
    job = MagicMock()
    job.text = "Hello world"
    job.provider = "auto"
    job.voice = "default"
    job.speed = 1.0
    job.output_format = "mp3"
    job.model = None
    job.text_length = 11
    job.routing_strategy = "balanced"
    return job


class TestLateFailureDoesNotOverbill:
    @pytest.mark.asyncio
    async def test_storage_failure_after_synthesis_emits_no_cost_event(self):
        """store() raising after a paid synthesis must NOT put a charge on the bus."""
        service = _make_service()
        storage = AsyncMock()
        storage.store = AsyncMock(side_effect=OSError("disk full"))
        service._storage = storage
        db = AsyncMock()

        result = GenerationResult(
            audio_data=b"paid-audio", format="mp3", sample_rate=24000, duration_seconds=2.0
        )
        gen_meta = {
            "resolved_provider": "openai",
            "estimated_cost": 0.05,
            "latency_ms": 9,
            "cache_hit": False,
        }

        with (
            patch.object(
                service, "generate_sync", AsyncMock(return_value=(result, gen_meta))
            ) as mock_gen,
            patch.object(service, "_emit_cost_event") as mock_emit,
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=_async_job())
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            with pytest.raises(OSError):
                await service.process_job(db, "job-1", tenant_uuid("t1"))

        # generate_sync was told to DEFER billing to process_job...
        assert mock_gen.call_args[1]["emit_cost"] is False
        # ...and since store() failed before completion, nothing was billed.
        mock_emit.assert_not_called()
        # The job is terminal-failed with a $0 error usage row — consistent.
        mock_jm.mark_failed.assert_awaited_once()
        assert mock_jm.record_usage.call_args[1]["is_error"] is True

    @pytest.mark.asyncio
    async def test_successful_job_bills_exactly_once_after_completion(self):
        """The happy path still bills — once, after the job is durably completed."""
        service = _make_service()
        storage = AsyncMock()
        storage.store = AsyncMock(return_value="audio/job-1.mp3")
        service._storage = storage
        db = AsyncMock()

        result = GenerationResult(
            audio_data=b"paid-audio", format="mp3", sample_rate=24000, duration_seconds=2.0
        )
        gen_meta = {
            "resolved_provider": "openai",
            "estimated_cost": 0.05,
            "latency_ms": 9,
            "cache_hit": False,
        }

        with (
            patch.object(
                service, "generate_sync", AsyncMock(return_value=(result, gen_meta))
            ) as mock_gen,
            patch.object(service, "_emit_cost_event") as mock_emit,
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=_async_job())
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            await service.process_job(db, "job-1", tenant_uuid("t1"))

        assert mock_gen.call_args[1]["emit_cost"] is False
        mock_jm.mark_failed.assert_not_awaited()
        # Billed once, attributed to the provider that actually produced audio.
        mock_emit.assert_called_once()
        assert mock_emit.call_args[1]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_async_cache_hit_does_not_bill_the_cost_bus(self):
        """A cache hit did no provider work — the async job path must NOT emit a
        cost event, matching the synchronous path (which returns before its emit
        on a cache hit) and the $0 usage row it records. Regression: process_job
        previously called _emit_cost_event unconditionally, so a cached async job
        for a paid provider was billed a nonzero cost recomputed from len(text)."""
        service = _make_service()
        storage = AsyncMock()
        storage.store = AsyncMock(return_value="audio/job-1.mp3")
        service._storage = storage
        db = AsyncMock()

        result = GenerationResult(
            audio_data=b"cached-audio", format="mp3", sample_rate=24000, duration_seconds=2.0
        )
        gen_meta = {
            "resolved_provider": "openai",
            "estimated_cost": 0.0,
            "latency_ms": 1,
            "cache_hit": True,
        }

        with (
            patch.object(
                service, "generate_sync", AsyncMock(return_value=(result, gen_meta))
            ) as mock_gen,
            patch.object(service, "_emit_cost_event") as mock_emit,
            patch("src.services.tts_service.JobManager") as mock_jm,
        ):
            mock_jm.get_job = AsyncMock(return_value=_async_job())
            mock_jm.mark_processing = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            await service.process_job(db, "job-1", tenant_uuid("t1"))

        assert mock_gen.call_args[1]["emit_cost"] is False
        mock_jm.mark_failed.assert_not_awaited()
        # Cache hit → no charge on the cost bus...
        mock_emit.assert_not_called()
        # ...but the usage row is still recorded, at $0 and flagged cache_hit.
        mock_jm.record_usage.assert_awaited_once()
        assert mock_jm.record_usage.call_args[1]["cache_hit"] is True
        assert mock_jm.record_usage.call_args[1]["estimated_cost_usd"] == 0.0
