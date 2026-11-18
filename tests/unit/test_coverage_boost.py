"""Targeted tests to boost coverage from 82% to 90%+.

Covers uncovered lines in:
- kokoro_provider.py: _get_pipeline, _generate_sync, is_available, generate, streaming
- kitten_provider.py: _get_model, _generate_sync, is_available, generate
- openai_provider.py: generate (HTTP success), generate_stream
- elevenlabs_provider.py: generate (HTTP success), generate_stream
- database/connection.py: _ensure_database_exists, create_engine_with_retry, get_engine
- asgi.py: lifespan, root, error handler, main
- cache/redis_client.py: connect, disconnect pubsub, get_pubsub not connected, module-level
- services/tts_service.py: _resolve_provider auto, _notify real NS, _apply_post_processing
- api/tts.py: async generate, streaming, batch, quota, download batch audio
"""

import queue
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest

from src.providers.base import AudioChunk, GenerationResult
from tests.conftest import tenant_uuid

if TYPE_CHECKING:
    import numpy as np

# Optional ML dependencies — the kokoro/kitten provider tests below exercise
# real ``numpy``/``soundfile``/``torch`` round-trips and are SKIPPED when those
# packages are not installed (i.e. when the service is built without the
# ``[kokoro]``/``[kitten]`` extras). The behavioural assertions inside still
# run real (no mocks for numpy/soundfile) when the deps are available.
_HAS_NUMPY = find_spec("numpy") is not None
_HAS_SOUNDFILE = find_spec("soundfile") is not None
_HAS_TORCH = find_spec("torch") is not None
_HAS_AUDIO_STACK = _HAS_NUMPY and _HAS_SOUNDFILE
requires_audio_stack = pytest.mark.skipif(
    not _HAS_AUDIO_STACK,
    reason="numpy + soundfile required (install with extras: tts-service[kokoro])",
)
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch required (install with extras: tts-service[kokoro])",
)


def _fake_create_task(coro, **kwargs):
    """Close the coroutine so it doesn't trigger 'coroutine was never awaited'."""
    coro.close()
    return MagicMock()


# We cannot import numpy/soundfile directly in tests due to system Python conflict.
# All tests that exercise code containing `import numpy` or `import soundfile`
# must mock these modules or call through asyncio.to_thread which is also mocked.

# ============================================================================
# Kokoro Provider - full coverage
# ============================================================================


class TestKokoroProviderGetPipeline:
    """Cover lines 160-190: _get_pipeline with mocked KPipeline."""

    def test_get_pipeline_import_error(self):
        from src.providers.kokoro_provider import KokoroProvider

        original = KokoroProvider._pipelines.copy()
        KokoroProvider._pipelines.clear()
        try:
            with patch.dict("sys.modules", {"kokoro": None}):
                with pytest.raises(RuntimeError, match="extra dependencies"):
                    KokoroProvider._get_pipeline("a")
        finally:
            KokoroProvider._pipelines.update(original)

    @requires_torch
    def test_get_pipeline_success(self):
        # ⚠️ `torch` MUST be imported BEFORE entering `patch.dict("sys.modules")`
        # below. `_get_pipeline` does a real `import torch` internally; if that
        # first import happens INSIDE the patched dict, `patch.dict`'s exit
        # (`clear()` + `update(snapshot)`) evicts `torch` and `torch._C` from
        # `sys.modules` again. The next `import torch` anywhere in the process
        # then takes CPython's `reload_singlephase_extension` path for
        # `torch._C`, which re-runs libtorch's `initModule()` in an already
        # initialised interpreter and SEGFAULTS (exit 139) — observed killing
        # the whole suite in `test_request_id_middleware.py`, whose app lifespan
        # imports kokoro/torch on a worker thread. Pre-importing puts torch in
        # the snapshot so the restore preserves it.
        import torch  # noqa: F401

        from src.providers.kokoro_provider import KokoroProvider

        original = KokoroProvider._pipelines.copy()
        KokoroProvider._pipelines.clear()
        try:
            mock_kpipeline_cls = MagicMock()
            mock_pipeline_instance = MagicMock()
            mock_kpipeline_cls.return_value = mock_pipeline_instance

            mock_kokoro = MagicMock()
            mock_kokoro.KPipeline = mock_kpipeline_cls

            with patch.dict("sys.modules", {"kokoro": mock_kokoro}):
                result = KokoroProvider._get_pipeline("a")
                assert result is mock_pipeline_instance
                # Second call should return cached
                result2 = KokoroProvider._get_pipeline("a")
                assert result2 is mock_pipeline_instance
                assert mock_kpipeline_cls.call_count == 1
        finally:
            KokoroProvider._pipelines.clear()
            KokoroProvider._pipelines.update(original)


@requires_audio_stack
class TestKokoroProviderGenerateSync:
    """Real-behavior tests for ``KokoroProvider._generate_sync``.

    These tests use real numpy arrays and real ``soundfile`` so the audio
    pipeline is exercised end-to-end. Only the upstream ``KPipeline`` (which
    requires native espeak/torch) and ``wav_to_mp3`` (which runs ffmpeg) are
    mocked — everything else runs for real, including the WAV encoding
    round-trip and the duration math at ``kokoro_provider.py:254``.
    """

    @staticmethod
    def _make_sine_audio(duration_s: float = 0.1, rate: int = 24000) -> "np.ndarray":  # noqa: UP037
        """Build a real float32 sine-wave array of known length."""
        import numpy as np

        n = int(duration_s * rate)
        t = np.arange(n, dtype=np.float32) / rate
        return (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    def _fake_pipeline_with(self, audio_array: "np.ndarray | None") -> MagicMock:  # noqa: UP037
        """Callable whose call returns a single pipeline result with ``audio``."""
        from types import SimpleNamespace

        result = SimpleNamespace(audio=audio_array)
        pipeline = MagicMock()
        pipeline.return_value = [result]
        return pipeline

    def test_generate_sync_wav_format_round_trips_through_soundfile(self) -> None:
        """Real WAV encoding: the bytes returned must decode back to the input audio.

        Catches regressions in:
          - the sample rate passed to ``sf.write`` (would decode at a wrong rate)
          - the format string (non-WAV would fail to parse)
          - duration math at ``kokoro_provider.py:254``
        """
        import io

        import numpy as np
        import soundfile as sf

        from src.providers.kokoro_provider import KOKORO_SAMPLE_RATE, KokoroProvider

        provider = KokoroProvider()
        audio_in = self._make_sine_audio(duration_s=0.1, rate=KOKORO_SAMPLE_RATE)
        pipeline = self._fake_pipeline_with(audio_in)

        with patch.object(KokoroProvider, "_get_pipeline", return_value=pipeline):
            result = provider._generate_sync("Hello", "af_heart", 1.0, "wav")

        # Metadata
        assert result.format == "wav"
        assert result.sample_rate == KOKORO_SAMPLE_RATE
        # len(audio_in) / rate == 0.1 s exactly
        assert result.duration_seconds == pytest.approx(len(audio_in) / KOKORO_SAMPLE_RATE)

        # The WAV bytes must round-trip through soundfile and produce the
        # same audio at the same sample rate.
        decoded, decoded_rate = sf.read(io.BytesIO(result.audio_data), dtype="float32")
        assert decoded_rate == KOKORO_SAMPLE_RATE
        np.testing.assert_allclose(decoded, audio_in, atol=1e-4)

        # And the pipeline was called exactly once with the request's args.
        pipeline.assert_called_once_with("Hello", voice="af_heart", speed=1.0)

    def test_generate_sync_mp3_format_writes_valid_wav_before_conversion(
        self, tmp_path: Path
    ) -> None:
        """MP3 path: assert the WAV handed to ``wav_to_mp3`` is a real, decodable WAV.

        Instead of mocking ``wav_to_mp3`` to return fake bytes, we give it a
        side_effect that (a) reads the wav path with ``sf.read`` and verifies
        the audio survived the first leg, (b) writes deterministic bytes to a
        real mp3 path and returns it. Both halves of the chain — sf.write and
        the mp3-path round-trip — are exercised.
        """
        import numpy as np
        import soundfile as sf

        from src.providers.kokoro_provider import KOKORO_SAMPLE_RATE, KokoroProvider

        provider = KokoroProvider()
        audio_in = self._make_sine_audio(duration_s=0.25, rate=KOKORO_SAMPLE_RATE)
        pipeline = self._fake_pipeline_with(audio_in)

        decoded_ref: list[np.ndarray] = []
        rate_ref: list[int] = []

        def fake_wav_to_mp3(wav_path: Path) -> Path:
            # Verify the wav path written by sf.write contains our audio.
            decoded, rate = sf.read(str(wav_path), dtype="float32")
            decoded_ref.append(decoded)
            rate_ref.append(int(rate))
            mp3_path = tmp_path / "out.mp3"
            mp3_path.write_bytes(b"ID3\x03" + b"\x00" * 32)  # plausible MP3 header
            return mp3_path

        with (
            patch.object(KokoroProvider, "_get_pipeline", return_value=pipeline),
            patch("src.providers.kokoro_provider.wav_to_mp3", side_effect=fake_wav_to_mp3),
        ):
            result = provider._generate_sync("Hello", "af_heart", 1.25, "mp3")

        assert result.format == "mp3"
        assert result.sample_rate == KOKORO_SAMPLE_RATE
        assert result.duration_seconds == pytest.approx(len(audio_in) / KOKORO_SAMPLE_RATE)
        assert result.audio_data == b"ID3\x03" + b"\x00" * 32

        # The WAV handed to wav_to_mp3 was a valid encoding of the pipeline audio.
        assert rate_ref == [KOKORO_SAMPLE_RATE]
        assert len(decoded_ref) == 1
        np.testing.assert_allclose(decoded_ref[0], audio_in, atol=1e-4)

        # And the pipeline saw the requested speed.
        pipeline.assert_called_once_with("Hello", voice="af_heart", speed=1.25)

    def test_generate_sync_empty_segments_raises_runtime_error(self) -> None:
        """Pipeline yielding no audio → RuntimeError at ``kokoro_provider.py:251``.

        Uses a real pipeline result with ``audio=None`` so the production
        ``if result.audio is not None`` filter runs for real.
        """
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        pipeline = self._fake_pipeline_with(None)

        with patch.object(KokoroProvider, "_get_pipeline", return_value=pipeline):
            with pytest.raises(RuntimeError, match="no audio"):
                provider._generate_sync("Hello", "af_heart", 1.0, "mp3")

    def test_generate_sync_concatenates_multiple_segments_in_order(self) -> None:
        """Multiple pipeline results are concatenated: duration sums, order preserved."""
        import io
        from types import SimpleNamespace

        import numpy as np
        import soundfile as sf

        from src.providers.kokoro_provider import KOKORO_SAMPLE_RATE, KokoroProvider

        provider = KokoroProvider()
        seg_a = self._make_sine_audio(duration_s=0.05, rate=KOKORO_SAMPLE_RATE)
        seg_b = (self._make_sine_audio(duration_s=0.05, rate=KOKORO_SAMPLE_RATE) * -1.0).astype(
            np.float32
        )
        expected = np.concatenate([seg_a, seg_b])

        pipeline = MagicMock()
        pipeline.return_value = [
            SimpleNamespace(audio=seg_a),
            SimpleNamespace(audio=None),  # Must be filtered out by the `is not None` guard.
            SimpleNamespace(audio=seg_b),
        ]

        with patch.object(KokoroProvider, "_get_pipeline", return_value=pipeline):
            result = provider._generate_sync("Hello", "af_heart", 1.0, "wav")

        assert result.duration_seconds == pytest.approx(len(expected) / KOKORO_SAMPLE_RATE)
        decoded, _ = sf.read(io.BytesIO(result.audio_data), dtype="float32")
        np.testing.assert_allclose(decoded, expected, atol=1e-4)


class TestKokoroProviderIsAvailable:
    """Cover lines 259-267: is_available."""

    @pytest.mark.asyncio
    async def test_is_available_no_kokoro(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        with patch.dict("sys.modules", {"kokoro": None}):
            result = await provider.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_no_espeak(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("shutil.which", return_value=None),
        ):
            result = await provider.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_no_ffmpeg(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("shutil.which", return_value="/usr/bin/espeak-ng"),
            patch("src.providers.kokoro_provider.is_ffmpeg_available", return_value=False),
        ):
            result = await provider.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_all_present(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("shutil.which", return_value="/usr/bin/espeak-ng"),
            patch("src.providers.kokoro_provider.is_ffmpeg_available", return_value=True),
        ):
            result = await provider.is_available()
            assert result is True


class TestKokoroProviderGenerate:
    """Cover line 277: async generate delegates to _generate_sync."""

    @pytest.mark.asyncio
    async def test_generate_delegates_to_sync(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()
        expected = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        with patch.object(provider, "_generate_sync", return_value=expected):
            result = await provider.generate("Hello", "af_heart", 1.0, "mp3")
            assert result is expected


class TestKokoroProviderStreaming:
    """Cover lines 279-363: _generate_segments_to_queue and generate_stream."""

    def test_generate_segments_to_queue_wav(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()

        mock_sf = MagicMock()
        mock_audio = MagicMock()
        mock_result = MagicMock()
        mock_result.audio = mock_audio

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [mock_result]

        q: queue.Queue[AudioChunk | None] = queue.Queue()

        with (
            patch.object(KokoroProvider, "_get_pipeline", return_value=mock_pipeline),
            patch.dict("sys.modules", {"soundfile": mock_sf}),
        ):
            # Mock io.BytesIO
            mock_buf = MagicMock()
            mock_buf.getvalue.return_value = b"wav_chunk"
            with patch("src.providers.kokoro_provider.io.BytesIO", return_value=mock_buf):
                provider._generate_segments_to_queue(
                    "Hello", "af_heart", 1.0, "wav", q, threading.Event(), []
                )

        chunks = []
        while True:
            chunk = q.get()
            if chunk is None:
                break
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[-1].is_final is True
        assert chunks[0].format == "wav"

    def test_generate_segments_to_queue_mp3(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()

        mock_sf = MagicMock()
        mock_audio = MagicMock()
        mock_result = MagicMock()
        mock_result.audio = mock_audio

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [mock_result]

        q: queue.Queue[AudioChunk | None] = queue.Queue()

        with (
            patch.object(KokoroProvider, "_get_pipeline", return_value=mock_pipeline),
            patch.dict("sys.modules", {"soundfile": mock_sf}),
            patch("src.providers.kokoro_provider.wav_to_mp3") as mock_wav_to_mp3,
        ):
            mock_mp3_path = MagicMock()
            mock_mp3_path.read_bytes.return_value = b"mp3chunk"
            mock_wav_to_mp3.return_value = mock_mp3_path

            mock_buf = MagicMock()
            mock_buf.getvalue.return_value = b"wav_for_conversion"
            with patch("src.providers.kokoro_provider.io.BytesIO", return_value=mock_buf):
                provider._generate_segments_to_queue(
                    "Hello", "af_heart", 1.0, "mp3", q, threading.Event(), []
                )

        chunks = []
        while True:
            chunk = q.get()
            if chunk is None:
                break
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[-1].is_final is True

    def test_generate_segments_to_queue_error(self):
        """A pipeline failure must surface as an error, NOT as an is_final sentinel.

        This test used to assert ``chunks[-1].is_final is True`` on the error
        path — i.e. it pinned the defect: a dead pipeline was reported as a
        completed synthesis, so the caller billed the full text and served a
        200 with zero audio bytes.
        """
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()

        mock_sf = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("pipeline error")

        q: queue.Queue[AudioChunk | None] = queue.Queue()
        error_holder: list[BaseException] = []

        with (
            patch.object(KokoroProvider, "_get_pipeline", return_value=mock_pipeline),
            patch.dict("sys.modules", {"soundfile": mock_sf}),
        ):
            provider._generate_segments_to_queue(
                "Hello", "af_heart", 1.0, "mp3", q, threading.Event(), error_holder
            )

        chunks = []
        while True:
            chunk = q.get()
            if chunk is None:
                break
            chunks.append(chunk)

        assert chunks == [], "a failed pipeline must not emit any chunk, least of all is_final"
        assert [str(e) for e in error_holder] == ["pipeline error"]

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        from src.providers.kokoro_provider import KokoroProvider

        provider = KokoroProvider()

        def mock_gen_to_queue(text, voice, speed, fmt, q, stop_event, error_holder):
            q.put(AudioChunk(data=b"data", chunk_index=0, format="mp3", sample_rate=24000))
            q.put(
                AudioChunk(
                    data=b"",
                    chunk_index=1,
                    is_final=True,
                    format="mp3",
                    sample_rate=24000,
                    total_bytes=4,
                )
            )
            q.put(None)

        with patch.object(provider, "_generate_segments_to_queue", side_effect=mock_gen_to_queue):
            chunks = []
            async for chunk in provider.generate_stream("Hello", "af_heart"):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].data == b"data"
            assert chunks[1].is_final is True


# ============================================================================
# Kitten Provider - full coverage
# ============================================================================


class TestKittenProviderGetModel:
    """Cover lines 66-79: _get_model."""

    def test_get_model_import_error(self):
        from src.providers.kitten_provider import KittenProvider

        original = KittenProvider._models.copy()
        KittenProvider._models.clear()
        try:
            with patch.dict("sys.modules", {"kittentts": None}):
                with pytest.raises(RuntimeError, match="extra dependencies"):
                    KittenProvider._get_model("test-model")
        finally:
            KittenProvider._models.clear()
            KittenProvider._models.update(original)

    def test_get_model_success(self):
        from src.providers.kitten_provider import KittenProvider

        original = KittenProvider._models.copy()
        KittenProvider._models.clear()
        try:
            mock_kittentts = MagicMock()
            mock_model_instance = MagicMock()
            mock_kittentts.KittenTTS.return_value = mock_model_instance

            with patch.dict("sys.modules", {"kittentts": mock_kittentts}):
                result = KittenProvider._get_model("test-model")
                assert result is mock_model_instance
                result2 = KittenProvider._get_model("test-model")
                assert result2 is mock_model_instance
                assert mock_kittentts.KittenTTS.call_count == 1
        finally:
            KittenProvider._models.clear()
            KittenProvider._models.update(original)


@requires_audio_stack
class TestKittenProviderGenerateSync:
    """Real-behavior tests for ``KittenProvider._generate_sync``.

    Only the upstream ``KittenTTS`` model (which requires a native weights
    download) and ``wav_to_mp3`` (which runs ffmpeg) are mocked. Everything
    else runs for real, including the ``isinstance(audio, np.ndarray)`` branch
    at ``kitten_provider.py:108`` and the list→array conversion path.
    """

    @staticmethod
    def _make_audio(duration_s: float = 0.1) -> "np.ndarray":  # noqa: UP037
        import numpy as np

        n = int(duration_s * 24000)
        return (np.linspace(-0.5, 0.5, n, dtype=np.float32)).astype(np.float32)

    def test_generate_sync_wav_round_trips_and_passes_speed_to_model(self) -> None:
        """WAV path: the bytes decode back to the input, and the model sees
        our ``speed`` argument.

        Catches regressions in:
          - ``model.generate`` call-site args at ``kitten_provider.py:101``
          - ``sf.write`` sample rate + format at ``kitten_provider.py:117``
          - ``duration = len(audio) / KITTEN_SAMPLE_RATE`` at line 113
        """
        import io

        import numpy as np
        import soundfile as sf

        from src.providers.kitten_provider import KITTEN_SAMPLE_RATE, KittenProvider

        provider = KittenProvider()
        audio_in = self._make_audio(duration_s=0.1)

        mock_model = MagicMock()
        mock_model.generate.return_value = audio_in

        with patch.object(KittenProvider, "_get_model", return_value=mock_model):
            result = provider._generate_sync("Hi", "Bella", 1.5, "wav", "mini")

        assert result.format == "wav"
        assert result.sample_rate == KITTEN_SAMPLE_RATE
        assert result.duration_seconds == pytest.approx(len(audio_in) / KITTEN_SAMPLE_RATE)

        decoded, decoded_rate = sf.read(io.BytesIO(result.audio_data), dtype="float32")
        assert decoded_rate == KITTEN_SAMPLE_RATE
        np.testing.assert_allclose(decoded, audio_in, atol=1e-4)

        mock_model.generate.assert_called_once_with("Hi", voice="Bella", speed=1.5)

    def test_generate_sync_mp3_hands_valid_wav_to_ffmpeg_bridge(self, tmp_path: Path) -> None:
        """MP3 path: the temp WAV written before ``wav_to_mp3`` must be a
        valid, decodable file containing the original audio.

        The side_effect reads the temp WAV with ``sf.read`` (real decode) and
        returns a fresh mp3-like file whose bytes the test asserts on.
        """
        import numpy as np
        import soundfile as sf

        from src.providers.kitten_provider import KITTEN_SAMPLE_RATE, KittenProvider

        provider = KittenProvider()
        audio_in = self._make_audio(duration_s=0.2)
        mock_model = MagicMock()
        mock_model.generate.return_value = audio_in

        decoded_ref: list[np.ndarray] = []

        def fake_wav_to_mp3(wav_path: Path) -> Path:
            decoded, rate = sf.read(str(wav_path), dtype="float32")
            assert rate == KITTEN_SAMPLE_RATE
            decoded_ref.append(decoded)
            mp3_path = tmp_path / "kitten.mp3"
            mp3_path.write_bytes(b"ID3\x04" + b"\xff" * 16)
            return mp3_path

        with (
            patch.object(KittenProvider, "_get_model", return_value=mock_model),
            patch("src.providers.kitten_provider.wav_to_mp3", side_effect=fake_wav_to_mp3),
        ):
            result = provider._generate_sync("Hi", "Bella", 1.0, "mp3", "mini")

        assert result.format == "mp3"
        assert result.sample_rate == KITTEN_SAMPLE_RATE
        assert result.duration_seconds == pytest.approx(len(audio_in) / KITTEN_SAMPLE_RATE)
        assert result.audio_data == b"ID3\x04" + b"\xff" * 16
        assert len(decoded_ref) == 1
        np.testing.assert_allclose(decoded_ref[0], audio_in, atol=1e-4)

    def test_generate_sync_converts_list_to_ndarray(self, tmp_path: Path) -> None:
        """If the model returns a plain list, the non-ndarray branch at
        ``kitten_provider.py:110-111`` converts it via ``np.array(..., dtype=float32)``.

        Provide a REAL list (not a mock), let the real conversion run, and
        assert the duration computed from the converted length is correct.
        """
        import io

        import numpy as np
        import soundfile as sf

        from src.providers.kitten_provider import KITTEN_SAMPLE_RATE, KittenProvider

        provider = KittenProvider()
        audio_in_list = [float(i) / 2400.0 for i in range(2400)]
        mock_model = MagicMock()
        mock_model.generate.return_value = audio_in_list

        with patch.object(KittenProvider, "_get_model", return_value=mock_model):
            result = provider._generate_sync("Hi", "Bella", 1.0, "wav", "mini")

        assert result.sample_rate == KITTEN_SAMPLE_RATE
        assert result.duration_seconds == pytest.approx(2400 / KITTEN_SAMPLE_RATE)

        decoded, _ = sf.read(io.BytesIO(result.audio_data), dtype="float32")
        np.testing.assert_allclose(decoded, np.array(audio_in_list, dtype=np.float32), atol=1e-4)

    def test_generate_sync_falls_back_when_model_rejects_speed_kwarg(self) -> None:
        """Some Kitten models don't accept ``speed``; production swallows
        ``TypeError`` and retries without it (``kitten_provider.py:102-106``).

        We fake a model whose first call raises ``TypeError`` and whose
        second call succeeds, and assert both call shapes.
        """
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        audio_in = self._make_audio(duration_s=0.05)

        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            TypeError("unexpected keyword argument 'speed'"),
            audio_in,
        ]

        with patch.object(KittenProvider, "_get_model", return_value=mock_model):
            result = provider._generate_sync("Hi", "Bella", 2.0, "wav", "mini")

        assert result.format == "wav"
        assert mock_model.generate.call_count == 2
        # First call included speed, second call fell back without it.
        assert mock_model.generate.call_args_list[0].kwargs == {"voice": "Bella", "speed": 2.0}
        assert mock_model.generate.call_args_list[1].kwargs == {"voice": "Bella"}


class TestKittenProviderIsAvailable:
    """Cover lines 131-136: is_available."""

    @pytest.mark.asyncio
    async def test_is_available_true(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            result = await provider.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        with patch.dict("sys.modules", {"kittentts": None}):
            result = await provider.is_available()
            assert result is False


class TestKittenProviderGenerate:
    """Cover lines 146-148: generate delegates via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_generate_delegates(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        expected = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        with patch.object(provider, "_generate_sync", return_value=expected):
            result = await provider.generate("Hello", "Bella")
            assert result is expected

    @pytest.mark.asyncio
    async def test_generate_with_model_kwarg(self):
        from src.providers.kitten_provider import KittenProvider

        provider = KittenProvider()
        expected = GenerationResult(
            audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
        )
        with patch.object(provider, "_generate_sync", return_value=expected) as mock_sync:
            result = await provider.generate("Hello", "Bella", model="mini")
            assert result is expected
            call_args = mock_sync.call_args
            assert call_args[0][4] == "KittenML/kitten-tts-mini-0.8"


# ============================================================================
# OpenAI Provider - generate and stream
# ============================================================================


class TestOpenAIProviderGenerateSuccess:
    """Real-behavior tests for ``OpenAIProvider.generate``.

    Uses ``httpx.MockTransport`` so the real request-building code runs
    and the request URL, headers, and JSON body are captured and asserted
    on. Only ``probe_duration`` (which shells out to ffprobe) is stubbed.
    """

    def _install_transport(
        self,
        provider,
        handler,  # type: ignore[no-untyped-def]
    ) -> None:
        """Replace the provider's httpx client with a MockTransport-backed one."""
        import httpx

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @pytest.mark.asyncio
    async def test_generate_sends_correct_request_and_parses_mpeg_response(self) -> None:
        """The real ``_call`` closure must hit the OpenAI URL with the right
        auth header, JSON body, and parse the mpeg content type back to
        ``format='mp3'``.
        """
        import json

        import httpx

        from src.providers.openai_provider import OPENAI_TTS_URL, OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key", model="tts-1-hd")

        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=b"\xff\xfb\x90real_mpeg_bytes",
                headers={"content-type": "audio/mpeg"},
            )

        self._install_transport(provider, handler)
        with patch(
            "src.providers.openai_provider.probe_duration", return_value=1.234
        ) as mock_probe:
            result = await provider.generate("Hello", "alloy", speed=0.9, output_format="mp3")

        assert seen["url"] == OPENAI_TTS_URL
        assert seen["auth"] == "Bearer sk-test-key"
        body = seen["body"]
        assert isinstance(body, dict)
        assert body == {
            "model": "tts-1-hd",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 0.9,
        }

        assert result.audio_data == b"\xff\xfb\x90real_mpeg_bytes"
        assert result.format == "mp3"  # derived from content-type header
        assert result.sample_rate == 24000
        assert result.duration_seconds == pytest.approx(1.234)
        mock_probe.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_non_mpeg_content_type_preserves_requested_format(self) -> None:
        """When the response content-type is not audio/mpeg, fall back to the
        ``output_format`` argument (``openai_provider.py:149``)."""
        import httpx

        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"RIFF....WAVE" + b"\x00" * 16,
                headers={"content-type": "audio/wav"},
            )

        self._install_transport(provider, handler)
        with patch("src.providers.openai_provider.probe_duration", return_value=0.0):
            result = await provider.generate("Hi", "nova", output_format="wav")

        assert result.format == "wav"
        assert result.audio_data.startswith(b"RIFF")

    @pytest.mark.asyncio
    async def test_generate_empty_response_raises_runtime_error(self) -> None:
        """Empty body → RuntimeError at ``openai_provider.py:139``.

        Exercises the real ``_call`` → ``raise_for_status`` → `` not response.content``
        chain — not a mocked short-circuit.
        """
        import httpx

        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"", headers={"content-type": "audio/mpeg"})

        self._install_transport(provider, handler)
        with pytest.raises(RuntimeError, match="empty response"):
            await provider.generate("Hello", "alloy")

    @pytest.mark.asyncio
    async def test_generate_missing_api_key_raises_before_network(self) -> None:
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="")
        with pytest.raises(ValueError, match="API key"):
            await provider.generate("Hello", "alloy")

    @pytest.mark.asyncio
    async def test_generate_text_too_long_raises_before_network(self) -> None:
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")
        with pytest.raises(ValueError, match="exceeds OpenAI TTS limit"):
            await provider.generate("x" * (provider.max_text_length + 1), "alloy")

    @pytest.mark.asyncio
    async def test_generate_retries_on_503_then_succeeds(self) -> None:
        """Tripwire for the retry wiring at ``openai_provider.py:131-135``.

        First call returns 503 (retryable), second returns 200. The real
        retry machinery must drive the two network calls. Without the
        ``_RetryableHTTPError`` mapping, the 503 would escape and the test
        would fail.
        """
        import httpx

        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")

        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(503, content=b"busy")
            return httpx.Response(
                200,
                content=b"\xff\xfbmpeg",
                headers={"content-type": "audio/mpeg"},
            )

        self._install_transport(provider, handler)
        # Patch asyncio.sleep inside retry to keep the test fast.
        with (
            patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock),
            patch("src.providers.openai_provider.probe_duration", return_value=0.0),
        ):
            result = await provider.generate("Hello", "alloy")

        assert call_count["n"] == 2
        assert result.audio_data == b"\xff\xfbmpeg"
        assert result.format == "mp3"


class TestOpenAIProviderStream:
    """Cover lines 130-157: generate_stream."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        async def mock_aiter_bytes(chunk_size=4096):
            yield b"chunk1"
            yield b"chunk2"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = mock_aiter_bytes

        # The stream context manager
        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        # Mock the instance client's stream method directly
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)
        provider._client = mock_client

        chunks = []
        async for chunk in provider.generate_stream("Hello", "alloy"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].data == b"chunk1"
        assert chunks[1].data == b"chunk2"
        assert chunks[2].is_final is True
        assert chunks[2].total_bytes == len(b"chunk1") + len(b"chunk2")


# ============================================================================
# ElevenLabs Provider - generate and stream
# ============================================================================


class TestElevenLabsProviderGenerateSuccess:
    """Real-behavior tests for ``ElevenLabsProvider.generate`` via MockTransport."""

    def _install_transport(
        self,
        provider,
        handler,  # type: ignore[no-untyped-def]
    ) -> None:
        import httpx

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @pytest.mark.asyncio
    async def test_generate_posts_to_voice_scoped_url_with_voice_settings(self) -> None:
        """Verify the voice-templated URL, ``xi-api-key`` header, and the
        full ``voice_settings`` JSON shape at ``elevenlabs_provider.py:182-198``.
        """
        import json

        import httpx

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(
            api_key="xi-test",  # pragma: allowlist secret
            model_id="eleven_multilingual_v2",
            stability=0.6,
            similarity_boost=0.8,
        )

        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["xi_key"] = request.headers.get("xi-api-key")
            seen["content_type"] = request.headers.get("content-type")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, content=b"\xff\xfbeleven_mpeg")

        self._install_transport(provider, handler)
        with patch("src.providers.elevenlabs_provider.probe_duration", return_value=2.5):
            result = await provider.generate("Hi there", "rachel", speed=1.1)

        # The PATH is what this test is about; the request now also carries an
        # explicit ?output_format= (pinning the format we claim to store), so
        # compare the path rather than the whole URL.
        assert str(seen["url"]).startswith("https://api.elevenlabs.io/v1/text-to-speech/rachel")
        assert seen["xi_key"] == "xi-test"
        assert seen["content_type"] == "application/json"
        assert seen["body"] == {
            "text": "Hi there",
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.6, "similarity_boost": 0.8},
            "speed": 1.1,
        }

        assert result.audio_data == b"\xff\xfbeleven_mpeg"
        assert result.format == "mp3"
        # 44100, not 24000: the request asks for ``mp3_44100_128`` and that is
        # what comes back. The declared rate must describe the bytes (B1).
        assert result.sample_rate == 44100
        assert result.duration_seconds == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_generate_kwarg_voice_settings_override_instance_defaults(self) -> None:
        """Per-call ``stability`` / ``similarity_boost`` kwargs must win over
        the values set on the provider instance (``elevenlabs_provider.py:171-180``).
        """
        import json

        import httpx

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="xi-test", stability=0.5, similarity_boost=0.75)
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, content=b"audio")

        self._install_transport(provider, handler)
        with patch("src.providers.elevenlabs_provider.probe_duration", return_value=0.0):
            await provider.generate("Hi", "rachel", stability=0.91, similarity_boost=0.12)

        assert len(bodies) == 1
        assert bodies[0]["voice_settings"] == {"stability": 0.91, "similarity_boost": 0.12}

    @pytest.mark.asyncio
    async def test_generate_url_encodes_voice_id_with_special_chars(self) -> None:
        """Voice id with a slash/space must be percent-encoded into the URL."""
        import httpx

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="xi-test")
        seen_url: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_url.append(str(request.url))
            return httpx.Response(200, content=b"audio")

        self._install_transport(provider, handler)
        with patch("src.providers.elevenlabs_provider.probe_duration", return_value=0.0):
            await provider.generate("Hi", "my voice/42")

        # `quote(..., safe="")` must turn both the space and slash into %.
        assert len(seen_url) == 1
        # Encoding of the voice id is the point here; an explicit
        # ?output_format= query now follows it, so assert on the path.
        assert urlparse(seen_url[0]).path.endswith("my%20voice%2F42")

    @pytest.mark.asyncio
    async def test_generate_empty_response_raises_runtime_error(self) -> None:
        import httpx

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="xi-test")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        self._install_transport(provider, handler)
        with pytest.raises(RuntimeError, match="empty response"):
            await provider.generate("Hello", "rachel")

    @pytest.mark.asyncio
    async def test_generate_missing_api_key_raises_before_network(self) -> None:
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="")
        with pytest.raises(ValueError, match="API key"):
            await provider.generate("Hello", "rachel")

    @pytest.mark.asyncio
    async def test_generate_text_too_long_raises_before_network(self) -> None:
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="xi-test")
        with pytest.raises(ValueError, match="exceeds ElevenLabs TTS limit"):
            await provider.generate("x" * (provider.max_text_length + 1), "rachel")

    @pytest.mark.asyncio
    async def test_generate_retries_on_429_then_succeeds(self) -> None:
        """Tripwire for the retry wiring at ``elevenlabs_provider.py:206-210``."""
        import httpx

        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="xi-test")
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(429, content=b"rate limited")
            return httpx.Response(200, content=b"\xff\xfbaudio")

        self._install_transport(provider, handler)
        with (
            patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock),
            patch("src.providers.elevenlabs_provider.probe_duration", return_value=0.0),
        ):
            result = await provider.generate("Hi", "rachel")

        assert call_count["n"] == 2
        assert result.audio_data == b"\xff\xfbaudio"


class TestElevenLabsProviderStream:
    """Cover lines 190-224: generate_stream."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test-key")

        async def mock_aiter_bytes(chunk_size=4096):
            yield b"elchunk1"
            yield b"elchunk2"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = mock_aiter_bytes

        # The stream context manager
        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        # Mock the instance client's stream method directly
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)
        provider._client = mock_client

        chunks = []
        async for chunk in provider.generate_stream("Hello", "rachel"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].data == b"elchunk1"
        assert chunks[1].data == b"elchunk2"
        assert chunks[2].is_final is True
        assert chunks[2].total_bytes == len(b"elchunk1") + len(b"elchunk2")


# ============================================================================
# Database Connection - _ensure_database_exists and create_engine_with_retry
# ============================================================================


class TestEnsureDatabaseExists:
    """Cover lines 40-100: _ensure_database_exists."""

    @pytest.mark.asyncio
    async def test_database_already_exists(self):
        from src.database.connection import _ensure_database_exists

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1

        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()

        with patch("w2v_common.database.connection.create_engine", return_value=mock_engine):
            await _ensure_database_exists()

    @pytest.mark.asyncio
    async def test_database_created(self):
        import w2v_common.database.connection as conn_mod

        from src.database.connection import _ensure_database_exists

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None

        executed: list[str] = []

        def _execute(statement, *_args, **_kwargs):
            # A side_effect FUNCTION, not a fixed-length list: the first call is
            # the existence probe and must yield a result whose .scalar() is None
            # (database absent); every following call is provisioning DDL whose
            # return value is unused. A list pinned the exact statement COUNT, so
            # w2v_common 0.65.0 adding a legitimate third statement (REVOKE
            # CONNECT ... FROM PUBLIC, the born-LOCKED policy) exhausted the
            # iterator and StopIteration surfaced as RuntimeError.
            executed.append(str(statement))
            return mock_result if len(executed) == 1 else None

        mock_conn.execute.side_effect = _execute
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()

        with patch("w2v_common.database.connection.create_engine", return_value=mock_engine):
            await _ensure_database_exists()

        # The point of this test: an absent database gets CREATED.
        assert executed, "no statement was executed at all"
        assert "pg_database" in executed[0], executed
        assert any(f"CREATE DATABASE {conn_mod.DB_NAME}" in stmt for stmt in executed), executed

    @pytest.mark.asyncio
    async def test_invalid_db_name(self):
        import w2v_common.database.connection as conn_mod

        original_name = conn_mod.DB_NAME
        conn_mod.DB_NAME = "bad;name"
        try:
            with pytest.raises(ValueError, match="invalid characters"):
                await conn_mod._ensure_database_exists()
        finally:
            conn_mod.DB_NAME = original_name

    @pytest.mark.asyncio
    async def test_database_already_exists_race_condition(self):
        from sqlalchemy.exc import ProgrammingError

        from src.database.connection import _ensure_database_exists

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None

        prog_error = ProgrammingError("", {}, Exception("database already exists"))
        mock_conn.execute.side_effect = [mock_result, prog_error]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()

        with patch("w2v_common.database.connection.create_engine", return_value=mock_engine):
            await _ensure_database_exists()

    @pytest.mark.asyncio
    async def test_connection_refused_retries(self):
        from src.database.connection import _ensure_database_exists

        mock_engine = MagicMock()
        mock_engine.connect.side_effect = ConnectionRefusedError("refused")
        mock_engine.dispose = MagicMock()

        with patch("w2v_common.database.connection.create_engine", return_value=mock_engine):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await _ensure_database_exists(max_retries=1)

    @pytest.mark.asyncio
    async def test_unexpected_error_raises(self):
        from src.database.connection import _ensure_database_exists

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = TypeError("unexpected")

        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()

        with patch("w2v_common.database.connection.create_engine", return_value=mock_engine):
            with pytest.raises(TypeError, match="unexpected"):
                await _ensure_database_exists()


def _make_async_cm(return_value):
    """Helper to create a proper async context manager mock."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestCreateEngineWithRetry:
    """Cover lines 124-166: create_engine_with_retry."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        from src.database.connection import create_engine_with_retry

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_make_async_cm(mock_conn))

        with patch("w2v_common.database.connection.create_async_engine", return_value=mock_engine):
            result = await create_engine_with_retry(max_retries=1, retry_delay=0.01)
            assert result is mock_engine

    @pytest.mark.asyncio
    async def test_database_not_exist_then_create(self):
        from sqlalchemy.exc import ProgrammingError

        from src.database.connection import create_engine_with_retry

        prog_error = ProgrammingError("", {}, Exception("database does not exist"))
        mock_conn_fail = AsyncMock()
        mock_conn_fail.execute.side_effect = prog_error
        mock_engine_fail = MagicMock()
        mock_engine_fail.begin = MagicMock(return_value=_make_async_cm(mock_conn_fail))

        mock_conn_ok = AsyncMock()
        mock_conn_ok.execute = AsyncMock()
        mock_engine_ok = MagicMock()
        mock_engine_ok.begin = MagicMock(return_value=_make_async_cm(mock_conn_ok))

        call_count = 0

        def create_engine_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return mock_engine_fail
            return mock_engine_ok

        with (
            patch(
                "w2v_common.database.connection.create_async_engine",
                side_effect=create_engine_side_effect,
            ),
            patch("w2v_common.database.connection._ensure_database_exists", new_callable=AsyncMock),
        ):
            result = await create_engine_with_retry(max_retries=2, retry_delay=0.01)
            assert result is mock_engine_ok

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self):
        from src.database.connection import create_engine_with_retry

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = TimeoutError("timed out")
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_make_async_cm(mock_conn))

        with patch("w2v_common.database.connection.create_async_engine", return_value=mock_engine):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await create_engine_with_retry(max_retries=1, retry_delay=0.01)

    @pytest.mark.asyncio
    async def test_generic_exception_retries(self):
        from src.database.connection import create_engine_with_retry

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = RuntimeError("generic error")
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_make_async_cm(mock_conn))

        with patch("w2v_common.database.connection.create_async_engine", return_value=mock_engine):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await create_engine_with_retry(max_retries=1, retry_delay=0.01)


class TestGetEngine:
    """Cover line 177: get_engine."""

    @pytest.mark.asyncio
    async def test_get_engine_creates_on_first_call(self):
        import w2v_common.database.connection as conn_mod

        original = conn_mod._engine
        conn_mod._engine = None
        try:
            mock_engine = AsyncMock()
            with patch.object(
                conn_mod, "create_engine_with_retry", AsyncMock(return_value=mock_engine)
            ):
                result = await conn_mod.get_engine()
                assert result is mock_engine
                assert conn_mod._engine is mock_engine
        finally:
            conn_mod._engine = original


# ============================================================================
# ASGI - lifespan, root, error handler, main
# ============================================================================


class TestASGILifespan:
    """Cover lines 62-182: lifespan function."""

    @pytest.mark.asyncio
    async def test_lifespan_success(self):
        from src.asgi import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_make_async_cm(mock_conn))

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_registry = MagicMock()
        mock_registry.provider_names = ["kokoro"]
        mock_registry.close_all = AsyncMock()

        mock_storage = AsyncMock()

        mock_db_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        with (
            patch("src.asgi.get_engine", AsyncMock(return_value=mock_engine)),
            patch("src.asgi.init_redis", AsyncMock(return_value=mock_redis)),
            patch("src.asgi.get_redis_client", return_value=mock_redis),
            patch("src.asgi.close_redis", AsyncMock()),
            patch("src.asgi.close_engine", AsyncMock()),
            patch("src.asgi.create_registry", return_value=mock_registry),
            patch("src.asgi.StorageService", return_value=mock_storage),
            patch("src.asgi.TTSService"),
            patch("src.database.models.Base") as mock_base,
            patch(
                "src.database.session.get_db_session",
                return_value=_make_async_cm(mock_db_session),
            ),
        ):
            mock_base.metadata.create_all = MagicMock()

            async with lifespan(mock_app):
                assert mock_app.state.db_engine is mock_engine
                assert mock_app.state.provider_registry is mock_registry

    @pytest.mark.asyncio
    async def test_lifespan_db_failure(self):
        from src.asgi import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        with patch("src.asgi.get_engine", AsyncMock(side_effect=RuntimeError("db fail"))):
            with pytest.raises(RuntimeError, match="db fail"):
                async with lifespan(mock_app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_redis_failure_continues(self):
        from src.asgi import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_make_async_cm(mock_conn))

        mock_registry = MagicMock()
        mock_registry.provider_names = ["kokoro"]
        mock_registry.close_all = AsyncMock()
        mock_storage = AsyncMock()

        mock_db_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        with (
            patch("src.asgi.get_engine", AsyncMock(return_value=mock_engine)),
            patch("src.asgi.init_redis", AsyncMock(side_effect=RuntimeError("redis fail"))),
            patch("src.asgi.close_redis", AsyncMock()),
            patch("src.asgi.close_engine", AsyncMock()),
            patch("src.asgi.create_registry", return_value=mock_registry),
            patch("src.asgi.StorageService", return_value=mock_storage),
            patch("src.asgi.TTSService"),
            patch("src.database.models.Base") as mock_base,
            patch(
                "src.database.session.get_db_session",
                return_value=_make_async_cm(mock_db_session),
            ),
        ):
            mock_base.metadata.create_all = MagicMock()

            async with lifespan(mock_app):
                pass


class TestASGIRoot:
    """Cover lines 220-238: root endpoint."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        from src.asgi import root

        result = await root()
        assert result["name"] == "TTS Service"
        assert "endpoints" in result
        assert result["status"] == "running"


class TestASGIMain:
    """Cover lines 285-287, 297: main function."""

    def test_main(self):
        mock_uvicorn = MagicMock()
        mock_uvicorn.run = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            from src.asgi import main

            main()
            mock_uvicorn.run.assert_called_once()


# ============================================================================
# Redis Client - connect, disconnect pubsub, module-level functions
# ============================================================================


class TestRedisClientBehavior:
    """Real-behavior tests for the tts_service ``RedisClient`` extension.

    Uses fakeredis in place of a live Redis server. Two layers are injected:
      - The base client / pool (``_client`` / ``_pool``) via ``inject_fakeredis``
        (shared helper in ``w2v_common.testing``).
      - The binary and pubsub clients, which the tts extension adds, are
        replaced with fresh fakeredis instances so ``set_bytes``/``get_bytes``
        and ``publish``/``get_pubsub`` exercise real Redis semantics.

    This replaces a previous mock-heavy suite that only verified that
    ``ConnectionPool.from_url`` was called and the private attributes were
    set — assertions that could not catch a wire-format regression.
    """

    def _build_fake_client(self):  # type: ignore[no-untyped-def]
        import fakeredis
        from w2v_common.testing import inject_fakeredis

        from src.cache.redis_client import RedisClient

        client = RedisClient(redis_url="redis://localhost:6379/0")
        inject_fakeredis(client)  # Sets _client / _pool with a decoded fakeredis
        # Binary and pubsub clients also run against fakeredis.
        binary_fake = fakeredis.aioredis.FakeRedis(decode_responses=False)
        client._binary_client = binary_fake
        client._binary_pool = binary_fake.connection_pool
        pubsub_fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
        client._pubsub_client = pubsub_fake
        client._pubsub_pool = pubsub_fake.connection_pool
        return client

    @pytest.mark.asyncio
    async def test_set_bytes_then_get_bytes_round_trips(self) -> None:
        """Binary payload round-trip through fakeredis.

        Tripwire for ``redis_client.py:95-116`` — a regression like "decode
        responses on the binary client" would break the round-trip because
        fakeredis would return ``str`` instead of ``bytes``.
        """
        client = self._build_fake_client()
        payload = b"\x00\x01\x02\xffbinary_audio_blob"

        stored = await client.set_bytes("audio:job-1", payload)
        assert stored is True

        fetched = await client.get_bytes("audio:job-1")
        assert fetched == payload
        assert isinstance(fetched, bytes)  # Must be bytes, not str.

    @pytest.mark.asyncio
    async def test_set_bytes_with_ttl_is_retrievable_before_expiry(self) -> None:
        """``ttl`` must propagate to ``SETEX`` (not ``SET``) so the key is set
        with an expiry but still readable immediately.
        """
        client = self._build_fake_client()
        ok = await client.set_bytes("audio:ephem", b"short-lived", ttl=60)
        assert ok is True
        assert await client.get_bytes("audio:ephem") == b"short-lived"

    @pytest.mark.asyncio
    async def test_get_bytes_missing_key_returns_none(self) -> None:
        client = self._build_fake_client()
        assert await client.get_bytes("nope") is None

    @pytest.mark.asyncio
    async def test_get_bytes_before_connect_raises(self) -> None:
        """``_ensure_binary_connected`` must raise when the binary client was
        never initialised (``redis_client.py:86-89``)."""
        from src.cache.redis_client import RedisClient

        client = RedisClient(redis_url="redis://localhost:6379/0")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.get_bytes("x")

    @pytest.mark.asyncio
    async def test_publish_with_active_subscriber_delivers_message(self) -> None:
        """Full pub/sub round-trip: subscriber receives the exact message.

        This crosses both ``publish`` (on the base client) and ``get_pubsub``
        (on the pubsub client). Catches regressions that would disconnect the
        two pools or mis-name the channel.
        """
        import asyncio

        client = self._build_fake_client()

        # Both publish and get_pubsub should use the same fakeredis instance
        # so the subscriber sees the message. Align them explicitly.
        shared = client._pubsub_client
        client._client = shared  # route publish() through the same fake
        client._pool = shared.connection_pool  # type: ignore[assignment]

        pubsub = client.get_pubsub()
        await pubsub.subscribe("channel-1")

        # Drain the subscription-confirmation message.
        await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)

        delivered = await client.publish("channel-1", "hello-world")
        assert delivered >= 1  # At least one subscriber received it.

        # Give the pubsub loop a tick.
        msg: dict[str, object] | None = None
        for _ in range(5):
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if msg is not None:
                break
            await asyncio.sleep(0.01)
        assert msg is not None
        assert msg["data"] == "hello-world"
        assert msg["channel"] == "channel-1"

        await pubsub.unsubscribe("channel-1")
        await pubsub.aclose()

    def test_get_pubsub_before_connect_raises(self) -> None:
        """``get_pubsub`` must raise before ``connect`` has initialised the pool
        (``redis_client.py:138-139``).
        """
        from src.cache.redis_client import RedisClient

        client = RedisClient(redis_url="redis://localhost:6379/0")
        with pytest.raises(RuntimeError, match="not connected"):
            client.get_pubsub()

    @pytest.mark.asyncio
    async def test_disconnect_clears_all_pool_attributes(self) -> None:
        """After ``disconnect``, every extension pool/client is ``None``.

        Catches regressions in the disconnect chain at ``redis_client.py:70-84``
        — dropping the ``_binary_*`` or ``_pubsub_*`` reset would show up here.
        """
        client = self._build_fake_client()
        await client.disconnect()

        assert client._binary_client is None
        assert client._binary_pool is None
        assert client._pubsub_client is None
        assert client._pubsub_pool is None


class TestRedisModuleLevelFunctions:
    """Cover lines 242-257: module-level init/close functions."""

    @pytest.mark.asyncio
    async def test_init_redis(self):
        import src.cache.redis_client as redis_mod

        mock_client = AsyncMock()
        redis_mod.get_redis_client.cache_clear()

        with patch("src.cache.redis_client.get_redis_client", return_value=mock_client):
            result = await redis_mod.init_redis()
            mock_client.connect.assert_called_once()
            assert result is mock_client

    @pytest.mark.asyncio
    async def test_close_redis(self):
        import src.cache.redis_client as redis_mod

        mock_client = AsyncMock()
        redis_mod.get_redis_client.cache_clear()
        original = redis_mod._redis_client
        redis_mod._redis_client = mock_client
        try:
            await redis_mod.close_redis()
            mock_client.disconnect.assert_called_once()
            assert redis_mod._redis_client is None
        finally:
            redis_mod._redis_client = original
            redis_mod.get_redis_client.cache_clear()

    @pytest.mark.asyncio
    async def test_close_redis_when_none(self):
        import src.cache.redis_client as redis_mod

        redis_mod.get_redis_client.cache_clear()
        original = redis_mod._redis_client
        redis_mod._redis_client = None
        try:
            await redis_mod.close_redis()
        finally:
            redis_mod._redis_client = original
            redis_mod.get_redis_client.cache_clear()


# ============================================================================
# TTS Service - additional coverage
# ============================================================================


class TestTTSServiceResolveProviderAutoWithInvalidStrategy:
    """Cover lines 86-87: invalid routing strategy defaults to BALANCED."""

    def _make_service(self, router=None):
        from src.services.tts_service import TTSService

        registry = MagicMock()
        storage = MagicMock()
        return TTSService(registry=registry, storage=storage, router=router)

    @pytest.mark.asyncio
    async def test_auto_with_invalid_strategy(self):
        from src.api.schemas import RoutingStrategy
        from src.services.router import ProviderRouter, RoutingDecision

        mock_router = AsyncMock(spec=ProviderRouter)
        decision = RoutingDecision(
            provider_name="openai",
            voice="alloy",
            strategy_used=RoutingStrategy.BALANCED,
            reason="test",
            fallback_chain=[],
        )
        mock_router.select_provider = AsyncMock(return_value=decision)

        service = self._make_service(router=mock_router)
        mock_provider = MagicMock()
        service._registry.get_provider.return_value = mock_provider

        _p, name = await service._resolve_provider("auto", "Hello", "invalid_strategy")
        assert name == "openai"


class TestTTSServiceNotifyWithRealNS:
    """Cover lines 251-259: _notify with real NotificationService."""

    @pytest.mark.asyncio
    async def test_notify_with_notification_service(self):
        from src.services.notification_service import NotificationService
        from src.services.tts_service import TTSService

        mock_ns = AsyncMock(spec=NotificationService)
        mock_ns.publish_event = AsyncMock()

        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage, notification_service=mock_ns)

        await service._notify("job_started", "j1", tenant_uuid("t1"), extra_data="test")
        mock_ns.publish_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_with_ns_publish_error(self):
        from src.services.notification_service import NotificationService
        from src.services.tts_service import TTSService

        mock_ns = AsyncMock(spec=NotificationService)
        mock_ns.publish_event = AsyncMock(side_effect=RuntimeError("publish fail"))

        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage, notification_service=mock_ns)

        await service._notify("job_started", "j1", tenant_uuid("t1"))


class TestTTSServiceApplyPostProcessing:
    """Cover lines 228-237: _apply_post_processing with real PostProcessingConfig."""

    @pytest.mark.asyncio
    async def test_apply_post_processing_no_changes(self):
        from src.api.schemas import NormalizationConfig, PostProcessingConfig
        from src.services.tts_service import TTSService

        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        pp = PostProcessingConfig(normalization=NormalizationConfig(enabled=False))
        assert pp.has_any_processing() is False

        audio_data, duration = await service._apply_post_processing(b"audio", "mp3", 24000, pp)
        assert audio_data == b"audio"
        assert duration == 0.0

    @pytest.mark.asyncio
    async def test_apply_post_processing_with_changes(self):
        from src.api.schemas import NormalizationConfig, PostProcessingConfig
        from src.services.tts_service import TTSService

        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        pp = PostProcessingConfig(normalization=NormalizationConfig(enabled=True))
        assert pp.has_any_processing() is True

        mock_processor = MagicMock()
        mock_pp_result = MagicMock()
        mock_pp_result.audio_data = b"processed_audio"
        mock_pp_result.duration_seconds = 2.5
        mock_processor.process.return_value = mock_pp_result

        with patch("src.utils.post_processor.AudioPostProcessor", return_value=mock_processor):
            audio_data, duration = await service._apply_post_processing(b"audio", "mp3", 24000, pp)
            assert audio_data == b"processed_audio"
            assert duration == 2.5


class TestTTSServiceGenerateSyncWithQuota:
    """Cover lines 295-297: generate_sync with quota check."""

    @pytest.mark.asyncio
    async def test_generate_sync_with_db_and_quota(self):
        """The pre-flight ceiling must receive the request's REAL projected cost.

        This used to assert only ``"projected_cost" in call.kwargs`` against a
        free provider — vacuously true even when every request projected $0.00
        and the monthly-budget ceiling could therefore never fire. Pin the
        VALUE, on a PRICED provider.
        """
        from src.services.tts_service import TTSService

        registry = MagicMock()
        storage = MagicMock()
        service = TTSService(registry=registry, storage=storage)

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="rachel")
        mock_provider.list_voices = AsyncMock(return_value=[])
        mock_provider.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
            )
        )
        registry.get_provider.return_value = mock_provider

        db = AsyncMock()
        text = "x" * 10_000

        with patch("src.services.quota_service.check_quota", new_callable=AsyncMock) as mock_quota:
            result, _meta = await service.generate_sync(
                text=text, provider="elevenlabs", db=db, tenant_id=tenant_uuid("t1"), no_cache=True
            )
            mock_quota.assert_called_once()
            call = mock_quota.call_args
            assert call.args == (db, tenant_uuid("t1"), 10_000)
            # elevenlabs = $0.030 / 1k chars * 10k chars = $0.30
            assert call.kwargs["projected_cost"] == pytest.approx(0.30)
            assert result.audio_data == b"audio"

    @pytest.mark.asyncio
    async def test_generate_sync_quota_projection_is_never_zero_for_auto(self):
        """``provider="auto"`` is the DEFAULT and must not project $0.00."""
        from src.api.schemas import RoutingStrategy
        from src.services.router import ProviderRouter, RoutingDecision
        from src.services.tts_service import TTSService

        mock_provider = AsyncMock()
        mock_provider.is_available = AsyncMock(return_value=True)
        mock_provider.get_default_voice = MagicMock(return_value="rachel")
        mock_provider.list_voices = AsyncMock(return_value=[])
        mock_provider.generate = AsyncMock(
            return_value=GenerationResult(
                audio_data=b"audio", format="mp3", sample_rate=24000, duration_seconds=1.0
            )
        )
        registry = MagicMock()
        registry.get_provider.return_value = mock_provider

        router = AsyncMock(spec=ProviderRouter)
        router.select_provider = AsyncMock(
            return_value=RoutingDecision(
                provider_name="elevenlabs",
                voice="rachel",
                strategy_used=RoutingStrategy.BALANCED,
                reason="test",
                fallback_chain=[],
            )
        )
        service = TTSService(registry=registry, storage=MagicMock(), router=router)

        with patch("src.services.quota_service.check_quota", new_callable=AsyncMock) as mock_quota:
            await service.generate_sync(
                text="x" * 10_000,
                provider="auto",
                db=AsyncMock(),
                tenant_id=tenant_uuid("t1"),
                no_cache=True,
            )

        assert mock_quota.call_args.kwargs["projected_cost"] == pytest.approx(0.30)


# ============================================================================
# API TTS - async mode, streaming, batch, quota, batch audio download
# ============================================================================


class TestGenerateAsyncMode:
    """Cover lines 84-97: generate with async_mode=True."""

    @pytest.mark.asyncio
    async def test_generate_async_mode(self):
        from src.api.schemas import AsyncJobResponse, TTSGenerateRequest
        from src.api.tts import generate

        body = TTSGenerateRequest(text="Hello", provider="kokoro", async_mode=True)
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(return_value="job-123")
        request.app.state.tts_service = mock_tts

        mock_registry = MagicMock()
        request.app.state.provider_registry = mock_registry

        db = AsyncMock()

        with patch("src.api.tts.asyncio.create_task", side_effect=_fake_create_task) as mock_task:
            result = await generate(body, request, db)
            assert isinstance(result, AsyncJobResponse)
            assert result.job_id == "job-123"
            assert result.status == "pending"
            mock_task.assert_called_once()


class TestGenerateQuotaExceeded:
    """Cover line 122: quota exceeded error."""

    @pytest.mark.asyncio
    async def test_generate_quota_exceeded(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSGenerateRequest
        from src.api.tts import generate
        from src.services.quota_service import QuotaExceededError

        body = TTSGenerateRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")
        request.headers = {}

        mock_tts = AsyncMock()
        mock_tts.generate_sync = AsyncMock(
            side_effect=QuotaExceededError("quota exceeded", "characters", 1000, 500)
        )
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate(body, request, db)
        assert exc_info.value.status_code == 429


class TestStreamGenerateEndpoints:
    """Cover lines 195-233: stream_generate SSE endpoint."""

    @pytest.mark.asyncio
    async def test_stream_generate_returns_streaming_response(self):
        from fastapi.responses import StreamingResponse

        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")

        mock_tts = AsyncMock()
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        result = await stream_generate(body, request)
        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_stream_generate_unknown_provider(self):
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


class TestStreamGenerateRawEndpoint:
    """Cover lines 267-277: stream_generate_raw."""

    @pytest.mark.asyncio
    async def test_stream_generate_raw_returns_streaming_response(self):
        from fastapi.responses import StreamingResponse

        from src.api.schemas import TTSStreamRequest
        from src.api.tts import stream_generate_raw

        body = TTSStreamRequest(text="Hello", provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")

        mock_tts = AsyncMock()
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        result = await stream_generate_raw(body, request)
        assert isinstance(result, StreamingResponse)
        assert result.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_stream_generate_raw_unknown_provider(self):
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


class TestGenerateBatchEndpoint:
    """Cover lines 330-335: batch endpoint error paths."""

    @pytest.mark.asyncio
    async def test_generate_batch_success(self):
        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(
            items=[TTSBatchItem(id="item1", text="Hello"), TTSBatchItem(id="item2", text="World")],
            provider="kokoro",
        )
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(side_effect=["job-1", "job-2"])
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        db = AsyncMock()

        with patch("src.api.tts.asyncio.create_task", side_effect=_fake_create_task):
            result = await generate_batch(body, request, db)
            assert result.status == "pending"
            assert len(result.job_ids) == 2

    @pytest.mark.asyncio
    async def test_generate_batch_unknown_provider(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(
            items=[TTSBatchItem(id="item1", text="Hello")], provider="nonexistent"
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
    async def test_generate_batch_quota_exceeded(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch
        from src.services.quota_service import QuotaExceededError

        body = TTSBatchRequest(items=[TTSBatchItem(id="item1", text="Hello")], provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(
            side_effect=QuotaExceededError("quota", "characters", 1000, 500)
        )
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate_batch(body, request, db)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_generate_batch_runtime_error(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(items=[TTSBatchItem(id="item1", text="Hello")], provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(side_effect=RuntimeError("provider down"))
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate_batch(body, request, db)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_generate_batch_value_error(self):
        from fastapi import HTTPException

        from src.api.schemas import TTSBatchItem, TTSBatchRequest
        from src.api.tts import generate_batch

        body = TTSBatchRequest(items=[TTSBatchItem(id="item1", text="Hello")], provider="kokoro")
        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        request.state.tenant_id = tenant_uuid("t1")

        mock_tts = AsyncMock()
        mock_tts.generate_async = AsyncMock(side_effect=ValueError("bad input"))
        request.app.state.tts_service = mock_tts
        request.app.state.provider_registry = MagicMock()

        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await generate_batch(body, request, db)
        assert exc_info.value.status_code == 400


class TestDownloadBatchAudio:
    """Cover lines 386-424: download_batch_audio endpoint."""

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
                await download_batch_audio("batch-1", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_concat_not_available(self):
        from fastapi import HTTPException

        from src.api.tts import download_batch_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.concatenated_audio_path = None
        mock_job.output_format = "mp3"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
            with pytest.raises(HTTPException) as exc_info:
                await download_batch_audio("batch-1", request, db)
            assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_concat_audio_success(self):
        from src.api.tts import download_batch_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.retrieve = AsyncMock(return_value=b"concat_audio")
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.concatenated_audio_path = "/path/concat.mp3"
        mock_job.output_format = "mp3"

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])
            result = await download_batch_audio("batch-1", request, db)
            assert result.body == b"concat_audio"
            assert result.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_concat_audio_file_missing(self):
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


class TestDownloadAudioMissingPath:
    """Cover lines for download_audio when audio_path is None or storage returns None."""

    @pytest.mark.asyncio
    async def test_audio_no_path(self):
        from fastapi import HTTPException

        from src.api.tts import download_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = None

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            with pytest.raises(HTTPException) as exc_info:
                await download_audio("j1", request, db)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_audio_storage_returns_none(self):
        from fastapi import HTTPException

        from src.api.tts import download_audio

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.retrieve = AsyncMock(return_value=None)
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.audio_path = "/path/audio.mp3"
        mock_job.output_format = "mp3"
        mock_job.audio_duration_seconds = 1.0

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            with pytest.raises(HTTPException) as exc_info:
                await download_audio("j1", request, db)
            assert exc_info.value.status_code == 404


class TestGetAudioPresignedUrlNotCompleted:
    """Cover line 573: job not completed."""

    @pytest.mark.asyncio
    async def test_presigned_url_not_completed(self):
        from fastapi import HTTPException

        from src.api.tts import get_audio_presigned_url

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "pending"
        mock_job.audio_path = None

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            with pytest.raises(HTTPException) as exc_info:
                await get_audio_presigned_url("j1", request, db)
            assert exc_info.value.status_code == 409


class TestGetJobStatusCompleted:
    """Cover lines for job status with completed job and presigned URL."""

    @pytest.mark.asyncio
    async def test_job_completed_with_presigned_url(self):
        from src.api.tts import get_job_status

        request = MagicMock()
        request.state.tenant_id = tenant_uuid("tenant-test")
        mock_storage = AsyncMock()
        mock_storage.get_presigned_url = AsyncMock(
            return_value="https://minio.example.com/audio.mp3"
        )
        request.app.state.storage_service = mock_storage
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "j1"
        mock_job.status = "completed"
        mock_job.provider = "openai"
        mock_job.voice = "alloy"
        mock_job.created_at = MagicMock()
        mock_job.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_job.completed_at = MagicMock()
        mock_job.completed_at.isoformat.return_value = "2026-01-01T00:01:00"
        mock_job.error_message = None
        mock_job.audio_path = "/audio/j1.mp3"
        mock_job.audio_duration_seconds = 2.5
        mock_job.audio_size_bytes = 50000
        mock_job.cached = False
        mock_job.detected_language = "en"
        mock_job.routing_strategy = "balanced"
        mock_job.estimated_cost_usd = 0.003

        with patch("src.api.tts.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            result = await get_job_status("j1", request, db)
            assert result.job_id == "j1"
            assert result.status == "completed"
            assert result.presigned_url == "https://minio.example.com/audio.mp3"
            assert result.audio_url == "/api/tts/jobs/j1/audio"


class TestConcatenateBatchBackground:
    """Cover lines 377-378: _concatenate_batch_background."""

    @pytest.mark.asyncio
    async def test_concatenate_batch_background_success(self):
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()
        mock_db = AsyncMock()

        mock_job_pending = MagicMock()
        mock_job_pending.status = "pending"

        mock_job_completed = MagicMock()
        mock_job_completed.status = "completed"

        call_count = 0

        async def mock_get_batch_jobs(db, batch_id, *, tenant_id):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return [mock_job_pending]
            return [mock_job_completed]

        with (
            patch("src.api.tts.get_db_session") as mock_get_db,
            patch("src.api.tts.JobManager") as mock_jm,
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_jm.get_batch_jobs = AsyncMock(side_effect=mock_get_batch_jobs)

            await _concatenate_batch_background(
                mock_tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )

            mock_tts.concatenate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_concatenate_batch_background_no_jobs(self):
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()
        mock_db = AsyncMock()

        with (
            patch("src.api.tts.get_db_session") as mock_get_db,
            patch("src.api.tts.JobManager") as mock_jm,
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_jm.get_batch_jobs = AsyncMock(return_value=[])

            await _concatenate_batch_background(
                mock_tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )

            mock_tts.concatenate_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_concatenate_batch_background_exception(self):
        from src.api.tts import _concatenate_batch_background

        mock_tts = AsyncMock()
        mock_tts.concatenate_batch = AsyncMock(side_effect=RuntimeError("concat fail"))
        mock_db = AsyncMock()

        mock_job = MagicMock()
        mock_job.status = "completed"

        with (
            patch("src.api.tts.get_db_session") as mock_get_db,
            patch("src.api.tts.JobManager") as mock_jm,
            patch("src.api.tts.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_jm.get_batch_jobs = AsyncMock(return_value=[mock_job])

            # Should not raise
            await _concatenate_batch_background(
                mock_tts, "batch-1", "mp3", 0, tenant_id=tenant_uuid("tenant-a")
            )
