"""Unit tests for the audio post processor."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.schemas import (
    BitrateConfig,
    FadeConfig,
    NormalizationConfig,
    PostProcessingConfig,
    SampleRateConfig,
    SilenceConfig,
    TrimConfig,
)
from src.utils.post_processor import AudioPostProcessor, PostProcessingResult


class TestPostProcessingResult:
    def test_creation(self):
        result = PostProcessingResult(
            audio_data=b"audio",
            format="mp3",
            sample_rate=24000,
            duration_seconds=1.5,
        )
        assert result.audio_data == b"audio"
        assert result.format == "mp3"
        assert result.sample_rate == 24000
        assert result.duration_seconds == 1.5


class TestAudioPostProcessorInit:
    @patch("shutil.which", return_value=None)
    def test_raises_if_ffmpeg_missing(self, mock_which):
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            AudioPostProcessor()

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_creates_with_ffmpeg(self, mock_which):
        processor = AudioPostProcessor()
        assert processor is not None


class TestAudioPostProcessorProcess:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_no_processing_returns_input(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(
            normalization=NormalizationConfig(enabled=False)
        )  # Explicitly disable all processing

        result = processor.process(b"audio", "mp3", 24000, config)
        assert result.audio_data == b"audio"
        assert result.format == "mp3"
        assert result.sample_rate == 24000

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_process_with_normalization(self, mock_run, mock_which, tmp_path):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(
            normalization=NormalizationConfig(enabled=True, target_lufs=-16.0)
        )

        # Mock subprocess for both ffmpeg and ffprobe
        def subprocess_side_effect(cmd, **kwargs):
            # Check if this is ffmpeg or ffprobe
            if "ffmpeg" in cmd[0]:
                # Write output file
                for i, arg in enumerate(cmd):
                    if arg.endswith(".mp3") and i == len(cmd) - 1:
                        Path(arg).write_bytes(b"processed audio")
                return MagicMock(returncode=0, stderr="", stdout="")
            # ffprobe
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "1.5"}}),
            )

        mock_run.side_effect = subprocess_side_effect

        result = processor.process(b"audio", "mp3", 24000, config)
        assert result.audio_data == b"processed audio"

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_process_with_silence(self, mock_run, mock_which, tmp_path):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(silence=SilenceConfig(leading_ms=500, trailing_ms=300))

        def subprocess_side_effect(cmd, **kwargs):
            if "ffmpeg" in cmd[0]:
                for i, arg in enumerate(cmd):
                    if arg.endswith(".mp3") and i == len(cmd) - 1:
                        Path(arg).write_bytes(b"with silence")
                return MagicMock(returncode=0, stderr="")
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "2.0"}}),
            )

        mock_run.side_effect = subprocess_side_effect

        result = processor.process(b"audio", "mp3", 24000, config)
        assert result.audio_data == b"with silence"

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_process_ffmpeg_failure(self, mock_run, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(normalization=NormalizationConfig(enabled=True))

        mock_run.return_value = MagicMock(returncode=1, stderr="error occurred")

        with pytest.raises(RuntimeError, match="Post-processing failed"):
            processor.process(b"audio", "mp3", 24000, config)


class TestBuildAudioFilters:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_trim_filter(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(trim=TrimConfig(enabled=True, threshold_db=-40.0))
        filters = processor._build_audio_filters(config)
        assert any("silenceremove" in f for f in filters)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_normalization_filter(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(
            normalization=NormalizationConfig(enabled=True, target_lufs=-16.0, true_peak=-1.5)
        )
        filters = processor._build_audio_filters(config)
        assert any("loudnorm" in f for f in filters)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_fade_in_filter(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(fade=FadeConfig(fade_in_ms=500))
        filters = processor._build_audio_filters(config)
        assert any("afade=t=in" in f for f in filters)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_fade_out_filter(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(fade=FadeConfig(fade_out_ms=300))
        filters = processor._build_audio_filters(config)
        assert any("afade=t=out" in f for f in filters)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_sample_rate_filter(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(sample_rate=SampleRateConfig(target_rate=48000))
        filters = processor._build_audio_filters(config)
        assert any("aresample=48000" in f for f in filters)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_default_config_includes_normalization(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig()
        filters = processor._build_audio_filters(config)
        assert any("loudnorm" in f for f in filters)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_no_filters_when_normalization_disabled(self, mock_which):
        processor = AudioPostProcessor()
        config = PostProcessingConfig(normalization=NormalizationConfig(enabled=False))
        filters = processor._build_audio_filters(config)
        assert filters == []


class TestOutputEncodingArgs:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_mp3_default_vbr(self, mock_which):
        config = PostProcessingConfig()
        args = AudioPostProcessor._output_encoding_args("mp3", config)
        assert "-codec:a" in args
        assert "libmp3lame" in args
        assert "-qscale:a" in args

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_mp3_with_bitrate(self, mock_which):
        config = PostProcessingConfig(bitrate=BitrateConfig(bitrate="192k"))
        args = AudioPostProcessor._output_encoding_args("mp3", config)
        assert "-b:a" in args
        assert "192k" in args

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_wav_encoding(self, mock_which):
        config = PostProcessingConfig()
        args = AudioPostProcessor._output_encoding_args("wav", config)
        assert "-codec:a" in args
        assert "pcm_s16le" in args


class TestProbeDuration:
    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_probe_success(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "2.5"}}),
        )
        result = AudioPostProcessor._probe_duration(Path("/tmp/test.mp3"))
        assert abs(result - 2.5) < 0.001

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_probe_failure(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = AudioPostProcessor._probe_duration(Path("/tmp/test.mp3"))
        assert result == 0.0

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_probe_bad_json(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")
        result = AudioPostProcessor._probe_duration(Path("/tmp/test.mp3"))
        assert result == 0.0
