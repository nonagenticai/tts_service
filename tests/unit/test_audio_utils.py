"""Unit tests for audio utility functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.audio import (
    concatenate_audio,
    is_ffmpeg_available,
    probe_duration,
    wav_to_mp3,
)


class TestIsFFmpegAvailable:
    @patch("shutil.which")
    def test_ffmpeg_available(self, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        assert is_ffmpeg_available() is True

    @patch("shutil.which")
    def test_ffmpeg_not_available(self, mock_which):
        mock_which.return_value = None
        assert is_ffmpeg_available() is False


class TestWavToMp3:
    @patch("shutil.which")
    def test_raises_if_ffmpeg_missing(self, mock_which):
        mock_which.return_value = None
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            wav_to_mp3(Path("/tmp/test.wav"))

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_successful_conversion(self, mock_run, mock_which, tmp_path):
        wav_path = tmp_path / "test.wav"
        mp3_path = tmp_path / "test.mp3"
        wav_path.write_bytes(b"fake wav data")

        # Simulate successful ffmpeg conversion
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # Create output file to simulate ffmpeg creating it
        mp3_path.write_bytes(b"fake mp3 data")

        result = wav_to_mp3(wav_path, mp3_path)
        assert result == mp3_path
        mock_run.assert_called_once()

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_default_output_path(self, mock_run, mock_which, tmp_path):
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"fake wav data")
        expected_mp3 = tmp_path / "test.mp3"
        expected_mp3.write_bytes(b"fake mp3 data")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = wav_to_mp3(wav_path)
        assert result == expected_mp3

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_failed_conversion(self, mock_run, mock_which, tmp_path):
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"fake wav data")

        mock_run.return_value = MagicMock(returncode=1, stderr="error occurred")

        with pytest.raises(RuntimeError, match="conversion failed"):
            wav_to_mp3(wav_path)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_empty_output_raises(self, mock_run, mock_which, tmp_path):
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"fake wav data")

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # Don't create output file -> should raise

        with pytest.raises(RuntimeError, match="empty or missing"):
            wav_to_mp3(wav_path)


class TestProbeDuration:
    @patch("subprocess.run")
    def test_successful_probe(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"format": {"duration": "5.123"}}',
        )
        result = probe_duration(Path("/tmp/test.mp3"))
        assert result is not None
        assert abs(result - 5.123) < 0.001

    @patch("subprocess.run")
    def test_failed_probe(self, mock_run):
        # UNMEASURED, not a measured zero. See tests/unit/test_unmeasured_duration.py.
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = probe_duration(Path("/tmp/test.mp3"))
        assert result is None

    @patch("subprocess.run")
    def test_invalid_json(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")
        result = probe_duration(Path("/tmp/test.mp3"))
        assert result is None

    @patch("subprocess.run")
    def test_missing_duration_field(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"format": {}}',
        )
        result = probe_duration(Path("/tmp/test.mp3"))
        assert result is None


class TestConcatenateAudio:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="No audio segments"):
            concatenate_audio([])

    def test_single_segment_returns_it(self):
        result = concatenate_audio([b"audio data"])
        assert result == b"audio data"

    @patch("shutil.which")
    def test_raises_if_ffmpeg_missing(self, mock_which):
        mock_which.return_value = None
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            concatenate_audio([b"seg1", b"seg2"])

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.utils.audio._concat_demuxer")
    def test_no_gaps_uses_demuxer(self, mock_demuxer, mock_which):
        mock_demuxer.return_value = b"concatenated"
        result = concatenate_audio([b"seg1", b"seg2"], output_format="mp3", gap_ms=0)
        assert result == b"concatenated"
        mock_demuxer.assert_called_once()

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.utils.audio._concat_with_gaps")
    def test_with_gaps_uses_gap_concat(self, mock_gap_concat, mock_which):
        mock_gap_concat.return_value = b"with-gaps"
        result = concatenate_audio([b"seg1", b"seg2"], output_format="mp3", gap_ms=500)
        assert result == b"with-gaps"
        mock_gap_concat.assert_called_once()
