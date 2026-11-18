"""Unit tests for audio concatenation helper functions."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.audio import _concat_demuxer, _concat_with_gaps


class TestConcatDemuxer:
    @patch("subprocess.run")
    def test_successful_concat(self, mock_run, tmp_path):
        # Create segment files
        seg1 = tmp_path / "seg_0000.mp3"
        seg2 = tmp_path / "seg_0001.mp3"
        seg1.write_bytes(b"seg1")
        seg2.write_bytes(b"seg2")

        # Create expected output
        output_path = tmp_path / "output.mp3"
        output_path.write_bytes(b"concatenated")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = _concat_demuxer([seg1, seg2], tmp_path, "mp3")
        assert result == b"concatenated"

    @patch("subprocess.run")
    def test_concat_failure(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.mp3"
        seg1.write_bytes(b"seg1")

        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        with pytest.raises(RuntimeError, match="concatenation failed"):
            _concat_demuxer([seg1], tmp_path, "mp3")

    @patch("subprocess.run")
    def test_concat_empty_output(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.mp3"
        seg1.write_bytes(b"seg1")

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # Don't create output file

        with pytest.raises(RuntimeError, match="empty output"):
            _concat_demuxer([seg1], tmp_path, "mp3")


class TestConcatWithGaps:
    @patch("subprocess.run")
    def test_successful_concat_with_gaps(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.mp3"
        seg2 = tmp_path / "seg_0001.mp3"
        seg1.write_bytes(b"seg1")
        seg2.write_bytes(b"seg2")

        output_path = tmp_path / "output.mp3"
        output_path.write_bytes(b"with gaps")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = _concat_with_gaps([seg1, seg2], tmp_path, "mp3", 500)
        assert result == b"with gaps"

    @patch("subprocess.run")
    def test_concat_with_gaps_failure(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.mp3"
        seg1.write_bytes(b"seg1")

        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        with pytest.raises(RuntimeError, match="concatenation with gaps failed"):
            _concat_with_gaps([seg1], tmp_path, "mp3", 500)

    @patch("subprocess.run")
    def test_concat_with_gaps_empty_output(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.mp3"
        seg1.write_bytes(b"seg1")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with pytest.raises(RuntimeError, match="empty output"):
            _concat_with_gaps([seg1], tmp_path, "mp3", 500)

    @patch("subprocess.run")
    def test_wav_format(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.wav"
        seg1.write_bytes(b"seg1")

        output_path = tmp_path / "output.wav"
        output_path.write_bytes(b"wav output")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = _concat_with_gaps([seg1], tmp_path, "wav", 0)
        assert result == b"wav output"

        # Verify wav codec args
        cmd = mock_run.call_args[0][0]
        assert "pcm_s16le" in cmd

    @patch("subprocess.run")
    def test_mp3_format(self, mock_run, tmp_path):
        seg1 = tmp_path / "seg_0000.mp3"
        seg2 = tmp_path / "seg_0001.mp3"
        seg1.write_bytes(b"seg1")
        seg2.write_bytes(b"seg2")

        output_path = tmp_path / "output.mp3"
        output_path.write_bytes(b"mp3 output")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = _concat_with_gaps([seg1, seg2], tmp_path, "mp3", 200)
        assert result == b"mp3 output"

        cmd = mock_run.call_args[0][0]
        assert "libmp3lame" in cmd
