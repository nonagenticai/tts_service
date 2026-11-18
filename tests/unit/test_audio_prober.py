"""Tests for the schema-free audio_prober module.

Mirrors the mock shape of tests/unit/test_audio_metrics.py but exercises the
pure module-level functions (no AudioPostProcessor / schema dependency).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.audio_prober import (
    AudioMetrics,
    loudnorm_measure_filter,
    loudnorm_two_pass_filter,
    measure_audio,
    measure_silence_ratio,
    probe_audio_duration,
)

# ============================================================================
# AudioMetrics dataclass
# ============================================================================


class TestAudioMetrics:
    def test_defaults(self):
        m = AudioMetrics()
        assert m.integrated_lufs is None
        assert m.loudness_range_lu is None
        assert m.true_peak_dbtp is None
        assert m.silence_ratio is None
        assert m.duration_seconds is None
        assert m.threshold_lufs is None


# ============================================================================
# loudnorm filter-string builders
# ============================================================================


class TestLoudnormFilters:
    def test_measure_filter_default_target(self):
        # Defaults must reproduce the historical literal byte-for-byte.
        assert (
            loudnorm_measure_filter() == "loudnorm=I=-16:TP=-1.5:print_format=json:measured_only=1"
        )

    def test_measure_filter_custom_target(self):
        assert (
            loudnorm_measure_filter(target_lufs=-23.0, true_peak=-2.0)
            == "loudnorm=I=-23:TP=-2:print_format=json:measured_only=1"
        )

    def test_two_pass_filter_emits_linear_and_measured(self):
        measured = AudioMetrics(
            integrated_lufs=-18.5,
            loudness_range_lu=6.3,
            true_peak_dbtp=-2.1,
            threshold_lufs=-28.0,
        )
        f = loudnorm_two_pass_filter(measured)

        assert f.startswith("loudnorm=I=-16:TP=-1.5")
        assert "measured_I=-18.5" in f
        assert "measured_TP=-2.1" in f
        assert "measured_LRA=6.3" in f
        assert "measured_thresh=-28.0" in f
        assert "linear=true" in f

    def test_two_pass_filter_custom_target(self):
        measured = AudioMetrics(
            integrated_lufs=-20.0,
            loudness_range_lu=5.0,
            true_peak_dbtp=-3.0,
            threshold_lufs=-30.0,
        )
        f = loudnorm_two_pass_filter(measured, target_lufs=-14.0, true_peak=-1.0)
        assert f.startswith("loudnorm=I=-14:TP=-1")
        assert "linear=true" in f


# ============================================================================
# measure_audio
# ============================================================================


class TestMeasureAudio:
    @patch("subprocess.run")
    def test_measure_audio_success(self, mock_run):
        loudnorm_json = json.dumps(
            {
                "input_i": "-18.5",
                "input_tp": "-2.1",
                "input_lra": "6.3",
                "input_thresh": "-28.0",
                "output_i": "-16.0",
                "output_tp": "-1.5",
                "output_lra": "5.0",
                "output_thresh": "-26.0",
                "normalization_type": "dynamic",
                "target_offset": "0.0",
            }
        )

        def subprocess_side_effect(cmd, **kwargs):
            if "loudnorm" in " ".join(cmd):
                return MagicMock(returncode=0, stderr=f"some noise\n{loudnorm_json}\n", stdout="")
            if "silencedetect" in " ".join(cmd):
                return MagicMock(
                    returncode=0,
                    stderr=(
                        "[silencedetect] silence_start: 0.1\n[silencedetect] silence_end: 0.3\n"
                    ),
                    stdout="",
                )
            if "ffprobe" in cmd[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "3.5"}}),
                )
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = subprocess_side_effect

        metrics = measure_audio(b"fake audio data", "mp3")

        assert metrics.integrated_lufs == -18.5
        assert metrics.loudness_range_lu == 6.3
        assert metrics.true_peak_dbtp == -2.1
        assert metrics.threshold_lufs == -28.0
        assert metrics.duration_seconds == 3.5
        assert metrics.silence_ratio is not None

    @patch("subprocess.run")
    def test_measure_audio_uses_default_target_string(self, mock_run):
        captured: list[list[str]] = []

        def subprocess_side_effect(cmd, **kwargs):
            captured.append(cmd)
            if "ffprobe" in cmd[0]:
                return MagicMock(returncode=0, stdout=json.dumps({"format": {"duration": "1.0"}}))
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = subprocess_side_effect

        measure_audio(b"fake", "mp3")

        loudnorm_cmds = [c for c in captured if "loudnorm" in " ".join(c)]
        assert loudnorm_cmds, "expected a loudnorm ffmpeg invocation"
        assert "loudnorm=I=-16:TP=-1.5:print_format=json:measured_only=1" in loudnorm_cmds[0]

    @patch("subprocess.run")
    def test_measure_audio_ffmpeg_failure(self, mock_run):
        def subprocess_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "2.0"}}),
                )
            return MagicMock(returncode=1, stderr="error", stdout="")

        mock_run.side_effect = subprocess_side_effect

        metrics = measure_audio(b"fake", "mp3")

        assert metrics.integrated_lufs is None
        assert metrics.loudness_range_lu is None
        assert metrics.true_peak_dbtp is None
        assert metrics.threshold_lufs is None
        assert metrics.duration_seconds == 2.0

    @patch("subprocess.run")
    def test_measure_audio_bad_json(self, mock_run):
        def subprocess_side_effect(cmd, **kwargs):
            if "loudnorm" in " ".join(cmd):
                return MagicMock(returncode=0, stderr="no json here", stdout="")
            if "ffprobe" in cmd[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "1.0"}}),
                )
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = subprocess_side_effect

        metrics = measure_audio(b"fake", "mp3")

        assert metrics.integrated_lufs is None
        assert metrics.duration_seconds == 1.0


# ============================================================================
# measure_silence_ratio
# ============================================================================


class TestMeasureSilenceRatio:
    @patch("subprocess.run")
    def test_silence_ratio_calculation(self, mock_run):
        def subprocess_side_effect(cmd, **kwargs):
            if "silencedetect" in " ".join(cmd):
                return MagicMock(
                    returncode=0,
                    stderr=(
                        "[silencedetect] silence_start: 0.0\n"
                        "[silencedetect] silence_end: 0.5\n"
                        "[silencedetect] silence_start: 4.0\n"
                        "[silencedetect] silence_end: 5.0\n"
                    ),
                    stdout="",
                )
            if "ffprobe" in cmd[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "5.0"}}),
                )
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = subprocess_side_effect

        # 0.5s + 1.0s = 1.5s silence out of 5.0s
        ratio = measure_silence_ratio(b"fake", "mp3", total_duration=5.0)
        assert ratio is not None
        assert abs(ratio - 0.3) < 0.01

    @patch("subprocess.run")
    def test_silence_ratio_ffmpeg_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error", stdout="")

        ratio = measure_silence_ratio(b"fake", "mp3", total_duration=3.0)
        assert ratio is None

    @patch("subprocess.run")
    def test_silence_ratio_zero_duration(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "0"}}),
        )

        ratio = measure_silence_ratio(b"fake", "mp3", total_duration=0.0)
        assert ratio is None


# ============================================================================
# probe_audio_duration
# ============================================================================


class TestProbeAudioDuration:
    @patch("subprocess.run")
    def test_probe_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "2.5"}}),
        )
        assert abs(probe_audio_duration(Path("/tmp/test.mp3")) - 2.5) < 0.001

    @patch("subprocess.run")
    def test_probe_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert probe_audio_duration(Path("/tmp/test.mp3")) == 0.0

    @patch("subprocess.run")
    def test_probe_bad_json(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")
        assert probe_audio_duration(Path("/tmp/test.mp3")) == 0.0
