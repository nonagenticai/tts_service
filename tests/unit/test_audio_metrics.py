"""Tests for audio metrics measurement and quality profiles."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.api.schemas import (
    NormalizationConfig,
    PostProcessingConfig,
    QualityProfile,
)
from src.utils.post_processor import AudioMetrics, AudioPostProcessor

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

    def test_with_values(self):
        m = AudioMetrics(
            integrated_lufs=-16.0,
            loudness_range_lu=7.2,
            true_peak_dbtp=-1.5,
            silence_ratio=0.05,
            duration_seconds=3.5,
        )
        assert m.integrated_lufs == -16.0
        assert m.loudness_range_lu == 7.2
        assert m.true_peak_dbtp == -1.5
        assert m.silence_ratio == 0.05
        assert m.duration_seconds == 3.5


# ============================================================================
# measure_audio
# ============================================================================


class TestMeasureAudio:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_measure_audio_success(self, mock_run, mock_which):
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

        processor = AudioPostProcessor()
        metrics = processor.measure_audio(b"fake audio data", "mp3")

        assert metrics.integrated_lufs == -18.5
        assert metrics.loudness_range_lu == 6.3
        assert metrics.true_peak_dbtp == -2.1
        assert metrics.duration_seconds == 3.5
        assert metrics.silence_ratio is not None

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_measure_audio_ffmpeg_failure(self, mock_run, mock_which):
        def subprocess_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "2.0"}}),
                )
            return MagicMock(returncode=1, stderr="error", stdout="")

        mock_run.side_effect = subprocess_side_effect

        processor = AudioPostProcessor()
        metrics = processor.measure_audio(b"fake", "mp3")

        assert metrics.integrated_lufs is None
        assert metrics.loudness_range_lu is None
        assert metrics.true_peak_dbtp is None
        assert metrics.duration_seconds == 2.0

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_measure_audio_bad_json(self, mock_run, mock_which):
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

        processor = AudioPostProcessor()
        metrics = processor.measure_audio(b"fake", "mp3")

        assert metrics.integrated_lufs is None
        assert metrics.duration_seconds == 1.0


# ============================================================================
# measure_silence_ratio
# ============================================================================


class TestMeasureSilenceRatio:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_silence_ratio_calculation(self, mock_run, mock_which):
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

        processor = AudioPostProcessor()
        ratio = processor.measure_silence_ratio(b"fake", "mp3", total_duration=5.0)

        # 0.5s + 1.0s = 1.5s silence out of 5.0s
        assert ratio is not None
        assert abs(ratio - 0.3) < 0.01

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_silence_ratio_no_silence(self, mock_run, mock_which):
        def subprocess_side_effect(cmd, **kwargs):
            if "silencedetect" in " ".join(cmd):
                return MagicMock(returncode=0, stderr="", stdout="")
            if "ffprobe" in cmd[0]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "3.0"}}),
                )
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = subprocess_side_effect

        processor = AudioPostProcessor()
        ratio = processor.measure_silence_ratio(b"fake", "mp3", total_duration=3.0)

        assert ratio == 0.0

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_silence_ratio_ffmpeg_failure(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=1, stderr="error", stdout="")

        processor = AudioPostProcessor()
        ratio = processor.measure_silence_ratio(b"fake", "mp3", total_duration=3.0)

        assert ratio is None

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_silence_ratio_zero_duration(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "0"}}),
        )

        processor = AudioPostProcessor()
        ratio = processor.measure_silence_ratio(b"fake", "mp3", total_duration=0.0)

        assert ratio is None


# ============================================================================
# QualityProfile
# ============================================================================


class TestQualityProfile:
    def test_narration_profile(self):
        config = PostProcessingConfig.from_profile(QualityProfile.NARRATION)

        assert config.normalization.enabled is True
        assert config.normalization.target_lufs == -16.0
        assert config.normalization.true_peak == -1.5
        assert config.trim.enabled is True
        assert config.trim.threshold_db == -40.0
        assert config.fade.fade_in_ms == 30
        assert config.fade.fade_out_ms == 50
        assert config.sample_rate.target_rate == 48000

    def test_default_profile(self):
        config = PostProcessingConfig.from_profile(QualityProfile.DEFAULT)

        # Default profile should match a fresh PostProcessingConfig
        default = PostProcessingConfig()
        assert config.normalization.enabled == default.normalization.enabled
        assert config.trim.enabled == default.trim.enabled
        assert config.fade.fade_in_ms == default.fade.fade_in_ms
        assert config.fade.fade_out_ms == default.fade.fade_out_ms

    def test_narration_profile_has_processing(self):
        config = PostProcessingConfig.from_profile(QualityProfile.NARRATION)
        assert config.has_any_processing() is True

    def test_profile_explicit_config_wins(self):
        """Explicit config takes precedence over profile (tested at API layer)."""
        explicit = PostProcessingConfig(
            normalization=NormalizationConfig(enabled=False),
        )
        # Profile config
        profile_config = PostProcessingConfig.from_profile(QualityProfile.NARRATION)
        # They should be different
        assert explicit.normalization.enabled is False
        assert profile_config.normalization.enabled is True


# ============================================================================
# ElevenLabs kwargs passthrough
# ============================================================================


class TestElevenLabsKwargsPassthrough:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_default_stability_values(self, mock_which):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test-key")  # pragma: allowlist secret
        assert provider._stability == 0.5
        assert provider._similarity_boost == 0.75

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_custom_init_values(self, mock_which):
        from src.providers.elevenlabs_provider import ElevenLabsProvider

        provider = ElevenLabsProvider(
            api_key="test-key",  # pragma: allowlist secret
            stability=0.8,
            similarity_boost=0.9,
        )
        assert provider._stability == 0.8
        assert provider._similarity_boost == 0.9


# ============================================================================
# Provider params validation
# ============================================================================


class TestProviderParamsSchema:
    def test_provider_params_accepted(self):
        from src.api.schemas import TTSGenerateRequest

        req = TTSGenerateRequest(
            text="Hello world",
            provider="elevenlabs",
            provider_params={"stability": 0.7, "similarity_boost": 0.85},
        )
        assert req.provider_params == {"stability": 0.7, "similarity_boost": 0.85}

    def test_provider_params_none_by_default(self):
        from src.api.schemas import TTSGenerateRequest

        req = TTSGenerateRequest(text="Hello world")
        assert req.provider_params is None

    def test_quality_profile_accepted(self):
        from src.api.schemas import TTSGenerateRequest

        req = TTSGenerateRequest(
            text="Hello world",
            quality_profile=QualityProfile.NARRATION,
        )
        assert req.quality_profile == QualityProfile.NARRATION

    def test_quality_profile_none_by_default(self):
        from src.api.schemas import TTSGenerateRequest

        req = TTSGenerateRequest(text="Hello world")
        assert req.quality_profile is None


# ============================================================================
# AudioMetricsResponse
# ============================================================================


class TestAudioMetricsResponse:
    def test_response_model(self):
        from src.api.schemas import AudioMetricsResponse

        resp = AudioMetricsResponse(
            integrated_lufs=-16.0,
            loudness_range_lu=7.0,
            true_peak_dbtp=-1.5,
            silence_ratio=0.05,
            duration_seconds=3.0,
        )
        assert resp.integrated_lufs == -16.0
        assert resp.loudness_range_lu == 7.0

    def test_response_defaults(self):
        from src.api.schemas import AudioMetricsResponse

        resp = AudioMetricsResponse()
        assert resp.integrated_lufs is None
        assert resp.silence_ratio is None


class TestJobStatusResponseWithMetrics:
    def test_audio_metrics_field(self):
        from src.api.schemas import AudioMetricsResponse, JobStatusResponse

        resp = JobStatusResponse(
            job_id="test-123",
            status="completed",
            audio_metrics=AudioMetricsResponse(
                integrated_lufs=-16.0,
                duration_seconds=3.5,
            ),
        )
        assert resp.audio_metrics is not None
        assert resp.audio_metrics.integrated_lufs == -16.0

    def test_audio_metrics_none_by_default(self):
        from src.api.schemas import JobStatusResponse

        resp = JobStatusResponse(job_id="test", status="pending")
        assert resp.audio_metrics is None
