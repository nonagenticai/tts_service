"""Audio post-processing pipeline using ffmpeg filter chains."""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.api.schemas import PostProcessingConfig
from src.utils import audio_prober
from src.utils.audio_prober import AudioMetrics as AudioMetrics  # re-export

logger = logging.getLogger(__name__)


@dataclass
class PostProcessingResult:
    """Result of post-processing pipeline."""

    audio_data: bytes
    format: str
    sample_rate: int
    duration_seconds: float


class AudioPostProcessor:
    """Builds and executes ffmpeg filter chains for audio post-processing."""

    def __init__(self) -> None:
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("ffmpeg is not installed or not on PATH")

    def process(
        self,
        audio_data: bytes,
        input_format: str,
        input_sample_rate: int,
        config: PostProcessingConfig,
        output_format: str = "mp3",
    ) -> PostProcessingResult:
        """Apply the full post-processing pipeline via a single ffmpeg command.

        Args:
            audio_data: Raw audio bytes.
            input_format: Format of the input audio (e.g. "mp3", "wav").
            input_sample_rate: Sample rate of the input audio.
            config: Post-processing configuration.
            output_format: Desired output format.

        Returns:
            PostProcessingResult with processed audio bytes.

        Raises:
            RuntimeError: If ffmpeg processing fails.
        """
        if not config.has_any_processing():
            return PostProcessingResult(
                audio_data=audio_data,
                format=input_format,
                sample_rate=input_sample_rate,
                duration_seconds=0.0,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / f"input.{input_format}"
            output_path = tmpdir_path / f"output.{output_format}"
            input_path.write_bytes(audio_data)

            needs_silence = config.silence.leading_ms > 0 or config.silence.trailing_ms > 0
            target_rate = (
                config.sample_rate.target_rate
                if config.sample_rate.target_rate > 0
                else input_sample_rate
            )

            if needs_silence:
                cmd = self._build_complex_command(
                    input_path, output_path, config, output_format, target_rate
                )
            else:
                cmd = self._build_simple_command(
                    input_path, output_path, config, output_format, target_rate
                )

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # nosec B603  # noqa: S603 -- ffmpeg invocation with controlled arguments, no user input
            if result.returncode != 0:
                raise RuntimeError(
                    f"Post-processing failed (exit {result.returncode}): {result.stderr[:500]}"
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("Post-processing produced empty or missing output")

            processed_data = output_path.read_bytes()
            duration = self._probe_duration(output_path)

            return PostProcessingResult(
                audio_data=processed_data,
                format=output_format,
                sample_rate=target_rate,
                duration_seconds=duration,
            )

    def _build_audio_filters(self, config: PostProcessingConfig) -> list[str]:
        """Build ordered list of ffmpeg audio filter expressions."""
        filters: list[str] = []

        # 1. Trim silence
        if config.trim.enabled:
            threshold_db = config.trim.threshold_db
            filters.append(f"silenceremove=start_periods=1:start_threshold={threshold_db}dB")
            filters.append(
                f"areverse,silenceremove=start_periods=1:start_threshold={threshold_db}dB,areverse"
            )

        # 2. Loudness normalization (EBU R128)
        if config.normalization.enabled:
            filters.append(
                f"loudnorm=I={config.normalization.target_lufs}:TP={config.normalization.true_peak}"
            )

        # 3. Fade in/out
        if config.fade.fade_in_ms > 0:
            filters.append(f"afade=t=in:d={config.fade.fade_in_ms / 1000.0}")
        if config.fade.fade_out_ms > 0:
            filters.append(f"afade=t=out:d={config.fade.fade_out_ms / 1000.0}")

        # 4. Sample rate conversion
        if config.sample_rate.target_rate > 0:
            filters.append(f"aresample={config.sample_rate.target_rate}")

        return filters

    def _build_simple_command(
        self,
        input_path: Path,
        output_path: Path,
        config: PostProcessingConfig,
        output_format: str,
        target_rate: int,
    ) -> list[str]:
        """Build ffmpeg command with simple -af filter chain (no silence padding)."""
        cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_path)]

        filters = self._build_audio_filters(config)
        if filters:
            cmd.extend(["-af", ",".join(filters)])

        cmd.extend(self._output_encoding_args(output_format, config))
        cmd.append(str(output_path))
        return cmd

    def _build_complex_command(
        self,
        input_path: Path,
        output_path: Path,
        config: PostProcessingConfig,
        output_format: str,
        target_rate: int,
    ) -> list[str]:
        """Build ffmpeg command with filter_complex for silence padding."""
        cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_path)]

        # Add silence inputs
        input_idx = 1
        leading_idx = None
        trailing_idx = None

        if config.silence.leading_ms > 0:
            lead_dur = config.silence.leading_ms / 1000.0
            cmd.extend(
                ["-f", "lavfi", "-t", str(lead_dur), "-i", f"anullsrc=r={target_rate}:cl=mono"]
            )
            leading_idx = input_idx
            input_idx += 1

        if config.silence.trailing_ms > 0:
            trail_dur = config.silence.trailing_ms / 1000.0
            cmd.extend(
                ["-f", "lavfi", "-t", str(trail_dur), "-i", f"anullsrc=r={target_rate}:cl=mono"]
            )
            trailing_idx = input_idx
            input_idx += 1

        # Build filter_complex
        filters = self._build_audio_filters(config)
        filter_chain = ",".join(filters) if filters else "anull"
        parts: list[str] = [f"[0:a]{filter_chain}[processed]"]

        # Build concat inputs
        concat_inputs: list[str] = []
        concat_count = 0
        if leading_idx is not None:
            concat_inputs.append(f"[{leading_idx}:a]")
            concat_count += 1
        concat_inputs.append("[processed]")
        concat_count += 1
        if trailing_idx is not None:
            concat_inputs.append(f"[{trailing_idx}:a]")
            concat_count += 1

        parts.append(f"{''.join(concat_inputs)}concat=n={concat_count}:v=0:a=1[out]")

        cmd.extend(["-filter_complex", ";".join(parts)])
        cmd.extend(["-map", "[out]"])
        cmd.extend(self._output_encoding_args(output_format, config))
        cmd.append(str(output_path))
        return cmd

    @staticmethod
    def _output_encoding_args(output_format: str, config: PostProcessingConfig) -> list[str]:
        """Return encoding arguments for the output format."""
        args: list[str] = []
        if output_format == "mp3":
            args.extend(["-codec:a", "libmp3lame"])
            if config.bitrate.bitrate:
                args.extend(["-b:a", config.bitrate.bitrate])
            else:
                args.extend(["-qscale:a", "2"])
        elif output_format == "wav":
            args.extend(["-codec:a", "pcm_s16le"])
        return args

    def measure_audio(self, audio_data: bytes, input_format: str) -> AudioMetrics:
        """Measure audio quality metrics using ffmpeg loudnorm filter.

        Thin delegate to the schema-free :mod:`src.utils.audio_prober`. Passes no
        target so the historical ``I=-16:TP=-1.5`` default is used → byte-identical.

        Returns AudioMetrics with LUFS, LRA, true peak, silence ratio, and duration.
        """
        return audio_prober.measure_audio(audio_data, input_format)

    def measure_silence_ratio(
        self, audio_data: bytes, input_format: str, total_duration: float | None = None
    ) -> float | None:
        """Measure the ratio of silence in an audio file using silencedetect filter.

        Thin delegate to :func:`src.utils.audio_prober.measure_silence_ratio`.

        Returns silence ratio (0.0 - 1.0) or None if measurement fails.
        """
        return audio_prober.measure_silence_ratio(audio_data, input_format, total_duration)

    @staticmethod
    def _probe_duration(path: Path) -> float:
        """Use ffprobe to get the duration of an audio file.

        Thin delegate to :func:`src.utils.audio_prober.probe_audio_duration`.
        """
        return audio_prober.probe_audio_duration(path)
