"""Schema-free audio loudness/measurement prober.

Pure ffmpeg/ffprobe + stdlib helpers for measuring audio quality and building
loudnorm filter strings. Deliberately free of any dependency on
``PostProcessingConfig`` or ``src.api.schemas`` so the SAME math can be ported
verbatim into sibling repos (VC-1 compiler, TTSB-1 tts_backend).

Defaults (``target_lufs=-16.0`` / ``true_peak=-1.5``) reproduce the historical
single-pass measurement command byte-for-byte. Because measurement runs with
``measured_only=1`` the ``I=``/``TP=`` targets are inert for pure measurement —
they exist so downstream normalizers can reuse the same filter-string builder.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "AudioMetrics",
    "loudnorm_measure_filter",
    "loudnorm_two_pass_filter",
    "measure_audio",
    "measure_silence_ratio",
    "probe_audio_duration",
]


@dataclass
class AudioMetrics:
    """Measured audio quality metrics."""

    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    true_peak_dbtp: float | None = None
    silence_ratio: float | None = None
    duration_seconds: float | None = None
    threshold_lufs: float | None = None


def _fmt(value: float) -> str:
    """Format a loudnorm target number without trailing zeros (``-16.0`` -> ``-16``)."""
    return f"{value:g}"


def loudnorm_measure_filter(*, target_lufs: float = -16.0, true_peak: float = -1.5) -> str:
    """Build the single-pass, measurement-only loudnorm filter string.

    With the defaults this yields exactly
    ``loudnorm=I=-16:TP=-1.5:print_format=json:measured_only=1`` — the historical
    literal used by ``AudioPostProcessor.measure_audio``.
    """
    return f"loudnorm=I={_fmt(target_lufs)}:TP={_fmt(true_peak)}:print_format=json:measured_only=1"


def loudnorm_two_pass_filter(
    measured: AudioMetrics, *, target_lufs: float = -16.0, true_peak: float = -1.5
) -> str:
    """Build a two-pass *linear* loudnorm filter from a prior measurement.

    Currently unwired in this repo — this is the reuse surface VC-1 / TTSB-1 port.
    Feeds the measured ``input_i``/``input_tp``/``input_lra``/``input_thresh`` back
    into a second normalization pass so ffmpeg can apply a linear (non-dynamic)
    gain to hit ``target_lufs`` / ``true_peak``.
    """
    return (
        f"loudnorm=I={_fmt(target_lufs)}:TP={_fmt(true_peak)}"
        f":measured_I={measured.integrated_lufs}"
        f":measured_TP={measured.true_peak_dbtp}"
        f":measured_LRA={measured.loudness_range_lu}"
        f":measured_thresh={measured.threshold_lufs}"
        ":linear=true"
    )


def measure_audio(
    audio_data: bytes,
    input_format: str,
    *,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
) -> AudioMetrics:
    """Measure audio quality metrics using the ffmpeg loudnorm filter.

    Returns AudioMetrics with LUFS, LRA, true peak, silence ratio, threshold and
    duration. ``target_lufs``/``true_peak`` parameterize the loudnorm target; with
    the defaults the emitted command is byte-identical to the historical form and,
    because measurement uses ``measured_only=1``, the result is independent of them.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input.{input_format}"
        input_path.write_bytes(audio_data)

        duration = probe_audio_duration(input_path)

        # Run loudnorm in measurement-only mode
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-af",
            loudnorm_measure_filter(target_lufs=target_lufs, true_peak=true_peak),
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # nosec B603  # noqa: S603 -- ffmpeg invocation with controlled arguments, no user input

        integrated_lufs = None
        loudness_range = None
        true_peak_measured = None
        threshold = None

        if result.returncode == 0:
            # Parse JSON from stderr (ffmpeg outputs filter data to stderr)
            stderr = result.stderr
            json_start = stderr.rfind("{")
            json_end = stderr.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    data = json.loads(stderr[json_start:json_end])
                    integrated_lufs = float(data.get("input_i", 0))
                    loudness_range = float(data.get("input_lra", 0))
                    true_peak_measured = float(data.get("input_tp", 0))
                    threshold = float(data.get("input_thresh", 0))
                except json.JSONDecodeError, ValueError, TypeError:
                    logger.debug("Failed to parse loudnorm JSON from ffmpeg stderr")

        silence_ratio = measure_silence_ratio(audio_data, input_format, duration)

        return AudioMetrics(
            integrated_lufs=integrated_lufs,
            loudness_range_lu=loudness_range,
            true_peak_dbtp=true_peak_measured,
            silence_ratio=silence_ratio,
            duration_seconds=duration,
            threshold_lufs=threshold,
        )


def measure_silence_ratio(
    audio_data: bytes, input_format: str, total_duration: float | None = None
) -> float | None:
    """Measure the ratio of silence in an audio file using the silencedetect filter.

    Returns silence ratio (0.0 - 1.0) or None if measurement fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input.{input_format}"
        input_path.write_bytes(audio_data)

        if total_duration is None:
            total_duration = probe_audio_duration(input_path)
        if not total_duration or total_duration <= 0:
            return None

        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-af",
            "silencedetect=noise=-40dB:d=0.3",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # nosec B603  # noqa: S603 -- ffmpeg invocation with controlled arguments, no user input
        if result.returncode != 0:
            return None

        # Parse silence_start/silence_end pairs from stderr
        stderr = result.stderr
        starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", stderr)]
        ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", stderr)]

        total_silence = 0.0
        for i, start in enumerate(starts):
            end = ends[i] if i < len(ends) else total_duration
            total_silence += end - start

        return min(1.0, total_silence / total_duration)


def probe_audio_duration(path: Path) -> float:
    """Use ffprobe to get the duration of an audio file in seconds.

    Named ``probe_audio_duration`` (not ``probe_duration``) to avoid clashing with
    the pre-existing public ``src.utils.audio.probe_duration``.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # nosec B603  # noqa: S603 -- ffprobe invocation with controlled arguments, no user input
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            return float(info.get("format", {}).get("duration", 0.0))
        except json.JSONDecodeError, ValueError:
            pass
    return 0.0
