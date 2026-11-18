"""Audio conversion utilities using ffmpeg."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def wav_to_mp3(wav_path: Path, mp3_path: Path | None = None) -> Path:
    """Convert a WAV file to MP3 using ffmpeg.

    Args:
        wav_path: Path to the input WAV file.
        mp3_path: Path for the output MP3 file. Defaults to same name with .mp3 extension.

    Returns:
        Path to the generated MP3 file.

    Raises:
        FileNotFoundError: If ffmpeg is not on PATH.
        RuntimeError: If conversion fails or produces an empty file.
    """
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg is not installed or not on PATH. "
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    if mp3_path is None:
        mp3_path = wav_path.with_suffix(".mp3")

    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(mp3_path),
    ]

    result = subprocess.run(  # nosec B603  # noqa: S603 -- ffmpeg invocation with controlled arguments, no user input
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg WAV→MP3 conversion failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg WAV→MP3 conversion produced empty or missing file: {mp3_path}")

    logger.info("Converted WAV→MP3: %s (%d bytes)", mp3_path, mp3_path.stat().st_size)
    return mp3_path


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def probe_duration(path: Path) -> float | None:
    """Use ffprobe to get the duration of an audio file in seconds.

    Returns ``None`` when the duration could NOT be measured — ffprobe absent or
    non-zero, output unparseable, or no ``duration`` in the format block. That is
    deliberately NOT ``0.0``: a real zero-length file measures ``0.0``, and a
    caller that cannot tell the two apart reports a fabricated length on a 200
    response carrying real audio, and (via the metadata cache) keeps reporting it
    for the whole TTL long after ffprobe came back. ``None`` is the same
    "unmeasured" shape ``AudioMetrics.duration_seconds`` already uses.
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
            raw = info.get("format", {}).get("duration")
            if raw is not None:
                return float(raw)
        except json.JSONDecodeError, ValueError:
            pass
    return None


def _encoding_args(output_format: str) -> list[str]:
    """ffmpeg encoder arguments for *output_format* (used when re-encoding)."""
    if output_format == "mp3":
        return ["-codec:a", "libmp3lame", "-qscale:a", "2"]
    if output_format == "wav":
        return ["-codec:a", "pcm_s16le"]
    return []


def concatenate_audio(
    audio_segments: list[bytes],
    output_format: str = "mp3",
    gap_ms: int = 0,
    segment_formats: list[str] | None = None,
) -> bytes:
    """Concatenate multiple audio segments into a single file using ffmpeg.

    Args:
        audio_segments: Ordered list of audio byte buffers.
        output_format: Output format ("mp3" or "wav").
        gap_ms: Milliseconds of silence to insert between segments.
        segment_formats: The REAL format of each buffer, positionally aligned
            with *audio_segments*. Defaults to ``output_format`` for every
            segment. This matters because each buffer is written to a temp file
            whose extension tells ffmpeg how to demux it: naming an MP3 buffer
            ``seg_0000.wav`` and then concatenating with ``-c copy`` produced a
            corrupt file. When any segment's real format differs from
            *output_format* the segments are re-encoded instead of stream-copied.

    Returns:
        Concatenated audio as bytes.

    Raises:
        FileNotFoundError: If ffmpeg is not on PATH.
        RuntimeError: If concatenation fails.
        ValueError: If no segments provided, or segment_formats is misaligned.
    """
    if not audio_segments:
        raise ValueError("No audio segments to concatenate")
    if segment_formats is not None and len(segment_formats) != len(audio_segments):
        raise ValueError(
            f"segment_formats has {len(segment_formats)} entries for {len(audio_segments)} segments"
        )
    if len(audio_segments) == 1:
        return audio_segments[0]
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg is not installed or not on PATH")

    formats = list(segment_formats) if segment_formats else [output_format] * len(audio_segments)
    # Stream-copy is only safe when every input is already the output codec.
    can_stream_copy = all(fmt == output_format for fmt in formats)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write each segment to a temp file named for ITS OWN format so ffmpeg
        # demuxes it correctly.
        segment_paths: list[Path] = []
        for i, (segment_data, seg_format) in enumerate(zip(audio_segments, formats, strict=True)):
            seg_path = tmpdir_path / f"seg_{i:04d}.{seg_format}"
            seg_path.write_bytes(segment_data)
            segment_paths.append(seg_path)

        if gap_ms > 0:
            # Use ffmpeg filter_complex with silence gaps (always re-encodes)
            return _concat_with_gaps(segment_paths, tmpdir_path, output_format, gap_ms)
        # Use ffmpeg concat demuxer (stream-copy only when formats are uniform)
        return _concat_demuxer(
            segment_paths, tmpdir_path, output_format, stream_copy=can_stream_copy
        )


def _concat_demuxer(
    segment_paths: list[Path],
    tmpdir: Path,
    output_format: str,
    *,
    stream_copy: bool = True,
) -> bytes:
    """Concatenate using ffmpeg concat demuxer.

    ``stream_copy=False`` re-encodes, which is required whenever the inputs are
    not already in ``output_format``.
    """
    list_file = tmpdir / "filelist.txt"
    lines = [f"file '{p}'\n" for p in segment_paths]
    list_file.write_text("".join(lines))

    output_path = tmpdir / f"output.{output_format}"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        *(["-c", "copy"] if stream_copy else _encoding_args(output_format)),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # nosec B603  # noqa: S603 -- ffmpeg invocation with controlled arguments, no user input
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio concatenation failed (exit {result.returncode}): {result.stderr[:500]}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Audio concatenation produced empty output")

    return output_path.read_bytes()


def _concat_with_gaps(
    segment_paths: list[Path], tmpdir: Path, output_format: str, gap_ms: int
) -> bytes:
    """Concatenate with silence gaps using ffmpeg filter_complex."""
    output_path = tmpdir / f"output.{output_format}"
    gap_sec = gap_ms / 1000.0

    # Build ffmpeg command with inputs and filter_complex
    cmd: list[str] = ["ffmpeg", "-y"]

    # Add each segment as an input
    for path in segment_paths:
        cmd.extend(["-i", str(path)])

    # Build filter_complex: insert silence between segments
    n = len(segment_paths)
    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for i in range(n):
        label = f"[{i}:a]"
        # Pad trailing silence on all but last segment
        if i < n - 1:
            padded = f"a{i}"
            filter_parts.append(f"{label}apad=pad_dur={gap_sec}[{padded}]")
            concat_inputs.append(f"[{padded}]")
        else:
            concat_inputs.append(label)

    concat_str = "".join(concat_inputs)
    filter_parts.append(f"{concat_str}concat=n={n}:v=0:a=1[out]")

    cmd.extend(["-filter_complex", ";".join(filter_parts)])
    cmd.extend(["-map", "[out]"])

    if output_format == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-qscale:a", "2"])
    elif output_format == "wav":
        cmd.extend(["-codec:a", "pcm_s16le"])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # nosec B603  # noqa: S603 -- ffmpeg invocation with controlled arguments, no user input
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio concatenation with gaps failed (exit {result.returncode}): "
            f"{result.stderr[:500]}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Audio concatenation with gaps produced empty output")

    return output_path.read_bytes()
