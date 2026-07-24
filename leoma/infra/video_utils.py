"""
Video processing using ffmpeg: frame extraction, clipping, stitching.
"""
import os
import base64
import asyncio
import subprocess
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

MAX_STDERR_CHARS = 500
DEFAULT_SCENE_THRESHOLD = 0.18
SCENE_CUT_PATTERN = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


class FFmpegError(Exception):
    pass


@dataclass(frozen=True)
class OneShotClipSelection:
    """Selection metadata for a 5s clip constrained to a single detected shot."""

    clip_start_seconds: float
    segment_start_seconds: float
    segment_end_seconds: float
    video_duration_seconds: float
    scene_cuts: List[float]


def _decode_stderr(stderr: bytes | str | None) -> str:
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode(errors="ignore")[:MAX_STDERR_CHARS]
    return stderr[:MAX_STDERR_CHARS]


async def _run_process(command: Sequence[str], *, text: bool = False):
    return await asyncio.to_thread(
        subprocess.run,
        list(command),
        capture_output=True,
        text=text,
    )


def _raise_ffmpeg_error(result: subprocess.CompletedProcess, action: str) -> None:
    if result.returncode != 0:
        raise FFmpegError(f"Failed to {action}: {_decode_stderr(result.stderr)}")


def _remove_dir_files(path: str) -> None:
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))


def _parse_scene_cut_timestamps(output: str) -> List[float]:
    cuts: List[float] = []
    for match in SCENE_CUT_PATTERN.finditer(output or ""):
        try:
            cuts.append(float(match.group(1)))
        except (TypeError, ValueError):
            continue
    # Deduplicate near-identical timestamps that may appear in both stdout/stderr.
    return sorted({round(ts, 3) for ts in cuts})


async def extract_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 6,
    fps: float = 2.0,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    _remove_dir_files(output_dir)
    result = await _run_process(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps}", "-frames:v", str(max_frames), "-q:v", "2",
            f"{output_dir}/frame_%02d.jpg",
        ]
    )
    _raise_ffmpeg_error(result, "extract frames")
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])


def frames_to_base64(frame_paths: List[str]) -> List[Dict[str, Any]]:
    content = []
    for path in frame_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


async def get_video_duration(video_path: str) -> float:
    result = await _run_process(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def get_video_duration_sync(video_path: str) -> float:
    """Synchronous duration probe for offline batch pipelines."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True,
        text=True,
    )
    _raise_ffmpeg_error(result, "probe video duration")
    try:
        return float((result.stdout or "").strip())
    except (TypeError, ValueError) as exc:
        raise FFmpegError(f"Failed to parse video duration: {_decode_stderr(result.stderr)}") from exc


async def get_video_resolution(video_path: str) -> tuple[int, int]:
    """Probe the first video stream for (width, height). Returns (0, 0) on failure."""
    result = await _run_process(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", video_path,
        ],
        text=True,
    )
    try:
        width_str, height_str = (result.stdout or "").strip().split("x", 1)
        return int(width_str), int(height_str)
    except (ValueError, AttributeError):
        return 0, 0


async def extract_clip(video_path: str, output_path: str, start_offset: float, duration: float) -> None:
    result = await _run_process(
        [
            "ffmpeg", "-y", "-ss", str(start_offset), "-i", video_path,
            "-t", str(duration), "-c:v", "libx264", "-crf", "23", "-an", output_path,
        ]
    )
    _raise_ffmpeg_error(result, "extract clip")


async def extract_first_frame(video_path: str, output_path: str, start_offset: float = 0) -> None:
    result = await _run_process(
        [
            "ffmpeg", "-y", "-ss", str(start_offset), "-i", video_path,
            "-vframes", "1", "-q:v", "2", output_path,
        ]
    )
    _raise_ffmpeg_error(result, "extract first frame")


async def detect_scene_cuts(
    video_path: str,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
) -> List[float]:
    """
    Detect likely hard scene cuts using ffmpeg scene-change detection.
    Returns list of cut timestamps in seconds.
    """
    result = await _run_process(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select='gt(scene,{scene_threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        text=True,
    )
    _raise_ffmpeg_error(result, "detect scene cuts")
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    return _parse_scene_cut_timestamps(output)


def detect_scene_cuts_sync(
    video_path: str,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
) -> List[float]:
    """Synchronous scene detection for offline batch pipelines."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-i", video_path,
            "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    _raise_ffmpeg_error(result, "detect scene cuts")
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    return _parse_scene_cut_timestamps(output)


async def choose_one_shot_clip_start(
    video_path: str,
    clip_duration: float,
    *,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    boundary_margin: float = 0.15,
    seed: "str | int | None" = None,
) -> Optional[OneShotClipSelection]:
    """
    Choose a clip start offset such that the full clip stays inside a single shot.
    Returns None when no detected one-shot segment can hold clip_duration.

    When ``seed`` is given the segment and offset are drawn from a seeded RNG, so the same
    ``(video, seed)`` always yields the same clip — used to make sampling deterministic off the
    rotation's block hash. With ``seed=None`` selection is random as before.
    """
    rng = random.Random(seed) if seed is not None else random
    duration = await get_video_duration(video_path)
    if duration < clip_duration:
        return None

    cuts = await detect_scene_cuts(video_path, scene_threshold=scene_threshold)
    cuts = [cut for cut in cuts if 0.0 < cut < duration]
    boundaries = [0.0] + cuts + [duration]

    candidates: List[tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        segment_start = boundaries[i]
        segment_end = boundaries[i + 1]
        if segment_end <= segment_start:
            continue
        segment_span = segment_end - segment_start
        margin = (
            boundary_margin
            if segment_span >= (clip_duration + (2 * boundary_margin))
            else 0.0
        )
        usable_start = (
            segment_start + margin if segment_start > 0 else segment_start
        )
        usable_end = segment_end - margin if segment_end < duration else segment_end
        if usable_end - usable_start >= clip_duration:
            candidates.append((usable_start, usable_end))

    if not candidates:
        return None

    candidates.sort(key=lambda seg: seg[1] - seg[0], reverse=True)
    best_span = candidates[0][1] - candidates[0][0]
    near_best = [seg for seg in candidates if (seg[1] - seg[0]) >= (best_span - 1.0)]
    selected_start, selected_end = rng.choice(near_best)
    max_start = selected_end - clip_duration
    clip_start = (
        selected_start
        if max_start <= selected_start
        else rng.uniform(selected_start, max_start)
    )

    return OneShotClipSelection(
        clip_start_seconds=round(clip_start, 3),
        segment_start_seconds=round(selected_start, 3),
        segment_end_seconds=round(selected_end, 3),
        video_duration_seconds=round(duration, 3),
        scene_cuts=cuts,
    )


# Pinned decode settings for the duel's ground truth. These are part of the
# consensus surface: every byte of the truth depends on them, and truth_sha256 in
# the corpus manifest is computed under exactly this command line. Change any of
# them and every manifest must be rebuilt.
DECODE_PIX_FMT = "rgb24"
DECODE_SCALE_FLAGS = "bicubic"


def decode_frames_rgb(
    video_path: str,
    *,
    start_seconds: float,
    duration_seconds: float,
    fps: int,
    num_frames: int,
    width: int,
    height: int,
):
    """Decode a clip window straight to a ``(T, H, W, 3)`` uint8 array.

    One ffmpeg call, raw RGB on stdout. This replaces the old
    ``extract_clip`` → x264 re-encode → ``extract_frames`` → JPEG → PIL chain,
    which had three problems, all of them fatal for a ground truth:

    * **it was lossy twice** (x264, then JPEG) — the "real continuation" every
      distance is measured against was a compressed approximation of itself;
    * **frame order broke at 100 frames** — ``frame_%02d.jpg`` sorted
      lexicographically puts ``frame_100`` before ``frame_11``, silently
      shuffling the truth for any duel longer than 99 frames;
    * **it leaked** a temp directory per clip.

    Deterministic given the same ffmpeg build: the scaler and pixel format are
    pinned above. Across *different* builds it may differ — which is exactly why
    the caller verifies the result against the manifest's ``truth_sha256`` rather
    than trusting it.
    """
    import numpy as np

    command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{float(start_seconds):.3f}",
        "-i", video_path,
        "-t", f"{float(duration_seconds):.3f}",
        "-vf", f"fps={int(fps)},scale={int(width)}:{int(height)}:flags={DECODE_SCALE_FLAGS}",
        "-frames:v", str(int(num_frames)),
        "-f", "rawvideo", "-pix_fmt", DECODE_PIX_FMT,
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise FFmpegError(f"Failed to decode frames: {_decode_stderr(result.stderr)}")

    frame_bytes = int(width) * int(height) * 3
    raw = result.stdout
    got = len(raw) // frame_bytes if frame_bytes else 0
    if got < int(num_frames):
        raise FFmpegError(
            f"decoded {got} frames, need {num_frames} "
            f"(clip window {start_seconds:.3f}s +{duration_seconds:.3f}s may run past the video)"
        )
    usable = raw[: int(num_frames) * frame_bytes]
    return np.frombuffer(usable, dtype=np.uint8).reshape(int(num_frames), int(height), int(width), 3)


def motion_energy(frames) -> float:
    """Mean absolute inter-frame delta — how much the clip actually *moves*.

    Used at manifest-build time to drop near-static clips. The freeze cheat (emit
    the conditioning frame forever) only pays on clips that barely move, so the
    corpus simply does not contain any: remove the surface, don't just detect the
    attack.
    """
    import numpy as np

    arr = np.asarray(frames, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr, axis=0))))


async def stitch_videos_side_by_side(left_path: str, right_path: str, output_path: str) -> None:
    result = await _run_process(
        [
            "ffmpeg", "-y", "-i", left_path, "-i", right_path,
            "-filter_complex",
            "[0:v]scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2[left];"
            "[1:v]scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2[right];"
            "[left][right]hstack=inputs=2",
            "-c:v", "libx264", "-crf", "23", "-an", output_path,
        ]
    )
    _raise_ffmpeg_error(result, "stitch videos")
