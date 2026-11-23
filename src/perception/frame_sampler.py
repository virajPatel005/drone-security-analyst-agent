"""Sample frames from a video at a configurable rate."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened, probed or read."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class SampledFrame:
    frame_index: int
    video_time_s: float
    image: np.ndarray


def probe_video(video_path: Path) -> VideoMetadata:
    """Read metadata without decoding the whole file.

    Failing early with a clear error is important because the sampler should
    not silently produce an empty stream for a missing or corrupt video.
    """

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoReadError(f"Could not open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            raise VideoReadError(
                "Invalid video metadata for "
                f"{video_path}: width={width}, height={height}, fps={fps}, "
                f"frame_count={frame_count}"
            )

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=frame_count / fps,
        )
    finally:
        capture.release()


def sample_video(video_path: Path, sample_rate_fps: float) -> Iterator[SampledFrame]:
    """Yield frames at ``sample_rate_fps`` frames per second of video.

    Seeking by source frame avoids decoding every 4K frame while still giving
    a deterministic one-sample-per-tick timeline. The sample count is ceil of
    duration times rate so a two-second video at 1 fps yields frames at 0 s
    and 1 s.
    """

    if sample_rate_fps <= 0:
        raise ValueError("sample_rate_fps must be positive")

    metadata = probe_video(video_path)
    total_samples = max(1, math.ceil(metadata.duration_s * sample_rate_fps - 1e-9))

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoReadError(f"Could not open video: {video_path}")

        for sample_offset in range(total_samples):
            video_time_s = sample_offset / sample_rate_fps
            if video_time_s >= metadata.duration_s:
                break
            source_frame = min(
                int(round(video_time_s * metadata.fps)),
                metadata.frame_count - 1,
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = capture.read()
            if not ok or image is None:
                raise VideoReadError(
                    f"Could not read frame at {video_time_s:.3f}s from {video_path}"
                )
            yield SampledFrame(
                frame_index=sample_offset + 1,
                video_time_s=video_time_s,
                image=image,
            )
    finally:
        capture.release()
