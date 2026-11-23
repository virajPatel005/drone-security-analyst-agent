"""Tests for video probing and sampling using a tiny synthetic video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.perception.frame_sampler import VideoReadError, probe_video, sample_video


def _write_synthetic_video(path: Path, *, fps: float = 10.0, frame_count: int = 20) -> None:
    width, height = 64, 48
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened(), f"Could not create synthetic video at {path}"
    try:
        for frame_number in range(frame_count):
            frame = np.full((height, width, 3), frame_number % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_probe_reports_metadata(tmp_path):
    video_path = tmp_path / "tiny.avi"
    _write_synthetic_video(video_path)

    metadata = probe_video(video_path)

    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == pytest.approx(10.0)
    assert metadata.frame_count == 20
    assert metadata.duration_s == pytest.approx(2.0)


def test_sampler_yields_one_frame_per_video_second(tmp_path):
    video_path = tmp_path / "tiny.avi"
    _write_synthetic_video(video_path)

    frames = list(sample_video(video_path, sample_rate_fps=1.0))

    assert [frame.frame_index for frame in frames] == [1, 2]
    assert [frame.video_time_s for frame in frames] == [0.0, 1.0]
    assert frames[0].image.shape == (48, 64, 3)


def test_missing_video_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        list(sample_video(tmp_path / "missing.avi", sample_rate_fps=1.0))


def test_unreadable_video_raises_clear_error(tmp_path):
    video_path = tmp_path / "broken.mp4"
    video_path.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoReadError, match="Could not open video"):
        list(sample_video(video_path, sample_rate_fps=1.0))
