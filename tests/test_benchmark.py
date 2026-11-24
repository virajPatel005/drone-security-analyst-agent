"""Tests for the Step 10 measurement helpers (no GPU, VLM or OpenAI needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark  # noqa: E402
from src.perception.frame_sampler import probe_video, sample_video  # noqa: E402


def test_sample_indices_match_sampler_count(tmp_path: Path) -> None:
    """The sequential decoder must target the same frames the seek sampler reads."""
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    try:
        for i in range(30):
            writer.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    finally:
        writer.release()

    meta = probe_video(video_path)
    indices = benchmark._sample_indices(meta, 4.0)
    sampled = list(sample_video(video_path, 4.0))

    assert len(indices) == len(sampled) == 12
    assert indices[0] == 0
    assert max(indices) <= meta.frame_count - 1


def test_sample_indices_never_exceed_duration() -> None:
    """A 2 s clip at 1 fps must yield exactly frames 0 s and 1 s."""
    meta = SimpleNamespace(duration_s=2.0, fps=30.0, frame_count=60)
    assert benchmark._sample_indices(meta, 1.0) == [0, 30]
