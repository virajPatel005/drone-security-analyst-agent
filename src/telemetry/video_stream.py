"""Attach the simulated clock and telemetry to sampled video frames."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

from src.perception.frame_sampler import sample_video
from src.telemetry.patrol import Waypoint, telemetry_for_video_time
from src.telemetry.records import FrameRecord


def video_frame_records(
    video_path: Path,
    *,
    sample_rate_fps: float,
    start_time: datetime,
    waypoints: Sequence[Waypoint],
    seconds_per_location: float,
    altitude_m: float,
) -> Iterator[FrameRecord]:
    """Turn a video into the same frame records used by text simulation."""

    for sample in sample_video(video_path, sample_rate_fps):
        telemetry = telemetry_for_video_time(
            sample.video_time_s,
            start_time=start_time,
            waypoints=waypoints,
            seconds_per_location=seconds_per_location,
            altitude_m=altitude_m,
        )
        yield FrameRecord(
            frame_index=sample.frame_index,
            video_time_s=sample.video_time_s,
            timestamp=telemetry.timestamp,
            location=telemetry.location,
            altitude_m=telemetry.altitude_m,
            drone_x=telemetry.drone_x,
            drone_y=telemetry.drone_y,
            image=sample.image,
        )
