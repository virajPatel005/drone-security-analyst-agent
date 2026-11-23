"""Deterministic mapping from video time to simulated wall-clock time."""

from __future__ import annotations

from datetime import datetime, timedelta


def timestamp_for_video_time(start_time: datetime, video_time_s: float) -> datetime:
    """Map elapsed video seconds onto the simulated clock.

    Using ``timedelta`` rather than the machine clock keeps tests deterministic
    and lets midnight rollover happen naturally when the offset crosses a day.
    """

    if video_time_s < 0:
        raise ValueError("video_time_s must be non-negative")
    return start_time + timedelta(seconds=video_time_s)
