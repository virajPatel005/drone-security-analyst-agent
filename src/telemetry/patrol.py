"""Simulated patrol telemetry derived from video time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.telemetry.clock import timestamp_for_video_time

Waypoint = tuple[str, float, float]


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    timestamp: datetime
    location: str
    altitude_m: float
    drone_x: float
    drone_y: float


def telemetry_for_video_time(
    video_time_s: float,
    *,
    start_time: datetime,
    waypoints: Sequence[Waypoint],
    seconds_per_location: float,
    altitude_m: float,
    location: str | None = None,
) -> TelemetrySample:
    """Build deterministic telemetry for one moment on the patrol route.

    The route cycles by elapsed video time so video mode needs no state, and
    the optional ``location`` lets text scenes name a place while still using
    the same clock, altitude and waypoint positions as video mode.
    """

    if seconds_per_location <= 0:
        raise ValueError("seconds_per_location must be positive")
    if not waypoints:
        raise ValueError("waypoints must not be empty")

    waypoint_lookup = {label: (x, y) for label, x, y in waypoints}
    if location is None:
        route_index = int(video_time_s // seconds_per_location) % len(waypoints)
        location, drone_x, drone_y = waypoints[route_index]
    else:
        if location not in waypoint_lookup:
            raise ValueError(f"Unknown patrol location: {location!r}")
        drone_x, drone_y = waypoint_lookup[location]

    return TelemetrySample(
        timestamp=timestamp_for_video_time(start_time, video_time_s),
        location=location,
        altitude_m=altitude_m,
        drone_x=drone_x,
        drone_y=drone_y,
    )
