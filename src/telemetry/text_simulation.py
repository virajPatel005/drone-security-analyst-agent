"""Scripted, model-free scenes that reuse the clock and telemetry path."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.telemetry.patrol import Waypoint, telemetry_for_video_time
from src.telemetry.records import FrameRecord

_LINE_PATTERN = re.compile(r"^Frame\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ScriptedScene:
    frame_index: int
    description: str
    location: str | None


def _location_from_text(description: str, allowed_locations: Sequence[str]) -> str | None:
    for location in allowed_locations:
        if re.search(rf"\b{re.escape(location)}\b", description, re.IGNORECASE):
            return location
    return None


def parse_scene_line(line: str, allowed_locations: Sequence[str]) -> ScriptedScene:
    """Parse one ``Frame N: description`` line.

    The location is read from the description when it names a known patrol
    area, so hand-written scenes can stay readable while still producing the
    same telemetry fields as video mode.
    """

    match = _LINE_PATTERN.match(line.strip())
    if match is None:
        raise ValueError(f"Invalid scene line: {line!r}")

    frame_index = int(match.group(1))
    if frame_index < 1:
        raise ValueError("Scene frame index must be 1 or greater")
    description = match.group(2).strip()
    return ScriptedScene(
        frame_index=frame_index,
        description=description,
        location=_location_from_text(description, allowed_locations),
    )


def load_scenes(path: Path, allowed_locations: Sequence[str]) -> list[ScriptedScene]:
    """Load scripted scenes from a text file, skipping blanks and comments."""

    if not path.is_file():
        raise FileNotFoundError(f"Scene script not found: {path}")

    scenes: list[ScriptedScene] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            scenes.append(parse_scene_line(line, allowed_locations))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return scenes


def records_from_scenes(
    scenes: Sequence[ScriptedScene],
    *,
    start_time: datetime,
    waypoints: Sequence[Waypoint],
    seconds_per_location: float,
    altitude_m: float,
    seconds_per_frame: float,
) -> list[FrameRecord]:
    """Convert scripted scenes into the shared frame record.

    Text mode advances one scene per configured interval, which lets a short
    script demonstrate a midnight rollover while still using the same clock,
    altitude and patrol waypoint positions as video mode.
    """

    if seconds_per_frame <= 0:
        raise ValueError("seconds_per_frame must be positive")

    records: list[FrameRecord] = []
    for scene in scenes:
        video_time_s = (scene.frame_index - 1) * seconds_per_frame
        telemetry = telemetry_for_video_time(
            video_time_s,
            start_time=start_time,
            waypoints=waypoints,
            seconds_per_location=seconds_per_location,
            altitude_m=altitude_m,
            location=scene.location,
        )
        records.append(
            FrameRecord(
                frame_index=scene.frame_index,
                video_time_s=video_time_s,
                timestamp=telemetry.timestamp,
                location=telemetry.location,
                altitude_m=telemetry.altitude_m,
                drone_x=telemetry.drone_x,
                drone_y=telemetry.drone_y,
                image=None,
                description=scene.description,
            )
        )
    return records


def records_from_text_script(
    path: Path,
    *,
    start_time: datetime,
    waypoints: Sequence[Waypoint],
    seconds_per_location: float,
    altitude_m: float,
    seconds_per_frame: float,
) -> list[FrameRecord]:
    """Load a script and convert it to frame records in one call."""

    allowed_locations = tuple(label for label, _, _ in waypoints)
    scenes = load_scenes(path, allowed_locations)
    return records_from_scenes(
        scenes,
        start_time=start_time,
        waypoints=waypoints,
        seconds_per_location=seconds_per_location,
        altitude_m=altitude_m,
        seconds_per_frame=seconds_per_frame,
    )
