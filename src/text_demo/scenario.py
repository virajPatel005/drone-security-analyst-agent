"""Parse JSON Lines scenarios for the text-simulation demo.

Each non-blank, non-comment line is one frame. Accepting both a full ISO
date-time in ``time`` and a separate ``date`` + ``time`` keeps hand-written
scenarios readable while still producing the full ISO timestamps that the
existing midnight-crossing queries rely on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Deliberately narrow: a typo should be a hard failure rather than a track
# that no alert rule recognises.
ALLOWED_CLASSES = frozenset({"person", "car", "truck", "bus"})


@dataclass(frozen=True, slots=True)
class ScenarioObject:
    """One object labelled in a scenario frame."""

    class_name: str
    label: str                 # stable across frames: one label means one track
    color: str | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioFrame:
    """One text frame: time, location, stored text and its labelled objects."""

    timestamp: datetime
    zone: str
    description: str
    objects: tuple[ScenarioObject, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A validated, ordered scenario. ``name`` comes from the file stem."""

    name: str
    frames: tuple[ScenarioFrame, ...]


def _required_string(payload: dict[str, Any], key: str, line_number: int, *, where: str = "entry") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {where} needs a non-empty string '{key}'")
    return value.strip()


def _parse_timestamp(payload: dict[str, Any], line_number: int) -> datetime:
    """Read `time` either as a full ISO date-time or with a separate `date`."""
    raw_time = _required_string(payload, "time", line_number)
    if "T" in raw_time:
        try:
            return datetime.fromisoformat(raw_time)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: bad 'time' format {raw_time!r}") from exc
    raw_date = _required_string(payload, "date", line_number)
    try:
        return datetime.fromisoformat(f"{raw_date}T{raw_time}")
    except ValueError as exc:
        raise ValueError(
            f"line {line_number}: bad 'date'/'time' format {raw_date!r} {raw_time!r}"
        ) from exc


def parse_scenario_line(raw: str, line_number: int) -> ScenarioFrame:
    """Parse and validate one JSONL line, raising ``ValueError`` with its line number."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")

    timestamp = _parse_timestamp(payload, line_number)
    zone = _required_string(payload, "zone", line_number)
    description = _required_string(payload, "description", line_number)

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError(f"line {line_number}: 'objects' must be a list")
    objects: list[ScenarioObject] = []
    for index, entry in enumerate(raw_objects, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"line {line_number}: object {index} must be a JSON object")
        where = f"object {index}"
        class_name = _required_string(entry, "class", line_number, where=where)
        if class_name not in ALLOWED_CLASSES:
            raise ValueError(f"line {line_number}: unknown class {class_name!r} in {where}")
        label = _required_string(entry, "label", line_number, where=where)
        color = entry.get("color")
        text = entry.get("text")
        if color is not None and not isinstance(color, str):
            raise ValueError(f"line {line_number}: {where} 'color' must be a string")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"line {line_number}: {where} 'text' must be a string")
        objects.append(
            ScenarioObject(
                class_name=class_name,
                label=label,
                color=color.strip() if color else None,
                text=text.strip() if text else None,
            )
        )

    return ScenarioFrame(
        timestamp=timestamp,
        zone=zone,
        description=description,
        objects=tuple(objects),
    )


def load_scenario(path: Path) -> Scenario:
    """Load a scenario file, rejecting empty files and non-increasing times."""
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    frames: list[ScenarioFrame] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        frames.append(parse_scenario_line(line, line_number))

    if not frames:
        raise ValueError(f"{path}: scenario contains no frames")
    for previous, current in zip(frames, frames[1:]):
        if current.timestamp <= previous.timestamp:
            raise ValueError(
                f"{path}: frame timestamps must strictly increase "
                f"({previous.timestamp.isoformat()} -> {current.timestamp.isoformat()})"
            )
    return Scenario(name=path.stem, frames=tuple(frames))
