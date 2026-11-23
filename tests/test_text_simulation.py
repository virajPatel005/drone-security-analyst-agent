"""Tests for parsing scripted scenes without video or models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.telemetry.text_simulation import (
    load_scenes,
    parse_scene_line,
    records_from_text_script,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 11, 23, 23, 55, 0)
WAYPOINTS = (
    ("gate", 0.0, 0.0),
    ("garage", 10.0, 5.0),
    ("driveway", 20.0, 10.0),
    ("perimeter", 30.0, 15.0),
)
ALLOWED_LOCATIONS = tuple(label for label, _, _ in WAYPOINTS)


def test_parse_scene_line_extracts_frame_description_and_location():
    scene = parse_scene_line("Frame 1: blue truck at gate", ALLOWED_LOCATIONS)

    assert scene.frame_index == 1
    assert scene.description == "blue truck at gate"
    assert scene.location == "gate"


def test_parse_scene_line_rejects_bad_format():
    with pytest.raises(ValueError, match="Invalid scene line"):
        parse_scene_line("blue truck at gate", ALLOWED_LOCATIONS)


def test_load_scenes_skips_blank_lines_and_comments(tmp_path):
    script_path = tmp_path / "scenes.txt"
    script_path.write_text(
        "# comment\n\nFrame 2: white car at garage\n",
        encoding="utf-8",
    )

    scenes = load_scenes(script_path, ALLOWED_LOCATIONS)

    assert len(scenes) == 1
    assert scenes[0].frame_index == 2
    assert scenes[0].location == "garage"


def test_midnight_example_crosses_date_and_preserves_scene_location():
    script_path = PROJECT_ROOT / "data" / "text_scripts" / "midnight_crossing.txt"

    records = records_from_text_script(
        script_path,
        start_time=START,
        waypoints=WAYPOINTS,
        seconds_per_location=10.0,
        altitude_m=30.0,
        seconds_per_frame=60.0,
    )

    assert len(records) == 7
    assert records[0].timestamp == START
    assert records[5].timestamp == datetime(2026, 11, 24, 0, 0, 0)
    assert records[5].location == "gate"
    assert records[5].description == "person loitering at gate"
    assert records[5].image is None
