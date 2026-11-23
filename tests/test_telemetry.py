"""Tests for deterministic simulated patrol telemetry."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.telemetry.patrol import telemetry_for_video_time

START = datetime(2026, 11, 23, 23, 55, 0)
WAYPOINTS = (
    ("gate", 0.0, 0.0),
    ("garage", 10.0, 5.0),
    ("driveway", 20.0, 10.0),
    ("perimeter", 30.0, 15.0),
)


def test_route_cycles_through_labels_by_video_time():
    samples = [
        telemetry_for_video_time(
            video_time_s,
            start_time=START,
            waypoints=WAYPOINTS,
            seconds_per_location=10.0,
            altitude_m=30.0,
        )
        for video_time_s in (0.0, 10.0, 20.0, 30.0, 40.0)
    ]

    assert [sample.location for sample in samples] == [
        "gate",
        "garage",
        "driveway",
        "perimeter",
        "gate",
    ]
    assert (samples[1].drone_x, samples[1].drone_y) == (10.0, 5.0)
    assert all(sample.altitude_m == 30.0 for sample in samples)


def test_same_inputs_produce_identical_telemetry():
    first = telemetry_for_video_time(
        37.5,
        start_time=START,
        waypoints=WAYPOINTS,
        seconds_per_location=10.0,
        altitude_m=30.0,
    )
    second = telemetry_for_video_time(
        37.5,
        start_time=START,
        waypoints=WAYPOINTS,
        seconds_per_location=10.0,
        altitude_m=30.0,
    )

    assert first == second
    assert first.timestamp == datetime(2026, 11, 23, 23, 55, 37, 500000)


def test_location_override_uses_matching_waypoint():
    sample = telemetry_for_video_time(
        0.0,
        start_time=START,
        waypoints=WAYPOINTS,
        seconds_per_location=10.0,
        altitude_m=30.0,
        location="garage",
    )

    assert sample.location == "garage"
    assert (sample.drone_x, sample.drone_y) == (10.0, 5.0)


def test_unknown_location_is_rejected():
    with pytest.raises(ValueError, match="Unknown patrol location"):
        telemetry_for_video_time(
            0.0,
            start_time=START,
            waypoints=WAYPOINTS,
            seconds_per_location=10.0,
            altitude_m=30.0,
            location="roof",
        )
