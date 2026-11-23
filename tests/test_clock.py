"""Tests for the deterministic simulated clock."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.telemetry.clock import timestamp_for_video_time

START = datetime(2026, 11, 23, 23, 55, 0)


def test_zero_maps_to_start_time():
    assert timestamp_for_video_time(START, 0.0) == START


def test_midnight_rollover_crosses_date():
    assert timestamp_for_video_time(START, 300.0) == datetime(2026, 11, 24, 0, 0, 0)
    assert timestamp_for_video_time(START, 360.0) == datetime(2026, 11, 24, 0, 1, 0)


def test_negative_video_time_is_rejected():
    with pytest.raises(ValueError):
        timestamp_for_video_time(START, -1.0)
