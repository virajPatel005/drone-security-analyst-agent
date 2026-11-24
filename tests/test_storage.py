"""Tests for the SQLite storage module — in-memory, no GPU/model dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.config.settings import DB_PATH, CLASS_CATEGORIES
from src.storage import (
    _EventRow,
    _TrackRow,
    init_db,
    query_counts_by_event_type,
    query_events_for_track,
    query_events_in_time_window,
    query_tracks_by_class,
    store_run,
    store_tracks_and_events,
)


# ---------------------------------------------------------------------------
# Helper builders (mirror the event builder's dataclass structure)
# ---------------------------------------------------------------------------

_EventType = str  # "appeared" | "entered_zone" | "left"


def _make_event(
    event_type: _EventType,
    track_id: int,
    category: str,
    class_name: str,
    timestamp: datetime,
    zone: str = "gate",
    dwell_s: float | None = None,
    end_of_video: bool = False,
) -> _EventRow:
    """Convenient factory for _EventRow test objects."""
    return _EventRow(
        id=0,
        run_id=1,
        track_id=track_id,
        event_type=event_type,
        timestamp=timestamp.isoformat(timespec="seconds"),
        zone=zone if zone else "",
        dwell_s=dwell_s if dwell_s is not None else 0.0,
        class_name=class_name,
        end_of_video=1 if end_of_video else 0,
    )


def _make_track(
    track_id: int,
    class_name: str,
    first_seen: datetime,
    last_seen: datetime,
    frames_seen: int,
    max_confidence: float,
) -> _TrackRow:
    return _TrackRow(
        track_id=track_id,
        class_name=class_name,
        first_seen=first_seen.isoformat(timespec="seconds"),
        last_seen=last_seen.isoformat(timespec="seconds"),
        frames_seen=frames_seen,
        max_confidence=max_confidence,
    )


# ---------------------------------------------------------------------------
# Fixture: in-memory database
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _storage_fixture():
    """Fresh in-memory DB for every test."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Tests: store_run / store_tracks_and_events
# ---------------------------------------------------------------------------

def test_store_and_query_events_across_midnight():
    """Store events that span midnight and retrieve them with a window query.

    This is the key scenario: the simulated clock starts at 23:59:45 and
    events continue into the next day (00:00:05).  The range query
    ``23:59:55 → 00:00:09`` must return those events without needing
    special midnight-handling code.
    """
    from src.storage import init_db

    conn = init_db(":memory:")

    # Simulate a run that starts at 23:59:45 and produces events across midnight
    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    events = [
        _make_event(
            "appeared",
            track_id=1,
            category="vehicle",
            class_name="car",
            timestamp=SIM_START + timedelta(seconds=10),  # 00:00:05 (next day... wait)
            # Actually let me reconsider. The timestamps are ISO strings.
            # Let me set them up so some are before midnight and some after.
        ),
        # Let me rethink this. I'll make events with timestamps that straddle midnight.
    ]

    # Actually, let me use absolute datimes that cross midnight.
    # The SIM_START is 2026-11-23 23:59:45. 
    # 10 seconds later = 2026-11-23 23:60:05 which is 2026-11-24 00:00:05
    # Let me be more careful.

    # Events spanning midnight:
    # - t = 0s -> 2026-11-23 23:59:45 (before midnight)
    # - t = 5s -> 2026-11-23 23:59:50 (before midnight)
    # - t = 10s -> 2026-11-23 23:59:55 (before midnight) 
    # - t = 15s -> 2026-11-24 00:00:00 (midnight)
    # - t = 20s -> 2026-11-24 00:00:05 (after midnight)

    # For the "events between 23:59:55 and 00:00:09" query:
    # We need events at 23:59:55, 00:00:00, 00:00:05, 00:00:09

    events = [
        _make_event(
            "appeared",
            track_id=1,
            category="vehicle",
            class_name="car",
            timestamp=SIM_START + timedelta(seconds=0),  # 23:59:45
        ),
        _make_event(
            "entered_zone",
            track_id=1,
            category="vehicle",
            class_name="car",
            timestamp=SIM_START + timedelta(seconds=5),  # 23:59:50
        ),
        _make_event(
            "entered_zone",
            track_id=1,
            category="vehicle",
            class_name="car",
            timestamp=SIM_START + timedelta(seconds=10),  # 23:59:55
        ),
        _make_event(
            "left",
            track_id=1,
            category="vehicle",
            class_name="car",
            timestamp=SIM_START + timedelta(seconds=20),  # 2026-11-24 00:00:05
        ),
    ]

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }

    run_id = store_run(
        conn,
        source_video="demo.mp4",
        start_time=SIM_START,
        tuning_settings=tuning,
    )

    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)

    # Query: events between 23:59:55 and 00:00:09
    # This window crosses midnight: from Nov 23 23:59:55 to Nov 24 00:00:09
    query_start = datetime(2026, 11, 23, 23, 59, 55)
    query_end = datetime(2026, 11, 24, 0, 0, 9)

    result = query_events_in_time_window(
        conn, start=query_start, end=query_end, run_id=run_id
    )

    # We should get the events at 23:59:55 and 00:00:05
    # The appeared at 23:59:45 should NOT be included (before start)
    # The left at 00:00:05 should be included
    assert len(result) >= 2, f"Expected at least 2 events in midnight-crossing window, got {len(result)}: {[r.timestamp for r in result]}"

    # The 23:59:55 entered_zone should be included
    entered_55 = [r for r in result if r.timestamp == "2026-11-23T23:59:55"]
    assert len(entered_55) >= 1, "Expected entered_zone at 23:59:55 in window"

    # The 00:00:05 left should be included
    left_0005 = [r for r in result if r.timestamp == "2026-11-24T00:00:05"]
    assert len(left_0005) >= 1, "Expected left at 00:00:05 in window"

    conn.close()


def test_query_events_for_track():
    """All events for a given track should be retrievable."""

    conn = init_db(":memory:")

    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=0)),
        _make_event("entered_zone", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=5)),
        _make_event("left", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=15)),
    ]

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }

    run_id = store_run(conn, source_video="demo.mp4", start_time=SIM_START, tuning_settings=tuning)
    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)

    track_events = query_events_for_track(conn, run_id=run_id, track_id=1)

    assert len(track_events) == 3, f"Expected 3 events for track 1, got {len(track_events)}"
    event_types = [e.event_type for e in track_events]
    assert event_types == ["appeared", "entered_zone", "left"]

    conn.close()


def test_query_tracks_by_class():
    """Tracks can be filtered by class name."""

    conn = init_db(":memory:")

    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=0)),
        _make_event("appeared", track_id=2, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=2)),
        _make_event("appeared", track_id=3, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=5)),
    ]

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }

    run_id = store_run(conn, source_video="demo.mp4", start_time=SIM_START, tuning_settings=tuning)
    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)

    car_tracks = query_tracks_by_class(conn, class_name="car", run_id=run_id)
    assert len(car_tracks) == 2, f"Expected 2 car tracks, got {len(car_tracks)}"

    person_tracks = query_tracks_by_class(conn, class_name="person", run_id=run_id)
    assert len(person_tracks) == 1, f"Expected 1 person track, got {len(person_tracks)}"

    conn.close()


def test_query_counts_by_event_type():
    """Counts by event type should be correct."""

    conn = init_db(":memory:")

    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=0)),
        _make_event("entered_zone", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=5)),
        _make_event("left", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=10)),
        _make_event("appeared", track_id=2, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=2)),
        _make_event("entered_zone", track_id=2, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=7)),
    ]

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }

    run_id = store_run(conn, source_video="demo.mp4", start_time=SIM_START, tuning_settings=tuning)
    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)

    counts = query_counts_by_event_type(conn, run_id=run_id)
    assert counts.get("appeared", 0) == 2
    assert counts.get("entered_zone", 0) == 2
    assert counts.get("left", 0) == 1

    # Without run filter
    all_counts = query_counts_by_event_type(conn)
    assert all_counts.get("appeared", 0) >= 2
    assert all_counts.get("entered_zone", 0) >= 2
    assert all_counts.get("left", 0) >= 1

    conn.close()


def test_query_events_in_time_window_with_type_and_class_filter():
    """Window query with both event_type and class_name filters."""

    conn = init_db(":memory:")

    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=0)),
        _make_event("entered_zone", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=5)),
        _make_event("left", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=10)),
        _make_event("appeared", track_id=2, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=2)),
    ]

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }

    run_id = store_run(conn, source_video="demo.mp4", start_time=SIM_START, tuning_settings=tuning)
    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)

    # Filter by type=appeared and class=person within a window
    result = query_events_in_time_window(
        conn,
        start=SIM_START,
        end=SIM_START + timedelta(seconds=3),
        event_type="appeared",
        class_name="person",
        run_id=run_id,
    )

    assert len(result) == 1, f"Expected 1 appeared+person event, got {len(result)}"
    assert result[0].event_type == "appeared"
    assert result[0].class_name == "person"

    conn.close()


# ---------------------------------------------------------------------------
# Test: per-track query with ID switching note
# ---------------------------------------------------------------------------

def test_per_track_query_with_id_switching_note():
    """Verify that per-track queries return only events for the requested track_id.

    Note: ByteTrack ID switching can split one physical object into several
    tracker IDs.  A per-track query will only return rows for the specified
    track_id, so one object that gets re-assigned a new ID will appear as
    separate result sets.
    """
    conn = init_db(":memory:")

    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    # Two tracks: one that stays stable, one that has an ID switch scenario
    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=0)),
        _make_event("entered_zone", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=5)),
        _make_event("left", track_id=1, category="vehicle", class_name="car",
                     timestamp=SIM_START + timedelta(seconds=15)),
        # track 2 appears, then later a new ID 3 takes over (simulating ID switch)
        _make_event("appeared", track_id=2, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=2)),
        _make_event("entered_zone", track_id=2, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=7)),
        # Simulate ID switch: same physical object now tracked as ID 3
        _make_event("appeared", track_id=3, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=12)),
        _make_event("entered_zone", track_id=3, category="person", class_name="person",
                     timestamp=SIM_START + timedelta(seconds=15)),
    ]

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }

    run_id = store_run(conn, source_video="demo.mp4", start_time=SIM_START, tuning_settings=tuning)
    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)

    # Query for track 2 only
    track2_events = query_events_for_track(conn, run_id=run_id, track_id=2)
    assert len(track2_events) == 2, f"Expected 2 events for track 2, got {len(track2_events)}"
    assert [e.event_type for e in track2_events] == ["appeared", "entered_zone"]

    # Query for track 3 only (simulated ID switch)
    track3_events = query_events_for_track(conn, run_id=run_id, track_id=3)
    assert len(track3_events) == 2, f"Expected 2 events for track 3, got {len(track3_events)}"
    assert [e.event_type for e in track3_events] == ["appeared", "entered_zone"]

    # Track 1 should have its own 3 events
    track1_events = query_events_for_track(conn, run_id=run_id, track_id=1)
    assert len(track1_events) == 3

    conn.close()


def test_store_and_reopen_file_db():
    """Verify that store_tracks_and_events commits internally so data persists
    across connection close and reopen with a file-based database.

    This tests the fix for the bug where missing conn.commit() after
    store_tracks_and_events caused data loss on conn.close().
    """
    import os
    from datetime import datetime

    db_path = ":memory:"  # start in-memory, then switch to file
    # Actually use a file-based DB to test persistence
    db_file = "_test_persistence.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    conn = init_db(db_file)

    tuning = {
        "yolo_model": "yolo11s.pt",
        "yolo_conf": 0.10,
        "yolo_imgsz": 1280,
        "sample_rate_fps": 10.0,
        "min_track_frames": 2,
        "lost_frames": 10,
        "tracker_config": "configs/bytetrack_0.9.yaml",
    }
    run_id = store_run(
        conn, source_video="demo.mp4", start_time=datetime(2026, 1, 1), tuning_settings=tuning
    )

    from src.storage import _EventRow
    events = [
        _EventRow(id=0, run_id=run_id, track_id=1, event_type='appeared', timestamp='2026-01-01T00:00:00', zone='gate', dwell_s=0.0, class_name='car', end_of_video=0, confidence=0.8),
        _EventRow(id=1, run_id=run_id, track_id=1, event_type='entered_zone', timestamp='2026-01-01T00:00:00', zone='gate', dwell_s=0.0, class_name='car', end_of_video=0, confidence=0.8),
        _EventRow(id=2, run_id=run_id, track_id=1, event_type='left', timestamp='2026-01-01T00:00:05', zone='gate', dwell_s=5.0, class_name='car', end_of_video=0, confidence=0.8),
    ]

    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)
    conn.close()

    # Reopen and verify data persisted
    conn2 = init_db(db_file)
    cursor = conn2.execute('SELECT COUNT(*) FROM events')
    count = cursor.fetchone()[0]
    assert count == 3, f"Expected 3 events after reopen, got {count}"

    cursor = conn2.execute('SELECT COUNT(*) FROM tracks')
    count = cursor.fetchone()[0]
    assert count == 1, f"Expected 1 track after reopen, got {count}"

    cursor = conn2.execute('SELECT COUNT(*) FROM runs')
    count = cursor.fetchone()[0]
    assert count == 1, f"Expected 1 run after reopen, got {count}"

    conn2.close()
    os.remove(db_file)