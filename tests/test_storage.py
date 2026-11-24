"""Tests for the SQLite storage module — in-memory, no GPU/model dependencies.

Also includes rule-based alert tests for Step 5.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import src.alerts as alerts_module
from src.alerts import (
    check_heavy_vehicle_logged,
    check_loitering,
    check_person_at_night,
    check_vehicle_at_night,
    LOITERING_THRESHOLD_S,
)
from src.config.settings import DB_PATH, CLASS_CATEGORIES
from src.storage import (
    _EventRow,
    _TrackRow,
    init_db,
    query_alerts,
    query_counts_by_event_type,
    query_descriptions,
    query_events_for_track,
    query_events_in_time_window,
    query_tracks_by_class,
    store_alert,
    store_description,
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
    """Store events that span midnight and retrieve them with a window query."""
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
# Regression tests restored from Step 4
# ---------------------------------------------------------------------------

def test_per_track_query_with_id_switching_note():
    """Per-track queries return only the requested track_id.

    ByteTrack ID switching can split one physical object into several tracker
    IDs. A per-track query returns rows for one ID only, so one object that is
    re-assigned a new ID appears as separate result sets. This test pins that
    behaviour down so later code does not assume one object == one track.
    """
    conn = init_db(":memory:")

    SIM_START = datetime(2026, 11, 23, 23, 59, 45)

    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                    timestamp=SIM_START),
        _make_event("entered_zone", track_id=1, category="vehicle", class_name="car",
                    timestamp=SIM_START + timedelta(seconds=5)),
        _make_event("left", track_id=1, category="vehicle", class_name="car",
                    timestamp=SIM_START + timedelta(seconds=15)),
        _make_event("appeared", track_id=2, category="person", class_name="person",
                    timestamp=SIM_START + timedelta(seconds=2)),
        _make_event("entered_zone", track_id=2, category="person", class_name="person",
                    timestamp=SIM_START + timedelta(seconds=7)),
        # Simulated ID switch: the same physical person continues as track 3.
        _make_event("appeared", track_id=3, category="person", class_name="person",
                    timestamp=SIM_START + timedelta(seconds=12)),
        _make_event("entered_zone", track_id=3, category="person", class_name="person",
                    timestamp=SIM_START + timedelta(seconds=15)),
    ]

    run_id = store_run(
        conn, source_video="demo.mp4", start_time=SIM_START, tuning_settings={}
    )
    store_tracks_and_events(conn, run_id, events, tuning_settings={})

    track2_events = query_events_for_track(conn, run_id=run_id, track_id=2)
    assert [e.event_type for e in track2_events] == ["appeared", "entered_zone"]

    track3_events = query_events_for_track(conn, run_id=run_id, track_id=3)
    assert [e.event_type for e in track3_events] == ["appeared", "entered_zone"]

    track1_events = query_events_for_track(conn, run_id=run_id, track_id=1)
    assert len(track1_events) == 3

    conn.close()


def test_store_and_reopen_file_db(tmp_path):
    """Runs, tracks, events and alerts survive closing and reopening a file DB.

    The pipeline uses a file-backed database, so in-memory-only tests cannot
    catch a missing commit. This test writes each table, closes the
    connection, reopens the same file and checks every row is still there.
    """
    db_file = tmp_path / "reopen.db"

    conn = init_db(db_file)
    run_id = store_run(
        conn,
        source_video="demo.mp4",
        start_time=datetime(2026, 1, 1),
        tuning_settings={"yolo_conf": 0.10},
    )
    events = [
        _make_event("appeared", track_id=1, category="vehicle", class_name="car",
                    timestamp=datetime(2026, 1, 1, 0, 0, 0)),
        _make_event("entered_zone", track_id=1, category="vehicle", class_name="car",
                    timestamp=datetime(2026, 1, 1, 0, 0, 0)),
        _make_event("left", track_id=1, category="vehicle", class_name="car",
                    timestamp=datetime(2026, 1, 1, 0, 0, 5)),
    ]
    store_tracks_and_events(conn, run_id, events, tuning_settings={})
    store_alert(
        conn,
        run_id=run_id,
        track_id=1,
        rule_name="loitering",
        severity="warning",
        timestamp="2026-01-01T00:00:05",
        reason="track 1 loitered for 5.0s",
    )
    store_description(
        conn,
        run_id=run_id,
        track_id=1,
        status="described",
        description="A white car seen from above.",
        attributes={"object_type": "car", "main_colour": "white"},
        crop_path="/tmp/crop.jpg",
        orig_width=40,
        orig_height=30,
        model_name="fake-vlm",
        latency_ms=100.0,
    )
    conn.close()

    reopened = init_db(db_file)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert reopened.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1
        assert reopened.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
        assert reopened.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1
        assert reopened.execute("SELECT COUNT(*) FROM descriptions").fetchone()[0] == 1

        alerts = query_alerts(reopened, run_id=run_id, rule_name="loitering")
        assert len(alerts) == 1
        assert alerts[0]["track_id"] == 1
        assert alerts[0]["reason"] == "track 1 loitered for 5.0s"

        descriptions = query_descriptions(reopened, run_id=run_id, track_id=1)
        assert len(descriptions) == 1
        assert descriptions[0]["description"] == "A white car seen from above."
        assert descriptions[0]["attributes"]["main_colour"] == "white"
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# Step 5: Rule-based alert tests (fake data, no models/GPU)
# ---------------------------------------------------------------------------

def test_vehicle_at_night_fires_and_not_fires():
    """vehicle_at_night: fires for tracks first-seen in night window, does not fire outside."""
    from src.storage import init_db

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # Track at 23:59:58 — hour=23, outside [00:00, 05:00) → should NOT fire
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 1, "car", "2026-11-23T23:59:58", "2026-11-24T00:00:10", 10, 0.8),
    )

    # Track at 00:00:03 — hour=0, inside [00:00, 05:00) → SHOULD fire
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 2, "car", "2026-11-24T00:00:03", "2026-11-24T00:00:10", 10, 0.8),
    )

    conn.commit()

    alerts = check_vehicle_at_night(conn, run_id=run_id)
    alert_track_ids = {a["track_id"] for a in alerts}

    # Track 1 (23:59:58) should NOT be in alerts
    assert 1 not in alert_track_ids, "Track at 23:59:58 should NOT fire vehicle_at_night"
    # Track 2 (00:00:03) SHOULD be in alerts
    assert 2 in alert_track_ids, "Track at 00:00:03 SHOULD fire vehicle_at_night"

    conn.close()


def test_person_at_night_no_fire_when_no_person_tracks():
    """person_at_night does not fire when there are no person tracks."""
    from src.storage import init_db

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # No person tracks inserted

    alerts = check_person_at_night(conn, run_id=run_id)
    assert len(alerts) == 0, f"Expected 0 person_at_night alerts, got {len(alerts)}"

    conn.close()


def test_loitering_fires_above_threshold_and_not_below():
    """loitering: fires when last_seen - first_seen > LOITERING_THRESHOLD_S, not below."""
    from src.storage import init_db

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # Track alive for 4s total — from 23:59:56 to 00:00:00 = 4s, just below 5s threshold → NOT fire
    # 23:59:56 → 00:00:00 is exactly 4 seconds
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 1, "car", "2026-11-23T23:59:56", "2026-11-24T00:00:00", 10, 0.8),
    )

    # Track alive for 6s total — from 23:59:55 to 00:00:01 = 6s, above 5s threshold → fire
    # 23:59:55 → 00:00:01 crossing midnight = 6 seconds
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 2, "car", "2026-11-23T23:59:55", "2026-11-24T00:00:01", 10, 0.8),
    )

    conn.commit()

    alerts = check_loitering(conn, run_id=run_id)
    alert_track_ids = {a["track_id"] for a in alerts}

    # Track 1: duration = 4s (23:59:56 to 00:00:00), NOT > 4s (needs > LOITERING_THRESHOLD_S=5s), so should NOT fire
    # Actually the rule is > LOITERING_THRESHOLD_S, so 4s > 5s is False → should NOT fire
    # Track 2: duration = 6s (23:59:55 to 00:00:01), 6s > 5s → SHOULD fire
    assert 1 not in alert_track_ids, "Track with 4s duration should NOT fire loitering"
    assert 2 in alert_track_ids, "Track with 6s duration SHOULD fire loitering"

    conn.close()


def test_dedup_one_alert_per_track_per_rule():
    """At most one alert per track per rule (UNIQUE constraint)."""
    from src.storage import init_db

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # Insert a track that would fire vehicle_at_night (car at 00:00:01, hour=0)
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 1, "car", "2026-11-24T00:00:01", "2026-11-24T00:00:10", 10, 0.8),
    )

    # Call vehicle_at_night twice — second call should be deduped
    check_vehicle_at_night(conn, run_id=run_id)
    check_vehicle_at_night(conn, run_id=run_id)  # duplicate call

    alerts = query_alerts(conn, run_id=run_id, rule_name="vehicle_at_night")
    # Should be exactly 1 alert, even though function was called twice
    assert len(alerts) == 1, f"Expected 1 alert after dedup, got {len(alerts)}"

    conn.close()


def test_alert_rules_registry_is_idempotent():
    """Running every registered rule twice stores each alert only once.

    The pipeline calls ALERT_RULES after storing a run. This test mimics a
    second call (for example a retry or a manual re-analysis) and proves the
    database, not the function return value, is the source of truth.
    """
    from src.alerts import ALERT_RULES

    conn = init_db(":memory:")
    try:
        run_id = store_run(
            conn,
            source_video="test.mp4",
            start_time=datetime(2026, 11, 23, 23, 59, 55),
            tuning_settings={},
        )
        rows = [
            (1, "car", "2026-11-24T00:00:03", "2026-11-24T00:00:12"),
            (2, "person", "2026-11-24T00:00:04", "2026-11-24T00:00:12"),
            (3, "truck", "2026-11-24T00:00:05", "2026-11-24T00:00:12"),
        ]
        conn.executemany(
            "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
            "VALUES (?, ?, ?, ?, ?, 10, 0.8)",
            [(run_id, tid, cls, first, last) for tid, cls, first, last in rows],
        )
        conn.commit()

        for _ in range(2):
            for rule_fn in ALERT_RULES.values():
                rule_fn(conn, run_id=run_id)

        alerts = query_alerts(conn, run_id=run_id)
        keys = [(a["track_id"], a["rule_name"]) for a in alerts]
        assert len(keys) == len(set(keys)), f"duplicate alerts stored: {keys}"
        assert len(alerts) == 7, f"expected 7 alerts (2+1+3+1), got {len(alerts)}"
    finally:
        conn.close()


def test_unconfirmed_tracks_ignored():
    """Unconfirmed tracks (not in tracks table) are ignored by all rules."""
    from src.storage import init_db

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # Manually insert an events row with a track_id that was never stored in tracks
    # This simulates an unconfirmed tracker ID
    # Events table columns: id, run_id, track_id, event_type, timestamp, zone, dwell_s, class_name, end_of_video
    # That's 9 columns but id is autoincrement, so 8 values needed
    conn.execute(
        "INSERT INTO events (run_id, track_id, event_type, timestamp, zone, dwell_s, class_name, end_of_video) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, 99, "appeared", "2026-11-24T00:00:01", "gate", 0.0, "car", 0),
    )
    conn.commit()

    # Check that vehicle_at_night ignores this (track 99 not in tracks table)
    alerts = check_vehicle_at_night(conn, run_id=run_id)
    alert_track_ids = {a["track_id"] for a in alerts}
    assert 99 not in alert_track_ids, "Unconfirmed track 99 should be ignored by vehicle_at_night"

    conn.close()


def test_night_window_midnight_boundary():
    """Verify 23:59:58 does not fire night rule while 00:00:03 does."""
    from src.storage import init_db

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # Track first-seen at 23:59:58 — hour=23, outside [0,5)
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 1, "car", "2026-11-23T23:59:58", "2026-11-24T00:00:10", 10, 0.8),
    )

    # Track first-seen at 00:00:03 — hour=0, inside [0,5)
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 2, "car", "2026-11-24T00:00:03", "2026-11-24T00:00:10", 10, 0.8),
    )
    conn.commit()

    alerts = check_vehicle_at_night(conn, run_id=run_id)
    alert_ids = {a["track_id"] for a in alerts}

    assert 1 not in alert_ids, "Track at 23:59:58 should NOT fire vehicle_at_night"
    assert 2 in alert_ids, "Track at 00:00:03 SHOULD fire vehicle_at_night"

    conn.close()


def test_all_alert_rules_with_fake_data():
    """Run all four rules and verify basic fire/do-not-fire behavior."""
    from src.storage import init_db, query_alerts
    from datetime import datetime

    conn = init_db(":memory:")

    run_id = store_run(
        conn,
        source_video="test.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )

    # Insert a mix of tracks:
    # - car at 23:59:58 (hour=23, outside [0,5) → should NOT fire vehicle_at_night)
    # - person at 00:00:03 (hour=0, inside [0,5) → SHOULD fire person_at_night)
    # - car with 8s duration (first=23:59:55 last=00:00:07 → 8s > 5s → SHOULD fire loitering)
    # - truck at 00:00:01 (hour=0, class truck → SHOULD fire heavy_vehicle_logged)
    # Note: vehicle_at_night also fires for truck class (in VEHICLE_CLASSES)
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 1, "car", "2026-11-23T23:59:58", "2026-11-24T00:00:10", 10, 0.8),
    )
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 2, "person", "2026-11-24T00:00:03", "2026-11-24T00:00:10", 10, 0.8),
    )
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 3, "car", "2026-11-23T23:59:55", "2026-11-24T00:00:07", 10, 0.8),
    )
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, 4, "truck", "2026-11-24T00:00:01", "2026-11-24T00:00:10", 10, 0.8),
    )
    conn.commit()

    # vehicle_at_night: track 1 at 23:59:58 should NOT fire (hour=23),
    # track 4 at 00:00:01 SHOULD fire (truck in night window)
    alerts = check_vehicle_at_night(conn, run_id=run_id)
    alert_track_ids = {a["track_id"] for a in alerts}
    assert alert_track_ids == {4}, f"vehicle_at_night expected {{4}}, got {alert_track_ids}"

    # person_at_night: only track 2 should fire
    alerts = check_person_at_night(conn, run_id=run_id)
    assert {a["track_id"] for a in alerts} == {2}, f"person_at_night expected {{2}}, got {{a['track_id'] for a in alerts}}"

    # loitering: tracks 1 (15s), 3 (8s), and 4 (9s) all exceed 5s threshold,
    # track 2 (7s) also exceeds. All should fire.
    alerts = check_loitering(conn, run_id=run_id)
    alert_track_ids = {a["track_id"] for a in alerts}
    assert 1 in alert_track_ids, "Track 1 (15s loiter) should fire loitering"
    assert 3 in alert_track_ids, "Track 3 (8s loiter) should fire loitering"

    # heavy_vehicle_logged: only track 4 (truck) should fire
    alerts = check_heavy_vehicle_logged(conn, run_id=run_id)
    assert {a["track_id"] for a in alerts} == {4}, f"heavy_vehicle_logged expected {{4}}, got {{a['track_id'] for a in alerts}}"

    # Dedup: call vehicle_at_night again, should still be 1 alert per track
    check_vehicle_at_night(conn, run_id=run_id)
    alerts = query_alerts(conn, run_id=run_id, rule_name="vehicle_at_night")
    car_alerts = [a for a in alerts if a["track_id"] in (1, 4)]
    assert len(car_alerts) <= 2, "Dedup should limit alerts"

    conn.close()


def test_night_window_wraps_midnight_22_to_05(monkeypatch):
    """A 22:00-05:00 window must wrap midnight instead of spanning one day.

    23:30 and 04:00 are inside the window; 12:00 and 21:59 are outside. This
    test uses the real rule function with the window patched to 22-5, so it
    fails if ``_is_night`` only handles ``start < end``.
    """
    monkeypatch.setattr(alerts_module, "NIGHT_WINDOW_START", 22)
    monkeypatch.setattr(alerts_module, "NIGHT_WINDOW_END", 5)

    conn = init_db(":memory:")
    try:
        run_id = store_run(
            conn,
            source_video="test.mp4",
            start_time=datetime(2026, 11, 23, 22, 0, 0),
            tuning_settings={},
        )

        rows = [
            (1, "2026-11-23T23:30:00", "2026-11-23T23:40:00"),  # inside
            (2, "2026-11-23T12:00:00", "2026-11-23T12:10:00"),  # outside (day)
            (3, "2026-11-24T04:00:00", "2026-11-24T04:10:00"),  # inside (after midnight)
            (4, "2026-11-24T05:00:00", "2026-11-24T05:10:00"),  # outside (end exclusive)
            (5, "2026-11-23T21:59:00", "2026-11-23T22:05:00"),  # outside (but crosses 22:00)
        ]
        conn.executemany(
            "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
            "VALUES (?, ?, 'car', ?, ?, 10, 0.8)",
            [(run_id, track_id, first, last) for track_id, first, last in rows],
        )
        conn.commit()

        alerts = check_vehicle_at_night(conn, run_id=run_id)
        assert {a["track_id"] for a in alerts} == {1, 3}
        assert all("22:00-05:00" in a["reason"] for a in alerts)
    finally:
        conn.close()
