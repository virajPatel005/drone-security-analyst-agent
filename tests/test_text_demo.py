"""No-network tests for the JSONL text-simulation demo.

These exercise the parser, the reuse of existing storage/alert code, frame
queries and idempotency. They test the alert logic only, never the vision
pipeline (YOLO, ByteTrack or the VLM are not imported here).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from src.storage import (
    init_db,
    query_alerts,
    query_frames_by_class,
    query_frames_for_track,
    query_frames_in_time_window,
)
from src.text_demo import ingest_scenario, load_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = PROJECT_ROOT / "data" / "scenarios"
VISIT_GAP_S = 300.0


def _scenario(name: str) -> Path:
    return SCENARIOS / f"{name}.jsonl"


def _ingest(db_path: Path, name: str) -> tuple[sqlite3.Connection, object]:
    conn = init_db(db_path)
    result = ingest_scenario(conn, load_scenario(_scenario(name)), visit_gap_s=VISIT_GAP_S)
    return conn, result


def _counts(conn: sqlite3.Connection, *tables: str) -> tuple[int, ...]:
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables)


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "scenario.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parser_accepts_full_iso_and_date_plus_time(tmp_path):
    path = _write(
        tmp_path,
        '{"time":"2026-11-24T00:00:00","zone":"gate","description":"first","objects":[]}',
        '{"date":"2026-11-24","time":"00:01:00","zone":"gate","description":"second",'
        '"objects":[{"class":"person","label":"p1","color":"dark","text":"a person"}]}',
    )

    scenario = load_scenario(path)

    assert scenario.name == "scenario"
    assert [f.timestamp for f in scenario.frames] == [
        datetime(2026, 11, 24, 0, 0), datetime(2026, 11, 24, 0, 1)
    ]
    assert scenario.frames[1].description == "second"
    assert scenario.frames[1].objects[0].class_name == "person"
    assert scenario.frames[1].objects[0].label == "p1"


def test_parser_rejects_missing_time(tmp_path):
    path = _write(tmp_path, '{"date":"2026-11-24","zone":"gate","description":"x","objects":[]}')

    with pytest.raises(ValueError, match="time"):
        load_scenario(path)


def test_parser_rejects_unknown_class(tmp_path):
    path = _write(
        tmp_path,
        '{"time":"2026-11-24T12:00:00","zone":"gate","description":"x",'
        '"objects":[{"class":"alien","label":"x1"}]}',
    )

    with pytest.raises(ValueError, match="unknown class"):
        load_scenario(path)


def test_parser_rejects_bad_time_format(tmp_path):
    path = _write(
        tmp_path,
        '{"date":"2026-11-24","time":"25:99:99","zone":"gate","description":"x","objects":[]}',
    )

    with pytest.raises(ValueError, match="bad"):
        load_scenario(path)


def test_parser_rejects_out_of_order_frames(tmp_path):
    path = _write(
        tmp_path,
        '{"time":"2026-11-24T12:00:00","zone":"gate","description":"a","objects":[]}',
        '{"time":"2026-11-24T11:00:00","zone":"gate","description":"b","objects":[]}',
    )

    with pytest.raises(ValueError, match="strictly increase"):
        load_scenario(path)


def test_same_label_produces_one_track(tmp_path):
    conn, result = _ingest(tmp_path / "one.db", "midnight_gate")
    tracks = conn.execute(
        "SELECT track_id, class_name FROM tracks WHERE run_id = ?", (result.run_id,)
    ).fetchall()
    assert tracks == [(1, "person")]
    assert result.tracks == 1
    conn.close()


def test_midnight_scenario_fires_person_at_night_and_loitering(tmp_path):
    conn, result = _ingest(tmp_path / "midnight.db", "midnight_gate")
    alerts = query_alerts(conn, run_id=result.run_id)
    by_rule = {a["rule_name"]: a for a in alerts}

    assert set(by_rule) == {"person_at_night", "loitering"}
    assert by_rule["person_at_night"]["severity"] == "warning"
    assert by_rule["person_at_night"]["timestamp"] == "2026-11-24T00:00:00"
    assert "180.0s" in by_rule["loitering"]["reason"]
    assert result.alerts.get("vehicle_at_night", 0) == 0
    assert result.alerts.get("heavy_vehicle_logged", 0) == 0
    conn.close()


def test_daytime_truck_logs_two_appearances_and_one_heavy_alert_per_track(tmp_path):
    conn, result = _ingest(tmp_path / "truck.db", "daytime_truck")
    appeared = conn.execute(
        "SELECT timestamp FROM events WHERE run_id = ? AND track_id = ? AND event_type = 'appeared' "
        "ORDER BY timestamp",
        (result.run_id, 1),
    ).fetchall()
    heavy = query_alerts(conn, run_id=result.run_id, rule_name="heavy_vehicle_logged")

    assert len(appeared) == 2
    assert [row[0] for row in appeared] == ["2026-11-24T12:00:00", "2026-11-24T16:00:00"]
    assert len(heavy) == 1
    assert heavy[0]["track_id"] == 1
    assert heavy[0]["severity"] == "info"
    assert result.tracks == 1
    conn.close()


def test_quiet_scenario_fires_nothing(tmp_path):
    conn, result = _ingest(tmp_path / "quiet.db", "quiet_daytime")

    assert query_alerts(conn, run_id=result.run_id) == []
    assert result.alerts == {}
    conn.close()


def test_ingesting_twice_creates_no_duplicates(tmp_path):
    db_path = tmp_path / "twice.db"
    conn, first = _ingest(db_path, "midnight_gate")
    before = _counts(conn, "runs", "tracks", "events", "frames", "frame_objects", "alerts")
    conn.close()

    conn, second = _ingest(db_path, "midnight_gate")
    after = _counts(conn, "runs", "tracks", "events", "frames", "frame_objects", "alerts")
    conn.close()

    assert second.run_id == first.run_id
    assert after == before == (1, 1, 3, 6, 4, 2)


def test_frame_queries_find_text_frames_by_window_class_and_track(tmp_path):
    conn, result = _ingest(tmp_path / "queries.db", "midnight_gate")
    window = query_frames_in_time_window(
        conn, run_id=result.run_id,
        start="2026-11-23T23:59:00", end="2026-11-24T00:03:00", limit=100,
    )
    by_class = query_frames_by_class(conn, run_id=result.run_id, class_name="person", limit=100)
    by_track = query_frames_for_track(conn, run_id=result.run_id, track_id=1, limit=100)

    assert window["total"] == 6
    assert by_class["total"] == 4
    assert by_track["total"] == 4
    assert [row["frame_index"] for row in by_track["rows"]] == [3, 4, 5, 6]
    conn.close()


def test_frame_row_stores_the_text_description(tmp_path):
    conn, result = _ingest(tmp_path / "text.db", "midnight_gate")
    row = conn.execute(
        "SELECT description FROM frames WHERE run_id = ? AND frame_index = 3", (result.run_id,)
    ).fetchone()

    assert row[0] == "A person in a dark jacket is now standing at the main gate."
    conn.close()


def test_all_committed_scenarios_parse():
    for name in ("midnight_gate", "daytime_truck", "quiet_daytime"):
        scenario = load_scenario(_scenario(name))
        assert scenario.frames
        assert all(frame.description for frame in scenario.frames)
