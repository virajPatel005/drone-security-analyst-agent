"""Frame-index tests: schema, idempotent writes, queries, tool and CLI.

No network, no GPU, no YOLO, no VLM, no OpenAI: the rows are built from
dataclasses and written to temporary SQLite files.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import agent
from src.indexing import FrameObjectRow, FrameRow, build_frame_index
from src.perception.observations import Observation
from src.storage import (
    FRAME_QUERY_DEFAULT_LIMIT,
    init_db,
    query_frames_by_class,
    query_frames_for_track,
    query_frames_in_time_window,
    query_objects_by_class,
    store_alert,
    store_frame_index,
    store_run,
    store_tracks_and_events,
)
from src.telemetry.records import FrameRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "query_frames.py"

# The backfill script lives in scripts/ (not a package), so import it by path.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import backfill_frames  # noqa: E402


def _record(index: int, when: datetime, zone: str = "gate", description: str | None = None):
    return FrameRecord(
        frame_index=index,
        video_time_s=float(index),
        timestamp=when,
        location=zone,
        altitude_m=30.0,
        drone_x=0.0,
        drone_y=0.0,
        description=description,
    )


def _event(
    event_type: str, track_id: int, class_name: str, when: datetime
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        event_type=event_type,
        timestamp=when,
        zone="gate",
        dwell_s=0.0,
        class_name=class_name,
        end_of_video=event_type == "left",
        confidence=0.9,
    )


def _seed(conn: sqlite3.Connection) -> int:
    """Two runs' worth of small, deterministic frame-index data in one run."""
    run_id = store_run(
        conn,
        source_video="fake.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )
    base = datetime(2026, 11, 23, 23, 59, 58)
    frames: list[FrameRow] = []
    objects: list[FrameObjectRow] = []
    for index, (offset, specs) in enumerate(
        [
            (0, [(2, "truck", 0.8), (3, "car", 0.7)]),
            (1, [(2, "truck", 0.9)]),
            (2, [(103, "truck", 0.6), (3, "car", 0.5)]),
        ],
        start=1,
    ):
        record = _record(index, base + timedelta(seconds=offset), "gate")
        observations = [
            Observation(tid, cls, conf, (1.0, 2.0, 3.0, 4.0))
            for tid, cls, conf in specs
        ]
        frame, frame_objects = build_frame_index(record, observations)
        frames.append(frame)
        objects.extend(frame_objects)
    store_frame_index(conn, run_id, frames, objects)
    return run_id


def test_schema_creates_frame_tables_and_indexes() -> None:
    conn = init_db(":memory:")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    for table in ("frames", "frame_objects"):
        assert table in names
    for index in (
        "idx_frames_run_timestamp",
        "idx_frame_objects_run_class",
        "idx_frame_objects_run_track",
    ):
        assert index in indexes
    frame_cols = {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
    object_cols = {r[1] for r in conn.execute("PRAGMA table_info(frame_objects)")}
    assert {
        "run_id",
        "frame_index",
        "timestamp",
        "zone",
        "detection_count",
        "description",
    } <= frame_cols
    assert {
        "run_id",
        "frame_index",
        "track_id",
        "class_name",
        "confidence",
        "bbox",
    } <= object_cols
    conn.close()


def test_build_frame_index_counts_detections_and_keeps_text_description() -> None:
    record = _record(1, datetime(2026, 11, 23, 23, 59, 58), "garage", "blue truck at gate")
    observations = [
        Observation(2, "truck", 0.8, (1.0, 2.0, 3.0, 4.0)),
        Observation(3, "car", 0.7, (5.0, 6.0, 7.0, 8.0)),
    ]
    frame, objects = build_frame_index(record, observations)
    assert frame.frame_index == 1
    assert frame.zone == "garage"
    assert frame.description == "blue truck at gate"
    assert frame.detection_count == 2 == len(objects)
    assert [o.track_id for o in objects] == [2, 3]
    assert objects[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_store_frame_index_is_idempotent(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "idem.db")
    run_id = _seed(conn)
    first_frames = query_frames_in_time_window(
        conn,
        run_id=run_id,
        start="2026-11-23T23:59:00",
        end="2026-11-24T00:00:10",
        limit=100,
    )
    first_objects = query_objects_by_class(conn, run_id=run_id, class_name="truck", limit=100)

    # Re-index the same run twice with the same data.
    frames = [
        FrameRow(1, datetime(2026, 11, 23, 23, 59, 58), "gate", 2),
        FrameRow(2, datetime(2026, 11, 23, 23, 59, 59), "gate", 1),
        FrameRow(3, datetime(2026, 11, 24, 0, 0, 0), "gate", 2),
    ]
    objects = [
        FrameObjectRow(1, 2, "truck", 0.8, (1.0, 2.0, 3.0, 4.0)),
        FrameObjectRow(1, 3, "car", 0.7, (1.0, 2.0, 3.0, 4.0)),
        FrameObjectRow(2, 2, "truck", 0.9, (1.0, 2.0, 3.0, 4.0)),
        FrameObjectRow(3, 103, "truck", 0.6, (1.0, 2.0, 3.0, 4.0)),
        FrameObjectRow(3, 3, "car", 0.5, (1.0, 2.0, 3.0, 4.0)),
    ]
    store_frame_index(conn, run_id, frames, objects)
    store_frame_index(conn, run_id, frames, objects)

    assert conn.execute("SELECT COUNT(*) FROM frames WHERE run_id=?", (run_id,)).fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM frame_objects WHERE run_id=?", (run_id,)).fetchone()[0] == 5
    again_frames = query_frames_in_time_window(
        conn,
        run_id=run_id,
        start="2026-11-23T23:59:00",
        end="2026-11-24T00:00:10",
        limit=100,
    )
    again_objects = query_objects_by_class(conn, run_id=run_id, class_name="truck", limit=100)
    assert again_frames == first_frames
    assert again_objects["total"] == first_objects["total"]
    conn.close()


def test_store_frame_index_does_not_touch_other_tables(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "protected.db")
    run_id = store_run(conn, source_video="fake.mp4", start_time=datetime(2026, 1, 1))
    when = datetime(2026, 1, 1)
    store_tracks_and_events(
        conn,
        run_id,
        [_event("appeared", 5, "truck", when), _event("left", 5, "truck", when)],
    )
    store_alert(
        conn,
        run_id=run_id,
        track_id=5,
        rule_name="heavy_vehicle_logged",
        severity="info",
        timestamp="2026-01-01T00:00:00",
        reason="test",
    )
    tables = ("runs", "tracks", "events", "alerts", "descriptions", "agent_costs")
    before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}

    store_frame_index(
        conn,
        run_id,
        [FrameRow(1, when, "gate", 1)],
        [FrameObjectRow(1, 5, "truck", 0.8, (0.0, 0.0, 1.0, 1.0))],
    )

    after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    assert after == before
    conn.close()


def test_query_time_window_crosses_midnight(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "window.db")
    run_id = _seed(conn)
    all_frames = query_frames_in_time_window(
        conn,
        run_id=run_id,
        start="2026-11-23T23:59:55",
        end="2026-11-24T00:00:10",
    )
    assert all_frames["total"] == 3
    assert [r["timestamp"] for r in all_frames["rows"]] == [
        "2026-11-23T23:59:58",
        "2026-11-23T23:59:59",
        "2026-11-24T00:00:00",
    ]
    before_midnight = query_frames_in_time_window(
        conn,
        run_id=run_id,
        start="2026-11-23T23:59:55",
        end="2026-11-23T23:59:59",
    )
    assert before_midnight["total"] == 2
    assert all(r["timestamp"].startswith("2026-11-23") for r in before_midnight["rows"])
    conn.close()


def test_query_by_class_and_track(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "class.db")
    run_id = _seed(conn)
    truck_frames = query_frames_by_class(conn, run_id=run_id, class_name="truck")
    assert truck_frames["total"] == 3
    assert [r["frame_index"] for r in truck_frames["rows"]] == [1, 2, 3]

    truck_objects = query_objects_by_class(conn, run_id=run_id, class_name="truck")
    assert truck_objects["total"] == 3
    assert {r["track_id"] for r in truck_objects["rows"]} == {2, 103}
    assert all(r["class_name"] == "truck" for r in truck_objects["rows"])

    track_2 = query_frames_for_track(conn, run_id=run_id, track_id=2)
    assert track_2["total"] == 2
    assert [r["frame_index"] for r in track_2["rows"]] == [1, 2]
    assert query_frames_for_track(conn, run_id=run_id, track_id=999)["total"] == 0
    conn.close()


def test_default_limit_50_and_truncated_flag(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "limit.db")
    run_id = store_run(conn, source_video="fake.mp4", start_time=datetime(2026, 1, 1))
    base = datetime(2026, 1, 1)
    frames = [
        FrameRow(i, base + timedelta(seconds=i), "gate", 0) for i in range(1, 61)
    ]
    store_frame_index(conn, run_id, frames, [])
    assert FRAME_QUERY_DEFAULT_LIMIT == 50

    result = query_frames_in_time_window(
        conn,
        run_id=run_id,
        start="2026-01-01T00:00:00",
        end="2026-01-01T00:02:00",
    )
    assert result["total"] == 60
    assert result["showing"] == 50
    assert result["truncated"] is True
    assert result["notice"] == "showing 50 of 60"

    small = query_frames_in_time_window(
        conn,
        run_id=run_id,
        start="2026-01-01T00:00:00",
        end="2026-01-01T00:02:00",
        limit=5,
    )
    assert small["showing"] == 5 and small["total"] == 60
    with pytest.raises(ValueError):
        query_frames_in_time_window(
            conn, run_id=run_id, start="2026-01-01", end="2026-01-02", limit=0
        )
    conn.close()


def test_queries_are_read_only(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "readonly.db")
    run_id = _seed(conn)
    tables = ("runs", "tracks", "events", "frames", "frame_objects")
    before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    query_frames_in_time_window(
        conn, run_id=run_id, start="2026-11-23", end="2026-11-25"
    )
    query_frames_by_class(conn, run_id=run_id, class_name="truck")
    query_objects_by_class(conn, run_id=run_id, class_name="truck")
    query_frames_for_track(conn, run_id=run_id, track_id=2)
    after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    assert after == before
    conn.close()


def test_agent_tool_is_allowed_and_returns_compact_json(tmp_path: Path) -> None:
    assert "query_frames" in agent.ALLOWED_TOOLS
    assert "query_frames" in {t["function"]["name"] for t in agent.TOOL_DEFINITIONS}
    conn = init_db(tmp_path / "tool.db")
    run_id = _seed(conn)
    for args in (
        {"run_id": run_id, "query": "time_window", "start": "2026-11-23T23:59:55", "end": "2026-11-24T00:00:10"},
        {"run_id": run_id, "query": "class_frames", "class_name": "truck"},
        {"run_id": run_id, "query": "class_objects", "class_name": "truck"},
        {"run_id": run_id, "query": "track_frames", "track_id": 2},
    ):
        payload = agent._run_tool("query_frames", args, conn)
        result = json.loads(payload)
        assert {"total", "showing", "truncated", "rows"} <= result.keys()
        for forbidden in ("image", "base64", "crop_path", "data:image"):
            assert forbidden not in payload.lower()
        assert len(payload) < 4000
    conn.close()


def test_old_agent_tools_are_unchanged(tmp_path: Path) -> None:
    original = {
        "list_runs",
        "get_alerts",
        "get_tracks",
        "get_events_for_track",
        "get_descriptions",
        "counts_by_event_type",
    }
    assert original <= set(agent.ALLOWED_TOOLS)
    assert original <= {t["function"]["name"] for t in agent.TOOL_DEFINITIONS}
    conn = init_db(tmp_path / "old.db")
    run_id = store_run(conn, source_video="fake.mp4", start_time=datetime(2026, 1, 1))
    store_tracks_and_events(
        conn,
        run_id,
        [_event("appeared", 5, "truck", datetime(2026, 1, 1))],
    )
    events = json.loads(
        agent._run_tool("get_events_for_track", {"run_id": run_id, "track_id": 5}, conn)
    )
    assert events["total"] == 1
    assert events["rows"][0]["event_type"] == "appeared"
    counts = json.loads(agent._run_tool("counts_by_event_type", {"run_id": run_id}, conn))
    assert counts == {"counts": {"appeared": 1}}
    conn.close()


def test_cli_queries_frame_index(tmp_path: Path) -> None:
    db_path = tmp_path / "cli.db"
    conn = init_db(db_path)
    _seed(conn)
    conn.close()

    def run(*extra: str) -> str:
        proc = subprocess.run(
            [sys.executable, str(CLI), "--run", "1", "--db", str(db_path), *extra],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    truck = run("--class", "truck", "--limit", "100")
    assert "distinct_track_ids=[2, 103]" in truck
    assert "total=3 showing=3" in truck

    window = run("--from", "2026-11-23T23:59:55", "--to", "2026-11-24T00:00:10")
    assert "total=3 showing=3" in window
    assert "2026-11-24T00:00:00" in window

    track = run("--track", "2")
    assert "frames for track=2 run=1" in track
    assert "total=2 showing=2" in track


# ---------------------------------------------------------------------------
# Backfill ID check: one-way rule (confirmed tracks must be present; extras OK)
# ---------------------------------------------------------------------------


def _id_check_db(tmp_path: Path, *, omit_confirmed: bool = False):
    """Two confirmed tracks (2, 3) and frame detections {2, 103[, 3]}.
    103 is an unconfirmed single-frame detection; omitting 3 simulates a broken
    index where a confirmed track has no stored detection.
    """
    conn = init_db(tmp_path / "idcheck.db")
    run_id = store_run(conn, source_video="fake.mp4", start_time=datetime(2026, 1, 1))
    when = datetime(2026, 1, 1)
    store_tracks_and_events(
        conn,
        run_id,
        [
            _event("appeared", 2, "truck", when),
            _event("appeared", 3, "car", when),
        ],
    )
    track_ids = [2, 103] if omit_confirmed else [2, 3, 103]
    objects = [
        FrameObjectRow(1, tid, "truck" if tid == 2 else "car", 0.8, (0.0, 0.0, 1.0, 1.0))
        for tid in track_ids
    ]
    store_frame_index(conn, run_id, [FrameRow(1, when, "gate", len(objects))], objects)
    return conn, run_id


def test_backfill_id_check_allows_unconfirmed_detections(tmp_path: Path, capsys) -> None:
    conn, run_id = _id_check_db(tmp_path)
    assert backfill_frames._compare_track_ids(conn, run_id) is True
    captured = capsys.readouterr().out
    assert "track_id_check: PASS" in captured
    assert "unconfirmed_detections: count=1 ids=[103]" in captured
    conn.close()


def test_backfill_id_check_fails_for_missing_confirmed_track(tmp_path: Path, capsys) -> None:
    conn, run_id = _id_check_db(tmp_path, omit_confirmed=True)
    assert backfill_frames._compare_track_ids(conn, run_id) is False
    captured = capsys.readouterr()
    assert "track_id_check: FAIL" in captured.err
    assert "missing_confirmed_tracks=[3]" in captured.err
    conn.close()
