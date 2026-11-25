"""Shared fake database fixture for the read-only Streamlit app tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.storage import (
    init_db,
    record_agent_call,
    record_agent_trace,
    store_alert,
    store_description,
    store_run,
    store_run_summary,
    store_tracks_and_events,
)


@pytest.fixture
def app_db(tmp_path: Path) -> Path:
    """Build a small file-backed DB with a midnight boundary and a truck alert."""
    db_path = tmp_path / "app.db"
    conn = init_db(db_path)
    run_id = store_run(
        conn,
        source_video=str(tmp_path / "night_clip.mp4"),
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={"yolo_model": "yolo11s.pt", "sample_rate_fps": 10.0},
    )

    def event(
        track_id: int,
        kind: str,
        when: datetime,
        zone: str,
        *,
        dwell: float = 0.0,
        class_name: str = "car",
    ):
        # Real runs store concrete YOLO classes in tracks.class_name; using the
        # abstract "vehicle" category here used to hide the UI count defect.
        return SimpleNamespace(
            track_id=track_id,
            event_type=kind,
            timestamp=when,
            zone=zone,
            dwell_s=dwell,
            class_name=class_name,
            end_of_video=kind == "left",
            confidence=0.8,
        )

    events = [
        # 23:59:58 must NOT fire a night alert; the night window starts at 00:00.
        event(2, "appeared", datetime(2026, 11, 23, 23, 59, 58), "gate"),
        event(2, "entered_zone", datetime(2026, 11, 24, 0, 0, 1), "gate"),
        event(2, "left", datetime(2026, 11, 24, 0, 0, 5), "gate"),
        event(3, "appeared", datetime(2026, 11, 24, 0, 0, 3), "garage"),
        event(3, "left", datetime(2026, 11, 24, 0, 0, 9), "garage"),
        event(5, "appeared", datetime(2026, 11, 24, 0, 0, 4), "driveway", class_name="truck"),
        event(5, "entered_zone", datetime(2026, 11, 24, 0, 0, 6), "driveway", class_name="truck"),
        event(5, "left", datetime(2026, 11, 24, 0, 0, 10), "driveway", class_name="truck"),
    ]
    store_tracks_and_events(conn, run_id, events)

    store_alert(
        conn,
        run_id=run_id,
        track_id=3,
        rule_name="vehicle_at_night",
        severity="warning",
        timestamp="2026-11-24T00:00:03",
        reason="Vehicle track first seen at 00:00:03 during night hours",
        zone="garage",
    )
    store_alert(
        conn,
        run_id=run_id,
        track_id=5,
        rule_name="heavy_vehicle_logged",
        severity="info",
        timestamp="2026-11-24T00:00:04",
        reason="Heavy vehicle track 5 logged",
        zone="driveway",
    )

    crop_path = tmp_path / "track_000005.jpg"
    Image.new("RGB", (16, 16), (40, 60, 80)).save(crop_path)
    store_description(
        conn,
        run_id=run_id,
        track_id=5,
        status="described",
        description="A brick building facade, not clearly a truck.",
        attributes={"object_type": "truck", "main_colour": "grey"},
        crop_path=str(crop_path),
        orig_width=40,
        orig_height=30,
        model_name="gemma-4-e4b",
        latency_ms=1200.0,
    )
    store_description(
        conn,
        run_id=run_id,
        track_id=3,
        status="too_small",
        description=None,
        attributes=None,
        crop_path=None,
        orig_width=8,
        orig_height=8,
        model_name="gemma-4-e4b",
        latency_ms=None,
    )

    summary = {
        "summary": "A vehicle appeared just after midnight and a heavy vehicle was logged.",
        "notable_alerts": [
            {
                "track_id": 3,
                "rule_name": "vehicle_at_night",
                "severity": "warning",
                "timestamp": "2026-11-24T00:00:03",
                "reason": "Vehicle track first seen during night hours",
            }
        ],
        "questionable": [5],
        "recommendations": ["Review the heavy vehicle crop before treating it as a truck."],
    }
    store_run_summary(
        conn,
        run_id=run_id,
        text=json.dumps(summary),
        model_name="gpt-5.4-mini",
        prompt_tokens=3600,
        completion_tokens=800,
        estimated_cost_usd=0.01,
    )
    record_agent_call(
        conn,
        run_id=run_id,
        command="summary",
        model_name="gpt-5.4-mini",
        prompt_tokens=3600,
        completion_tokens=800,
        estimated_cost_usd=0.01,
        latency_s=3.0,
    )
    record_agent_trace(
        conn,
        run_id=run_id,
        mode="summary",
        question=None,
        tool_calls=[],
        answer=json.dumps(summary),
        model_name="gpt-5.4-mini",
        prompt_tokens=3600,
        completion_tokens=800,
        estimated_cost_usd=0.01,
        latency_ms=3000.0,
    )
    record_agent_trace(
        conn,
        run_id=run_id,
        mode="ask",
        question="List events for track 3",
        tool_calls=[{"name": "get_events_for_track", "arguments": "{}", "result_size": 120}],
        answer="Run 1: track 3 has an appeared and a left event.",
        model_name="gpt-5.4-mini",
        prompt_tokens=3400,
        completion_tokens=100,
        estimated_cost_usd=0.003,
        latency_ms=1500.0,
    )
    conn.close()
    return db_path
