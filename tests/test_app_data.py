"""No-network tests for the Streamlit data layer, using a temporary DB."""

from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.storage import init_db, record_agent_call

from app import data


def test_list_runs_counts_and_summary(app_db: Path) -> None:
    conn = data.connect_read_only(app_db)
    try:
        runs = data.list_runs(conn)
        assert len(runs) == 1
        assert runs[0].video_name == "night_clip.mp4"
        assert runs[0].start_time == "2026-11-23T23:59:55"
        assert runs[0].tuning["yolo_model"] == "yolo11s.pt"

        counts = data.run_counts(conn, runs[0].id)
        assert (counts.tracks, counts.events, counts.alerts) == (3, 8, 2)

        summary = data.get_summary(conn, runs[0].id)
        assert summary is not None
        assert summary.model_name == "gpt-5.4-mini"
        assert summary.estimated_cost_usd == 0.01
        assert summary.parsed is not None
        assert summary.parsed["summary"].startswith("A vehicle appeared")
        assert summary.questionable == [5]
    finally:
        conn.close()


def test_midnight_alerts_filters_and_provenance(app_db: Path) -> None:
    conn = data.connect_read_only(app_db)
    try:
        alerts = data.list_alerts(conn, 1)
        assert {alert.track_id for alert in alerts} == {3, 5}
        # The 23:59:58 track has no alert; the 00:00:03 track does.
        assert "2026-11-24T00:00:03" in {alert.timestamp for alert in alerts}

        info = data.list_alerts(conn, 1, severity="info")
        assert [alert.track_id for alert in info] == [5]
        rule = data.list_alerts(conn, 1, rule_name="heavy_vehicle_logged")
        assert [alert.track_id for alert in rule] == [5]

        provenance = data.get_provenance(conn, info[0])
        assert provenance.track is not None
        assert provenance.track.class_name == "vehicle"
        assert provenance.track.event_count == 3
        assert [event.event_type for event in provenance.events] == [
            "appeared",
            "entered_zone",
            "left",
        ]
        assert provenance.description is not None
        assert provenance.description.status == "described"
        assert "brick building facade" in (provenance.description.description or "")
        assert provenance.description.has_crop
        assert provenance.description.latency_ms == 1200.0
        assert provenance.tuning["yolo_model"] == "yolo11s.pt"
        # The badge reads only the stored summary's questionable list.
        assert provenance.flagged_questionable is True

        night = data.get_provenance(conn, alerts[0])
        assert night.description is not None and night.description.status == "too_small"
        assert not night.description.has_crop
        assert night.flagged_questionable is False
    finally:
        conn.close()


def test_traces_are_typed_and_cost_total_from_sqlite(app_db: Path) -> None:
    conn = data.connect_read_only(app_db)
    try:
        traces = data.list_traces(conn, 1)
        assert len(traces) == 2
        assert traces[0].mode == "summary" and traces[0].cached is False
        assert traces[1].tool_names == ["get_events_for_track"]
        assert data.agent_state(conn).running_total_usd == 0.01
    finally:
        conn.close()


def test_agent_state_key_and_cap(app_db: Path, monkeypatch) -> None:
    monkeypatch.delenv(settings.OPENAI_API_KEY_ENV, raising=False)
    conn = data.connect_read_only(app_db)
    try:
        blocked = data.agent_state(conn)
        assert blocked.can_call is False
        assert "not set" in blocked.blocked_reason
        assert blocked.cap_usd == settings.OPENAI_COST_CAP_USD
    finally:
        conn.close()

    # A fake value only tests presence; nothing is sent or logged.
    monkeypatch.setenv(settings.OPENAI_API_KEY_ENV, "fake-not-a-real-key")
    conn = data.connect_read_only(app_db)
    try:
        assert data.agent_state(conn).can_call is True
    finally:
        conn.close()

    writer = init_db(app_db)
    record_agent_call(
        writer,
        run_id=1,
        command="ask",
        model_name="gpt-5.4-mini",
        prompt_tokens=1,
        completion_tokens=1,
        estimated_cost_usd=float(settings.OPENAI_COST_CAP_USD),
        latency_s=0.1,
    )
    writer.close()
    conn = data.connect_read_only(app_db)
    try:
        capped = data.agent_state(conn)
        assert capped.can_call is False
        assert "cap reached" in capped.blocked_reason
    finally:
        conn.close()


def test_read_only_connection_rejects_writes(app_db: Path) -> None:
    conn = data.connect_read_only(app_db)
    try:
        import pytest

        with pytest.raises(Exception):
            conn.execute("DELETE FROM alerts")
    finally:
        conn.close()
