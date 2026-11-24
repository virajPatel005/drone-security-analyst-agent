"""Read-only data layer for the Streamlit app.

Every function here wraps an existing ``src.storage`` query and returns a small
typed object, so the UI never handles sqlite rows and the layer can be tested
with a temporary database without importing Streamlit.  Connections are opened
in SQLite read-only mode: the app may only write through the agent commands,
which open their own connection to the same file.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import settings
from src.storage import (
    query_agent_cost_total,
    query_agent_traces,
    query_alerts,
    query_counts_by_event_type,
    query_descriptions,
    query_events_for_track,
    query_run_summary,
    query_tracks_by_class,
)

# The event model maps every road vehicle to "vehicle" and people to "person";
# counts must cover both, so the class list is kept here rather than in a page.
_TRACK_CLASSES = ("vehicle", "person")


@dataclass(frozen=True)
class RunInfo:
    id: int
    source_video: str
    start_time: str
    tuning: dict[str, Any] = field(default_factory=dict)

    @property
    def video_name(self) -> str:
        """Show the file name, not the absolute path recorded by the pipeline."""
        return Path(self.source_video).name


@dataclass(frozen=True)
class RunCounts:
    tracks: int
    events: int
    alerts: int


@dataclass(frozen=True)
class SummaryView:
    text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    created_time: str
    parsed: dict[str, Any] | None

    @property
    def questionable(self) -> list[Any]:
        """Track IDs the *stored* summary flagged, never a fresh model guess."""
        if not self.parsed:
            return []
        value = self.parsed.get("questionable", [])
        return value if isinstance(value, list) else []


@dataclass(frozen=True)
class AlertView:
    id: int
    run_id: int
    track_id: int
    rule_name: str
    severity: str
    timestamp: str
    reason: str
    zone: str | None


@dataclass(frozen=True)
class TrackView:
    track_id: int
    class_name: str
    first_seen: str
    last_seen: str
    event_count: int


@dataclass(frozen=True)
class EventView:
    timestamp: str
    event_type: str
    zone: str
    dwell_s: float
    class_name: str


@dataclass(frozen=True)
class DescriptionView:
    status: str
    description: str | None
    crop_path: str | None
    latency_ms: float | None
    model_name: str
    attributes: dict[str, Any] | None

    @property
    def has_crop(self) -> bool:
        return bool(self.crop_path and Path(self.crop_path).is_file())


@dataclass(frozen=True)
class Provenance:
    alert: AlertView
    track: TrackView | None
    events: list[EventView]
    description: DescriptionView | None
    tuning: dict[str, Any]
    flagged_questionable: bool


@dataclass(frozen=True)
class TraceView:
    id: int
    run_id: int
    mode: str
    question: str | None
    tool_calls: list[dict[str, Any]]
    answer: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    cached: bool
    created_time: str

    @property
    def tool_names(self) -> list[str]:
        return [call.get("name", "?") for call in self.tool_calls]


@dataclass(frozen=True)
class AgentState:
    key_present: bool
    running_total_usd: float
    cap_usd: float
    can_call: bool

    @property
    def blocked_reason(self) -> str:
        """Explain the disablement without ever exposing key material."""
        if not self.key_present:
            return f"OpenAI key not set ({settings.OPENAI_API_KEY_ENV})."
        if self.running_total_usd >= self.cap_usd:
            return f"Cost cap reached: ${self.running_total_usd:.6f} of ${self.cap_usd:.2f}."
        return ""


def connect_read_only(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database read-only, so UI browsing cannot mutate pipeline data."""
    path = Path(db_path) if db_path is not None else Path(settings.DB_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def list_runs(conn: sqlite3.Connection) -> list[RunInfo]:
    """Return all runs; the agent tool does the same SELECT, but returns JSON."""
    rows = conn.execute(
        "SELECT id, source_video, start_time, tuning_settings FROM runs ORDER BY id"
    ).fetchall()
    result: list[RunInfo] = []
    for run_id, source_video, start_time, tuning_json in rows:
        try:
            tuning = json.loads(tuning_json or "{}")
        except (TypeError, ValueError):
            tuning = {}
        result.append(
            RunInfo(
                id=int(run_id),
                source_video=str(source_video),
                start_time=str(start_time),
                tuning=tuning if isinstance(tuning, dict) else {},
            )
        )
    return result


def run_counts(conn: sqlite3.Connection, run_id: int) -> RunCounts:
    """Count tracks, events and alerts from the existing storage queries."""
    tracks = sum(
        len(query_tracks_by_class(conn, class_name=name, run_id=run_id))
        for name in _TRACK_CLASSES
    )
    events = sum(query_counts_by_event_type(conn, run_id=run_id).values())
    alerts = len(query_alerts(conn, run_id=run_id))
    return RunCounts(tracks=tracks, events=events, alerts=alerts)


def get_summary(conn: sqlite3.Connection, run_id: int) -> SummaryView | None:
    """Return the stored summary, rendering structured JSON without a new call."""
    row = query_run_summary(conn, run_id=run_id)
    if row is None:
        return None
    try:
        parsed = json.loads(row["text"])
        if not isinstance(parsed, dict):
            parsed = None
    except (TypeError, ValueError):
        # Older rows may hold plain prose; the page still renders it as text.
        parsed = None
    return SummaryView(
        text=str(row["text"]),
        model_name=str(row["model_name"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        estimated_cost_usd=float(row["estimated_cost_usd"]),
        created_time=str(row["created_time"]),
        parsed=parsed,
    )


def list_alerts(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    rule_name: str | None = None,
    severity: str | None = None,
) -> list[AlertView]:
    """Return alerts after UI filters; filtering stays in the data layer."""
    rows = query_alerts(conn, run_id=run_id, rule_name=rule_name)
    return [
        AlertView(
            id=int(row["id"]),
            run_id=int(row["run_id"]),
            track_id=int(row["track_id"]),
            rule_name=str(row["rule_name"]),
            severity=str(row["severity"]),
            timestamp=str(row["timestamp"]),
            reason=str(row["reason"]),
            zone=row["zone"],
        )
        for row in rows
        if severity is None or row["severity"] == severity
    ]


def _find_track(conn: sqlite3.Connection, run_id: int, track_id: int) -> TrackView | None:
    for name in _TRACK_CLASSES:
        for track in query_tracks_by_class(conn, class_name=name, run_id=run_id):
            if track.track_id == track_id:
                return TrackView(
                    track_id=int(track.track_id),
                    class_name=str(track.class_name),
                    first_seen=str(track.first_seen),
                    last_seen=str(track.last_seen),
                    # frames_seen is a historical misnomer: it is the event count.
                    event_count=len(query_events_for_track(conn, run_id, track_id)),
                )
    return None


def get_provenance(conn: sqlite3.Connection, alert: AlertView) -> Provenance:
    """Assemble the evidence behind one alert, read entirely from stored rows."""
    track = _find_track(conn, alert.run_id, alert.track_id)
    events = [
        EventView(
            timestamp=str(event.timestamp),
            event_type=str(event.event_type),
            zone=str(event.zone),
            dwell_s=float(event.dwell_s),
            class_name=str(event.class_name),
        )
        for event in query_events_for_track(conn, alert.run_id, alert.track_id)
    ]
    descriptions = query_descriptions(conn, run_id=alert.run_id, track_id=alert.track_id)
    description = None
    if descriptions:
        row = descriptions[0]
        description = DescriptionView(
            status=str(row["status"]),
            description=row["description"],
            crop_path=row["crop_path"],
            latency_ms=row["latency_ms"],
            model_name=str(row["model_name"]),
            attributes=row["attributes"],
        )
    run_row = conn.execute(
        "SELECT tuning_settings FROM runs WHERE id = ?", (alert.run_id,)
    ).fetchone()
    try:
        tuning = json.loads(run_row[0] or "{}") if run_row else {}
    except (TypeError, ValueError):
        tuning = {}
    summary = get_summary(conn, alert.run_id)
    return Provenance(
        alert=alert,
        track=track,
        events=events,
        description=description,
        tuning=tuning if isinstance(tuning, dict) else {},
        flagged_questionable=bool(summary and alert.track_id in summary.questionable),
    )


def list_traces(conn: sqlite3.Connection, run_id: int) -> list[TraceView]:
    """Return stored answers with tool calls already parsed into plain objects."""
    return [
        TraceView(
            id=int(row["id"]),
            run_id=int(row["run_id"]),
            mode=str(row["mode"]),
            question=row["question"],
            tool_calls=list(row["tool_calls"]),
            answer=str(row["answer"]),
            model_name=str(row["model_name"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            estimated_cost_usd=float(row["estimated_cost_usd"]),
            latency_ms=float(row["latency_ms"]),
            cached=bool(row["cached"]),
            created_time=str(row["created_time"]),
        )
        for row in query_agent_traces(conn, run_id=run_id)
    ]


def agent_state(conn: sqlite3.Connection) -> AgentState:
    """Combine key presence with the SQLite-backed cost guard for UI gating."""
    key_present = bool(os.getenv(settings.OPENAI_API_KEY_ENV, ""))
    total = query_agent_cost_total(conn)
    cap = float(settings.OPENAI_COST_CAP_USD)
    return AgentState(
        key_present=key_present,
        running_total_usd=total,
        cap_usd=cap,
        can_call=key_present and total < cap,
    )
