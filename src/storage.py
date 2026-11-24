"""SQLite storage module using only stdlib sqlite3.

Keeps all provenance (run ID, source video, tuning settings) and makes it
possible to query events by time window, by track, by class, and by event type.

Schema decisions are documented in docs/design-decisions.md under Step 4.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema (run once per database file)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_video    TEXT NOT NULL,
    start_time      TEXT NOT NULL,          -- ISO 8601 datetime when the run started
    tuning_settings TEXT NOT NULL DEFAULT '{}',  -- JSON string of the settings used
    created_time    TEXT NOT NULL           -- when the row was inserted (ISO 8601)
);

CREATE TABLE IF NOT EXISTS tracks (
    run_id          INTEGER NOT NULL,
    track_id        INTEGER NOT NULL,       -- ByteTrack-assigned ID
    class_name      TEXT NOT NULL,
    first_seen      TEXT NOT NULL,          -- ISO 8601 datetime
    last_seen       TEXT NOT NULL,          -- ISO 8601 datetime
    frames_seen     INTEGER NOT NULL,       -- number of stored events for this track, NOT detected frames
    max_confidence  REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (run_id, track_id)
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    track_id        INTEGER,              -- may be NULL for orphan events
    event_type      TEXT NOT NULL,        -- 'appeared' | 'entered_zone' | 'left'
    timestamp       TEXT NOT NULL,        -- ISO 8601 datetime (full date-time, essential for midnight-range queries)
    zone            TEXT,
    dwell_s         REAL NOT NULL DEFAULT 0.0,
    class_name      TEXT NOT NULL,
    end_of_video    INTEGER NOT NULL DEFAULT 0
);

-- Index so that "events between X and Y" is fast without a full scan.
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

-- Index so that "all events for one track" is fast.
CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);

-- Composite index so that "events of a given type within a time window" is fast.
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, timestamp);

-- Alerts table for rule-based detections
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    track_id        INTEGER NOT NULL,
    rule_name       TEXT    NOT NULL,
    severity        TEXT    NOT NULL,   -- 'warning' or 'info'
    timestamp       TEXT    NOT NULL,   -- ISO 8601 datetime
    reason          TEXT    NOT NULL,   -- plain-English reason
    zone            TEXT,               -- zone at time of alert
    UNIQUE (run_id, track_id, rule_name)   -- one alert per track per rule
);

-- Index so that "all alerts for one run" is fast.
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id);

-- Index so that "all alerts of a given rule" is fast.
CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alerts(rule_name);

-- VLM descriptions of confirmed tracks. One row per track per run, ever:
-- the UNIQUE constraint is what makes the "describe once" rule enforceable.
CREATE TABLE IF NOT EXISTS descriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    track_id        INTEGER NOT NULL,
    status          TEXT    NOT NULL,   -- 'described' | 'too_small' | 'error'
    description     TEXT,               -- sentence, or raw text when status='error'
    attributes      TEXT,               -- JSON: object_type, main_colour, ...
    crop_path       TEXT,               -- saved crop file; NULL when too_small
    orig_width      INTEGER NOT NULL,   -- original box width in source pixels
    orig_height     INTEGER NOT NULL,   -- original box height in source pixels
    model_name      TEXT NOT NULL,
    latency_ms      REAL,
    created_time    TEXT NOT NULL,
    UNIQUE (run_id, track_id)
);

-- Index so that "all descriptions for one run" is fast.
CREATE INDEX IF NOT EXISTS idx_descriptions_run ON descriptions(run_id);

-- Run summaries (one row per run, written once by the agent)
CREATE TABLE IF NOT EXISTS run_summaries (
    run_id          INTEGER PRIMARY KEY,
    text            TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    created_time    TEXT NOT NULL
);

-- Index so that "summary for one run" is fast.
CREATE INDEX IF NOT EXISTS idx_run_summaries_run ON run_summaries(run_id);

-- Every OpenAI chat call is recorded here. The running cost total must live in
-- SQLite, not in process memory, so a new command sees earlier spend.
CREATE TABLE IF NOT EXISTS agent_costs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER,           -- NULL for non-run calls (e.g. model listing)
    command             TEXT NOT NULL,     -- 'summary' | 'ask'
    model_name          TEXT NOT NULL,
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    estimated_cost_usd  REAL NOT NULL,
    latency_s           REAL NOT NULL,
    created_time        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_costs_created ON agent_costs(created_time);
"""

_CREATE_RUN_SQL = """
INSERT INTO runs (source_video, start_time, tuning_settings, created_time)
VALUES (?, ?, ?, ?);
"""

_STORE_TRACK_SQL = """
INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

_STORE_EVENT_SQL = """
INSERT INTO events (run_id, track_id, event_type, timestamp, zone, dwell_s, class_name, end_of_video)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

_GET_EVENTS_IN_WINDOW_SQL = """
SELECT id, run_id, track_id, event_type, timestamp, zone, dwell_s, class_name, end_of_video
FROM events
WHERE timestamp >= ? AND timestamp <= ?
{type_filter}
{class_filter}
ORDER BY timestamp ASC;
"""

_GET_EVENTS_FOR_TRACK_SQL = """
SELECT id, run_id, track_id, event_type, timestamp, zone, dwell_s, class_name, end_of_video
FROM events
WHERE run_id = ? AND track_id = ?
ORDER BY timestamp ASC;
"""

_GET_TRACKS_BY_CLASS_SQL = """
SELECT track_id, class_name, first_seen, last_seen, frames_seen, max_confidence
FROM tracks
WHERE class_name = ?
{run_filter}
ORDER BY track_id ASC;
"""

_GET_COUNTS_BY_TYPE_SQL = """
SELECT event_type, COUNT(*) AS cnt
FROM events
{run_where}
GROUP BY event_type
ORDER BY cnt DESC;
"""

# ---------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------__


def _iso(dt: Any) -> str:
    """Return an ISO 8601 plain string (no timezone) for SQLite storage.

    Accepts a ``datetime`` or an already‑formatted ISO string; the latter
    is returned as‑is so that test helpers that pass strings work without
    modification.
    """
    if isinstance(dt, str):
        # Already an ISO string — possibly with a trailing 'S' timespec;
        # normalise to seconds for consistency.
        return dt.split(".")[0]
    # datetime object — strip timezone info if present.
    return dt.replace(tzinfo=None).isoformat(timespec="seconds")


def _parse_iso(s: str) -> datetime:
    """Reverse of _iso — expects the format produced by .isoformat(seconds)."""
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Create all tables and indexes at the given ``db_path``.

    Idempotent — safe to call multiple times.
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Write path — store one run (tracks + events) in a single transaction
# -------------------------------------------------------------------------__


def store_run(
    conn: sqlite3.Connection,
    *,
    source_video: str,
    start_time: datetime,
    tuning_settings: dict[str, Any] | None = None,
) -> int:
    """Insert a new ``runs`` row and return its auto-generated ``id``.

    The ``start_time`` is the simulated start time of the pipeline (e.g.
    ``SIM_START_DATETIME`` from settings).  ``created_time`` is the insertion
    timestamp.
    """
    if tuning_settings is None:
        tuning_settings = {}
    cur = conn.execute(
        _CREATE_RUN_SQL,
        (
            source_video,
            _iso(start_time),
            json.dumps(tuning_settings, sort_keys=True),
            _iso(datetime.now(timezone.utc)),
        ),
    )
    conn.commit()
    return cur.lastrowid


def _max_confidence_per_track(events: list["TrackEvent"]) -> dict[int, float]:
    """Compute max confidence per track_id from a list of TrackEvent objects."""
    confs: dict[int, list[float]] = {}
    for ev in events:
        tid = ev.track_id
        if tid is None:
            continue
        confs.setdefault(tid, []).append(ev.confidence)
    return {tid: max(c) for tid, c in confs.items()}


def store_tracks_and_events(
    conn: sqlite3.Connection,
    run_id: int,
    events: list["TrackEvent"],
    tuning_settings: dict[str, Any] | None = None,
) -> None:
    """Store all tracks and events for one run in a single transaction.

    The *events* list is the full output from the event builder (appeared,
    entered_zone, left).  Tracks are derived from the events so that the
    ``tracks`` table is consistent without requiring the caller to compute
    max‑confidence separately.

    This function is designed to be called *once* per pipeline run, inside
    the caller's ``with conn:`` (or equivalent) transaction block.
    """
    if tuning_settings is None:
        tuning_settings = {}

    # Compute max_confidence per track from the events we are about to store.
    max_confs = _max_confidence_per_track(events)

    # Collect the set of track_ids that appear in the events, so we can
    # store one row per track.  We use the "appeared" event as the marker
    # for a new track entry.
    appeared_track_ids: set[int] = set()
    for ev in events:
        if ev.event_type == "appeared":
            appeared_track_ids.add(ev.track_id)

    # Store tracks for every track_id that received an "appeared" event.
    for tid in appeared_track_ids:
        # Grab the earliest and latest timestamps for this track across all events.
        track_events = [e for e in events if e.track_id == tid]
        first_seen = min(e.timestamp for e in track_events)
        last_seen = max(e.timestamp for e in track_events)

        conn.execute(
            _STORE_TRACK_SQL,
            (
                run_id,
                tid,
                track_events[0].class_name,  # class name from the first event
                _iso(first_seen),
                _iso(last_seen),
                # Misnamed historical field: this is the number of stored
                # events for the track, not the number of detected frames.
                len(track_events),
                max_confs.get(tid, 0.0),
            ),
        )

    # Store events
    for ev in events:
        conn.execute(
            _STORE_EVENT_SQL,
            (
                run_id,
                ev.track_id,
                ev.event_type,
                _iso(ev.timestamp),
                ev.zone if ev.zone else "",
                ev.dwell_s if ev.dwell_s is not None else 0.0,
                ev.class_name,
                1 if ev.end_of_video else 0,
            ),
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Read path — queries returning plain typed objects
# -------------------------------------------------------------------------__


class _EventRow:
    """Plain object returned by ``query_events_*`` functions.

    Keeps the same field names as ``TrackEvent`` so that callers can use
    the same access patterns without knowing they come from SQLite.
    """

    __slots__ = (
        "id",
        "run_id",
        "track_id",
        "event_type",
        "timestamp",
        "zone",
        "dwell_s",
        "class_name",
        "end_of_video",
        "confidence",
    )

    def __init__(
        self,
        id: int,
        run_id: int,
        track_id: int | None,
        event_type: str,
        timestamp: str,
        zone: str | None,
        dwell_s: float,
        class_name: str,
        end_of_video: int,
        confidence: float = 0.0,
    ) -> None:
        self.id = id
        self.run_id = run_id
        self.track_id = track_id
        self.event_type = event_type
        self.timestamp = timestamp  # ISO string; caller may .fromisoformat()
        self.zone = zone if zone else ""
        self.dwell_s = dwell_s
        self.class_name = class_name
        self.end_of_video = bool(end_of_video)
        self.confidence = confidence


class _TrackRow:
    __slots__ = ("track_id", "class_name", "first_seen", "last_seen", "frames_seen", "max_confidence")

    def __init__(
        self,
        track_id: int,
        class_name: str,
        first_seen: str,
        last_seen: str,
        frames_seen: int,
        max_confidence: float,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.frames_seen = frames_seen
        self.max_confidence = max_confidence


# ---------------------------------------------------------------------------
# Query: events in a time window (optionally filtered by type and class)
# -------------------------------------------------------------------------__


def query_events_in_time_window(
    conn: sqlite3.Connection,
    *,
    start: datetime,
    end: datetime,
    event_type: str | None = None,
    class_name: str | None = None,
    run_id: int | None = None,
) -> list[_EventRow]:
    """Return all events whose ``timestamp`` falls inside [start, end].

    Parameters
    ----------
    start, end : datetime
        Inclusive range.  Full ISO 8601 strings work, so a window that
        crosses midnight (e.g. 23:59:55 → 00:00:09) is a normal range query.
    event_type : str | None
        If given, only events matching this type are returned.
    class_name : str | None
        If given, only events matching this class are returned.
    run_id : int | None
        If given, only events from this run.

    Returns
    -------
    list[_EventRow]
        Rows with the same attribute names as ``TrackEvent``.
    """
    # Build WHERE clause parts and parameter list in order.
    wheres: list[str] = ["timestamp >= ? AND timestamp <= ?"]
    params: list[Any] = [_iso(start), _iso(end)]

    if event_type is not None:
        wheres.append("event_type = ?")
        params.append(event_type)

    if class_name is not None:
        wheres.append("class_name = ?")
        params.append(class_name)

    if run_id is not None:
        wheres.append("run_id = ?")
        params.append(run_id)

    where_clause = " AND ".join(wheres)

    sql = f"""
        SELECT id, run_id, track_id, event_type, timestamp, zone, dwell_s, class_name, end_of_video
        FROM events
        WHERE {where_clause}
        ORDER BY timestamp ASC;
    """

    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    return [
        _EventRow(
            id=r[0],
            run_id=r[1],
            track_id=r[2],
            event_type=r[3],
            timestamp=r[4],
            zone=r[5],
            dwell_s=r[6],
            class_name=r[7],
            end_of_video=r[8],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Query: all events for a given track (within one run)
# -------------------------------------------------------------------------__


def query_events_for_track(
    conn: sqlite3.Connection,
    run_id: int,
    track_id: int,
) -> list[_EventRow]:
    """Return every event associated with *track_id* in *run_id*, ordered by time."""

    cursor = conn.execute(
        _GET_EVENTS_FOR_TRACK_SQL,
        (run_id, track_id),
    )
    rows = cursor.fetchall()
    return [
        _EventRow(
            id=r[0],
            run_id=r[1],
            track_id=r[2],
            event_type=r[3],
            timestamp=r[4],
            zone=r[5],
            dwell_s=r[6],
            class_name=r[7],
            end_of_video=r[8],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Query: tracks by class (optionally filtered to one run)
# -------------------------------------------------------------------------__


def query_tracks_by_class(
    conn: sqlite3.Connection,
    *,
    class_name: str,
    run_id: int | None = None,
) -> list[_TrackRow]:
    """Return all tracks of a given *class_name*, optionally limited to *run_id*. """

    run_filter = ""
    params: list[Any] = [class_name]

    if run_id is not None:
        run_filter = " AND run_id = ? "
        params.append(run_id)

    sql = _GET_TRACKS_BY_CLASS_SQL.format(run_filter=run_filter)
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    return [
        _TrackRow(
            track_id=r[0],
            class_name=r[1],
            first_seen=r[2],
            last_seen=r[3],
            frames_seen=r[4],
            max_confidence=r[5],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Query: counts by event type (optionally filtered to one run)
# -------------------------------------------------------------------------__


def query_counts_by_event_type(
    conn: sqlite3.Connection,
    *,
    run_id: int | None = None,
) -> dict[str, int]:
    """Return a mapping ``event_type -> count`` for all events (or one run)."""

    run_where = ""
    params: list[Any] = []

    if run_id is not None:
        run_where = "WHERE run_id = ? "
        params.append(run_id)

    sql = _GET_COUNTS_BY_TYPE_SQL.format(run_where=run_where)
    cursor = conn.execute(sql, params)
    return {row[0]: row[1] for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# Alert storage writes — each function commits its own transaction
# ---------------------------------------------------------------------------


def store_alert(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    track_id: int,
    rule_name: str,
    severity: str,
    timestamp: str,
    reason: str,
    zone: str | None = None,
) -> int:
    """Store one alert row and return its auto-generated ``id``.

    The ``UNIQUE (run_id, track_id, rule_name)`` constraint ensures at most
    one alert per track per rule; duplicate inserts are silently ignored.
    Each caller is responsible for wrapping the call in its own transaction
    (``with conn:`` or explicit ``commit``) if they want atomic grouping.
    """
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO alerts (run_id, track_id, rule_name, severity, timestamp, reason, zone)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            track_id,
            rule_name,
            severity,
            timestamp,
            reason,
            zone if zone else "",
        ),
    )
    conn.commit()
    return cur.lastrowid


def query_alerts(
    conn: sqlite3.Connection,
    *,
    run_id: int | None = None,
    rule_name: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return alerts filtered by run, rule, and/or time window.

    The ``start``/``end`` window may cross midnight — the timestamp strings
    are stored as full ISO 8601 date‑times, so ordinary range comparisons work
    for windows like 23:59:55 → 00:00:09 without special handling.

    Parameters
    ----------
    run_id : int | None
    rule_name : str | None
    start : datetime | None
        Inclusive lower bound (ISO 8601 string or ``datetime``).
    end : datetime | None
        Inclusive upper bound (ISO 8601 string or ``datetime``).

    Returns
    -------
    list[dict]
        Each dict has the same keys as the alerts table columns.
    """
    wheres: list[str] = []
    params: list[Any] = []

    if run_id is not None:
        wheres.append("run_id = ?")
        params.append(run_id)

    if rule_name is not None:
        wheres.append("rule_name = ?")
        params.append(rule_name)

    if start is not None and end is not None:
        wheres.append("timestamp >= ? AND timestamp <= ?")
        params.extend([_iso(start), _iso(end)])

    where_clause = " AND ".join(wheres) if wheres else "1=1"

    sql = f"""
        SELECT id, run_id, track_id, rule_name, severity, timestamp, reason, zone
        FROM alerts
        WHERE {where_clause}
        ORDER BY timestamp ASC;
    """

    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "run_id": r[1],
            "track_id": r[2],
            "rule_name": r[3],
            "severity": r[4],
            "timestamp": r[5],
            "reason": r[6],
            "zone": r[7] if r[7] else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Description storage — each function commits its own transaction
# ---------------------------------------------------------------------------


def store_description(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    track_id: int,
    status: str,
    description: str | None,
    attributes: dict[str, Any] | None,
    crop_path: str | None,
    orig_width: int,
    orig_height: int,
    model_name: str,
    latency_ms: float | None,
) -> int:
    """Store one track description and return its row id.

    ``status`` is ``described``, ``too_small`` or ``error``. For ``error`` the
    raw model text (or exception text) is kept in ``description`` so a failure
    is never silently dropped. ``attributes`` is serialised as JSON.

    The ``UNIQUE (run_id, track_id)`` constraint plus ``INSERT OR IGNORE``
    means a track can never be described twice, even if the stage is retried.
    The commit is inside this function because each row is independently
    useful if a later call fails.
    """
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO descriptions (
            run_id, track_id, status, description, attributes, crop_path,
            orig_width, orig_height, model_name, latency_ms, created_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            track_id,
            status,
            description,
            json.dumps(attributes, sort_keys=True) if attributes is not None else None,
            crop_path,
            orig_width,
            orig_height,
            model_name,
            latency_ms,
            _iso(datetime.now(timezone.utc)),
        ),
    )
    conn.commit()
    return cur.lastrowid


def query_descriptions(
    conn: sqlite3.Connection,
    *,
    run_id: int | None = None,
    track_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return descriptions filtered by run and/or track, ordered by track."""
    wheres: list[str] = []
    params: list[Any] = []

    if run_id is not None:
        wheres.append("run_id = ?")
        params.append(run_id)
    if track_id is not None:
        wheres.append("track_id = ?")
        params.append(track_id)

    where_clause = " AND ".join(wheres) if wheres else "1=1"
    cursor = conn.execute(
        f"""
        SELECT id, run_id, track_id, status, description, attributes, crop_path,
               orig_width, orig_height, model_name, latency_ms, created_time
        FROM descriptions
        WHERE {where_clause}
        ORDER BY track_id ASC;
        """,
        params,
    )
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "run_id": r[1],
            "track_id": r[2],
            "status": r[3],
            "description": r[4],
            "attributes": json.loads(r[5]) if r[5] else None,
            "crop_path": r[6],
            "orig_width": r[7],
            "orig_height": r[8],
            "model_name": r[9],
            "latency_ms": r[10],
            "created_time": r[11],
        }
        for r in rows
    ]

def store_run_summary(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    text: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
) -> None:
    """Store one run summary; INSERT OR REPLACE so it can be written once."""
    cur = conn.execute(
        """
        INSERT OR REPLACE INTO run_summaries
            (run_id, text, model_name, prompt_tokens, completion_tokens,
             estimated_cost_usd, created_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            text,
            model_name,
            prompt_tokens,
            completion_tokens,
            estimated_cost_usd,
            _iso(datetime.now(timezone.utc)),
        ),
    )
    conn.commit()


def query_run_summary(
    conn: sqlite3.Connection,
    *,
    run_id: int,
) -> dict[str, Any] | None:
    """Return the stored summary for *run_id*, or ``None`` if none exists."""
    row = conn.execute(
        """
        SELECT run_id, text, model_name, prompt_tokens, completion_tokens,
               estimated_cost_usd, created_time
        FROM run_summaries
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "text": row[1],
        "model_name": row[2],
        "prompt_tokens": row[3],
        "completion_tokens": row[4],
        "estimated_cost_usd": row[5],
        "created_time": row[6],
    }


def record_agent_call(
    conn: sqlite3.Connection,
    *,
    run_id: int | None,
    command: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
    latency_s: float,
) -> None:
    """Append one OpenAI usage row; the sum of these rows is the cost guard state."""
    conn.execute(
        """
        INSERT INTO agent_costs
            (run_id, command, model_name, prompt_tokens, completion_tokens,
             estimated_cost_usd, latency_s, created_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            command,
            model_name,
            prompt_tokens,
            completion_tokens,
            estimated_cost_usd,
            latency_s,
            _iso(datetime.now(timezone.utc)),
        ),
    )
    conn.commit()


def query_agent_cost_total(conn: sqlite3.Connection) -> float:
    """Return the sum of all recorded OpenAI call costs, in USD."""
    row = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) FROM agent_costs"
    ).fetchone()
    return float(row[0])
