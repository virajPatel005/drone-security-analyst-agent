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
    frames_seen     INTEGER NOT NULL,
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
                len(track_events),  # approximate frames_seen
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