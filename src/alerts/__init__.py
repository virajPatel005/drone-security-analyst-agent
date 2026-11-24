"""Rule-based alerts for confirmed tracks, reading only from stored database rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlite3

from src.storage import _iso, store_alert


# ---------------------------------------------------------------------------
# Configuration (tunable from src/config or environment)
# ---------------------------------------------------------------------------

NIGHT_WINDOW_START = 0  # hour inclusive (default 00:00)
NIGHT_WINDOW_END = 5    # hour exclusive (default 05:00)

# Loitering threshold: seconds a confirmed track must be alive to flag.
# Chosen for ~15-20 s clips: 5 s catches a track present for ~1/3 of the clip.
LOITERING_THRESHOLD_S = 5

# Classes considered "vehicles" for the vehicle_at_night rule.
VEHICLE_CLASSES = ("truck", "bus", "car", "motorcycle")

# Classes considered "heavy vehicles" for heavy_vehicle_logged.
HEAVY_VEHICLE_CLASSES = ("truck", "bus")


def _is_night(timestamp: datetime) -> bool:
    """Return True if *timestamp* falls inside the night window.

    The default window is 00:00‑05:00 (hour inclusive, hour exclusive).
    We only check the hour component, so it works for timestamps on either
    side of midnight without any special-date logic.

    Windows where the start hour is greater than the end hour (for example
    22:00‑05:00) wrap around midnight, so the test is "at or after start OR
    before end" rather than a single between check.
    """
    h = timestamp.hour
    if NIGHT_WINDOW_START <= NIGHT_WINDOW_END:
        return NIGHT_WINDOW_START <= h < NIGHT_WINDOW_END
    return h >= NIGHT_WINDOW_START or h < NIGHT_WINDOW_END


def _window_label() -> str:
    """Human-readable night window used in alert reasons."""
    return f"{NIGHT_WINDOW_START:02d}:00-{NIGHT_WINDOW_END:02d}:00"


def _all_tracks(conn: sqlite3.Connection, run_id: int):
    """Yield ``(track_id, class_name, first_seen, last_seen)`` for one run."""
    cur = conn.execute(
        "SELECT track_id, class_name, first_seen, last_seen FROM tracks WHERE run_id = ?",
        (run_id,),
    )
    for row in cur.fetchall():
        yield {"track_id": row[0], "class_name": row[1], "first_seen": row[2], "last_seen": row[3]}


def _zone_at_time(conn: sqlite3.Connection, run_id: int, track_id: int, moment: datetime) -> str | None:
    """Return the zone the track was in nearest to *moment*.

    Looks at ``entered_zone`` events for the track and returns the zone that
    was active at the given moment.  If no event covers *moment*, returns ``None``.
    """
    cur = conn.execute(
        """
        SELECT timestamp, zone FROM events
        WHERE run_id = ? AND track_id = ? AND event_type = 'entered_zone'
        ORDER BY timestamp ASC
        """,
        (run_id, track_id),
    )
    rows = cur.fetchall()

    if not rows:
        return None

    selected_zone = None
    for ts_str, zone in rows:
        ts = datetime.fromisoformat(ts_str)
        if ts <= moment:
            selected_zone = zone
        else:
            break  # rows are ASC sorted

    return selected_zone


def check_vehicle_at_night(
    conn,
    *,
    run_id: int,
) -> list[dict[str, Any]]:
    """Return ``vehicle_at_night`` alerts for *run_id*.

    Fires once per confirmed track whose ``first_seen`` hour is within the
    night window (default 00:00‑05:00).  Unconfirmed tracks are ignored
    (they never appear in the ``tracks`` table).
    """
    alerts: list[dict[str, Any]] = []

    for tr in _all_tracks(conn, run_id):
        if tr["class_name"] not in VEHICLE_CLASSES:
            continue
        first_seen = datetime.fromisoformat(tr["first_seen"])
        if _is_night(first_seen):
            reason = (
                f"vehicle track {tr['track_id']} first seen at {tr['first_seen']} "
                f"in night window ({_window_label()})"
            )
            store_alert(
                conn,
                run_id=run_id,
                track_id=tr["track_id"],
                rule_name="vehicle_at_night",
                severity="warning",
                timestamp=tr["first_seen"],
                reason=reason,
                zone=None,
            )
            alerts.append(
                {
                    "rule_name": "vehicle_at_night",
                    "severity": "warning",
                    "timestamp": tr["first_seen"],
                    "reason": reason,
                    "zone": None,
                    "track_id": tr["track_id"],
                }
            )

    return alerts


def check_person_at_night(
    conn,
    *,
    run_id: int,
) -> list[dict[str, Any]]:
    """Return ``person_at_night`` alerts for *run_id*.

    Fires once per confirmed person track whose ``first_seen`` hour is within
    the night window (default 00:00‑05:00).
    """
    alerts: list[dict[str, Any]] = []

    for tr in _all_tracks(conn, run_id):
        if tr["class_name"] != "person":
            continue
        first_seen = datetime.fromisoformat(tr["first_seen"])
        if _is_night(first_seen):
            reason = (
                f"person track {tr['track_id']} first seen at {tr['first_seen']} "
                f"in night window ({_window_label()})"
            )
            store_alert(
                conn,
                run_id=run_id,
                track_id=tr["track_id"],
                rule_name="person_at_night",
                severity="warning",
                timestamp=tr["first_seen"],
                reason=reason,
                zone=None,
            )
            alerts.append(
                {
                    "rule_name": "person_at_night",
                    "severity": "warning",
                    "timestamp": tr["first_seen"],
                    "reason": reason,
                    "zone": None,
                    "track_id": tr["track_id"],
                }
            )

    return alerts


def check_loitering(
    conn,
    *,
    run_id: int,
) -> list[dict[str, Any]]:
    """Return ``loitering`` alerts for *run_id*.

    Fires once per confirmed track whose ``last_seen - first_seen`` exceeds
    ``LOITERING_THRESHOLD_S`` seconds.  Measure from the track's own first/last
    seen timestamps, not from zone dwell.

    Only one alert per track, even if the track stays up for much longer.
    """
    alerts: list[dict[str, Any]] = []

    for tr in _all_tracks(conn, run_id):
        first_seen = datetime.fromisoformat(tr["first_seen"])
        last_seen = datetime.fromisoformat(tr["last_seen"])
        duration_s = (last_seen - first_seen).total_seconds()

        if duration_s > LOITERING_THRESHOLD_S:
            reason = (
                f"track {tr['track_id']} loitered for {duration_s:.1f}s "
                f"(first_seen={tr['first_seen']}, last_seen={tr['last_seen']})"
            )
            store_alert(
                conn,
                run_id=run_id,
                track_id=tr["track_id"],
                rule_name="loitering",
                severity="warning",
                timestamp=last_seen.isoformat(timespec="seconds"),
                reason=reason,
                zone=None,
            )
            alerts.append(
                {
                    "rule_name": "loitering",
                    "severity": "warning",
                    "timestamp": last_seen.isoformat(timespec="seconds"),
                    "reason": reason,
                    "zone": None,
                    "track_id": tr["track_id"],
                }
            )

    return alerts


def check_heavy_vehicle_logged(
    conn,
    *,
    run_id: int,
) -> list[dict[str, Any]]:
    """Return ``heavy_vehicle_logged`` alerts for *run_id*.

    Fires once per track classed as truck or bus, any hour.  Severity is
    ``info`` (not warning).
    """
    alerts: list[dict[str, Any]] = []

    for tr in _all_tracks(conn, run_id):
        if tr["class_name"] not in HEAVY_VEHICLE_CLASSES:
            continue
        reason = (
            f"track {tr['track_id']} classified as {tr['class_name']} "
            f"(truck/bus logged)"
        )
        store_alert(
            conn,
            run_id=run_id,
            track_id=tr["track_id"],
            rule_name="heavy_vehicle_logged",
            severity="info",
            timestamp=tr["last_seen"],
            reason=reason,
            zone=None,
        )
        alerts.append(
            {
                "rule_name": "heavy_vehicle_logged",
                "severity": "info",
                "timestamp": tr["last_seen"],
                "reason": reason,
                "zone": None,
                "track_id": tr["track_id"],
            }
        )

    return alerts


# Ordered registry used by the pipeline so every run evaluates every rule.
# Values are callables with the signature ``(conn, *, run_id)``.
ALERT_RULES = {
    "vehicle_at_night": check_vehicle_at_night,
    "person_at_night": check_person_at_night,
    "loitering": check_loitering,
    "heavy_vehicle_logged": check_heavy_vehicle_logged,
}