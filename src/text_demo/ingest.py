"""Turn a parsed scenario into the existing tracks/events/frame-index rows.

The point of this module is reuse: it emits the same ``TrackEvent`` and index
row types as the video pipeline, then calls the same storage functions and the
same ``ALERT_RULES``. There is no text-specific alert engine.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.alerts import ALERT_RULES
from src.indexing import FrameObjectRow, FrameRow
from src.perception.events import TrackEvent
from src.perception.observations import BBox
from src.storage import store_frame_index, store_run, store_tracks_and_events
from src.text_demo.scenario import Scenario

SOURCE_PREFIX = "text-simulated:"
# Text frames have no image, so detections carry a placeholder box. The index
# stores it for schema compatibility; no crop or VLM call is ever made.
_ZERO_BBOX: BBox = (0.0, 0.0, 0.0, 0.0)
_CONFIDENCE = 1.0  # a hand-written label is a certainty, not a model score


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Counts read back after a successful, idempotent ingestion."""

    run_id: int
    source: str
    frames: int
    objects: int
    tracks: int
    events: int
    alerts: dict[str, int]


def source_name(scenario_name: str) -> str:
    """Mark runs clearly as simulated so they are never mistaken for video."""
    return f"{SOURCE_PREFIX}{scenario_name}"


def _category(class_name: str) -> str:
    return "person" if class_name == "person" else "vehicle"


def _track_ids(scenario: Scenario) -> dict[str, int]:
    """Assign integer track IDs by first appearance, so a label maps to one ID."""
    ids: dict[str, int] = {}
    for frame in scenario.frames:
        for obj in frame.objects:
            if obj.label not in ids:
                ids[obj.label] = len(ids) + 1
    return ids


def build_events(
    scenario: Scenario,
    track_ids: dict[str, int],
    *,
    visit_gap_s: float,
) -> list[TrackEvent]:
    """Build appeared / entered_zone / left events for every labelled object.

    A sighting separated from the previous one by more than ``visit_gap_s`` is
    a new *visit*: it emits a second ``appeared`` for the same track ID. That
    is how the truck scenario logs two appearances while remaining one track.
    """
    if visit_gap_s < 0:
        raise ValueError("visit_gap_s must not be negative")

    by_label: dict[str, list[tuple[int, object, object]]] = defaultdict(list)
    for frame_index, frame in enumerate(scenario.frames, 1):
        for obj in frame.objects:
            by_label[obj.label].append((frame_index, frame, obj))

    events: list[TrackEvent] = []
    for label in sorted(by_label, key=track_ids.get):
        track_id = track_ids[label]
        sightings = by_label[label]
        visits: list[list[tuple[int, object, object]]] = []
        for sighting in sightings:
            previous = visits[-1][-1] if visits else None
            if previous is None or (
                sighting[1].timestamp - previous[1].timestamp
            ).total_seconds() > visit_gap_s:
                visits.append([sighting])
            else:
                visits[-1].append(sighting)

        for visit_index, visit in enumerate(visits):
            first = visit[0]
            first_ts = first[1].timestamp
            zone = first[1].zone
            class_name = first[2].class_name
            category = _category(class_name)

            events.append(
                TrackEvent("appeared", track_id, category, class_name, first_ts,
                           zone, _ZERO_BBOX, _CONFIDENCE)
            )
            events.append(
                TrackEvent("entered_zone", track_id, category, class_name, first_ts,
                           zone, _ZERO_BBOX, _CONFIDENCE, dwell_s=0.0)
            )

            zone_entered_at = first_ts
            current_zone = zone
            for _frame_index, frame, _obj in visit[1:]:
                if frame.zone != current_zone:
                    dwell_s = (frame.timestamp - zone_entered_at).total_seconds()
                    events.append(
                        TrackEvent("entered_zone", track_id, category, class_name,
                                   frame.timestamp, frame.zone, _ZERO_BBOX,
                                   _CONFIDENCE, dwell_s=dwell_s)
                    )
                    current_zone = frame.zone
                    zone_entered_at = frame.timestamp

            last_ts = visit[-1][1].timestamp
            events.append(
                TrackEvent("left", track_id, category, class_name, last_ts,
                           current_zone, _ZERO_BBOX, _CONFIDENCE,
                           dwell_s=(last_ts - zone_entered_at).total_seconds(),
                           end_of_video=(visit_index == len(visits) - 1))
            )

    # appeared before entered_zone before left when timestamps coincide.
    events.sort(key=lambda e: (e.timestamp, e.event_type != "appeared", e.event_type == "left"))
    return events


def ingest_scenario(
    conn: sqlite3.Connection,
    scenario: Scenario,
    *,
    visit_gap_s: float,
) -> IngestResult:
    """Store one scenario and evaluate the existing rules on the stored rows.

    Idempotent per scenario name: the run row is reused and its derived rows
    are replaced, so a second ingestion cannot duplicate a run or any row.
    """
    source = source_name(scenario.name)
    tuning = {
        "source": "text-simulated",
        "scenario": scenario.name,
        "visit_gap_s": visit_gap_s,
    }
    existing = conn.execute(
        "SELECT id FROM runs WHERE source_video = ? ORDER BY id LIMIT 1", (source,)
    ).fetchone()
    if existing is None:
        run_id = store_run(
            conn,
            source_video=source,
            start_time=scenario.frames[0].timestamp,
            tuning_settings=tuning,
        )
    else:
        run_id = int(existing[0])
        # store_tracks_and_events and store_alert only INSERT, so clear this
        # run's derived rows first. store_frame_index does its own replace.
        conn.execute("DELETE FROM alerts WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM tracks WHERE run_id = ?", (run_id,))
        conn.commit()

    track_ids = _track_ids(scenario)
    frame_rows = [
        FrameRow(
            frame_index=index,
            timestamp=frame.timestamp,
            zone=frame.zone,
            detection_count=len(frame.objects),
            description=frame.description,
        )
        for index, frame in enumerate(scenario.frames, 1)
    ]
    object_rows = [
        FrameObjectRow(
            frame_index=index,
            track_id=track_ids[obj.label],
            class_name=obj.class_name,
            confidence=_CONFIDENCE,
            bbox=_ZERO_BBOX,
        )
        for index, frame in enumerate(scenario.frames, 1)
        for obj in frame.objects
    ]
    events = build_events(scenario, track_ids, visit_gap_s=visit_gap_s)

    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning)
    store_frame_index(conn, run_id, frame_rows, object_rows)
    # The rules are deliberately unchanged: they only read stored DB rows.
    for rule_fn in ALERT_RULES.values():
        rule_fn(conn, run_id=run_id)

    alert_counts = dict(
        conn.execute(
            "SELECT rule_name, COUNT(*) FROM alerts WHERE run_id = ? GROUP BY rule_name",
            (run_id,),
        ).fetchall()
    )
    return IngestResult(
        run_id=run_id,
        source=source,
        frames=len(frame_rows),
        objects=len(object_rows),
        tracks=len(track_ids),
        events=len(events),
        alerts=alert_counts,
    )
