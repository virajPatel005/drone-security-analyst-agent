"""Backfill the frame index for an EXISTING run using that run's stored settings.

This script deliberately:
  * never inserts a row into ``runs`` (the run_id must already exist);
  * never deletes or rebuilds ``data/demo.db`` — it opens the same file with
    ``init_db``, whose CREATE TABLE IF NOT EXISTS only adds the new tables;
  * never calls the VLM or OpenAI; it re-runs sampling, YOLO and ByteTrack only;
  * writes only the ``frames`` and ``frame_objects`` tables, replacing any
    previous index for that run so a retry cannot duplicate rows.

After writing it checks that no other table's row count (or the stored OpenAI
cost total) changed, then verifies that every confirmed track ID in the tracks
table is present in the frame index. Extra frame-only IDs are reported as
unconfirmed detections; a missing confirmed track fails the check. Nothing is
repaired or filtered to force a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import (
    DB_PATH,
    DRONE_ALTITUDE_M,
    PATROL_SECONDS_PER_LOCATION,
    PATROL_WAYPOINTS,
    YOLO_DEVICE,
)
from src.indexing import FrameObjectRow, FrameRow, build_frame_index
from src.perception.tracker import YoloByteTrackTracker
from src.storage import init_db, store_frame_index
from src.telemetry.video_stream import video_frame_records

_PROTECTED_TABLES = (
    "runs",
    "tracks",
    "events",
    "alerts",
    "descriptions",
    "run_summaries",
    "agent_costs",
    "agent_traces",
)


def _protected_snapshot(conn) -> dict[str, float | int]:
    """Counts of every table a backfill must not disturb, plus the cost ledger."""
    snapshot: dict[str, float | int] = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _PROTECTED_TABLES
    }
    snapshot["agent_costs_total_usd"] = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) FROM agent_costs"
    ).fetchone()[0]
    return snapshot


def _compare_track_ids(conn, run_id: int) -> bool:
    """Return True when every confirmed track ID appears in the frame index.

    The check is deliberately one-way. Every ID in ``tracks`` is confirmed (it
    emitted an ``appeared`` event) and therefore MUST have been seen in a stored
    detection. The reverse is not true: ByteTrack also emits short-lived IDs
    that never get confirmed, so the frame index legitimately holds extra IDs.
    Those are reported as unconfirmed detections rather than silently dropped.
    """
    frame_ids = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT track_id FROM frame_objects "
            "WHERE run_id = ? AND track_id IS NOT NULL",
            (run_id,),
        )
    }
    stored_ids = {
        row[0]
        for row in conn.execute(
            "SELECT track_id FROM tracks WHERE run_id = ?", (run_id,)
        )
    }
    missing = stored_ids - frame_ids
    extras = frame_ids - stored_ids

    if missing:
        print(f"track_id_check: FAIL run_id={run_id}", file=sys.stderr)
        print(f"  missing_confirmed_tracks={sorted(missing)}", file=sys.stderr)
        print(
            "  A confirmed track has no stored detection; refusing to accept this index.",
            file=sys.stderr,
        )
        return False

    print(
        f"track_id_check: PASS run_id={run_id} "
        f"confirmed_tracks_present={len(stored_ids)} frame_ids={len(frame_ids)}"
    )
    print(f"  unconfirmed_detections: count={len(extras)} ids={sorted(extras)}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)

    conn = init_db(DB_PATH)
    try:
        row = conn.execute(
            "SELECT source_video, start_time, tuning_settings FROM runs WHERE id = ?",
            (args.run_id,),
        ).fetchone()
        if row is None:
            raise SystemExit(
                f"No run with id {args.run_id}; refusing to create a new run."
            )

        source_video, start_time_text, settings_text = row
        settings = json.loads(settings_text)
        before = _protected_snapshot(conn)

        # The tracker is built from the run's own stored tuning, never from the
        # current shell/.env values, so the backfill reproduces the same IDs.
        tracker = YoloByteTrackTracker(
            model_name=settings["yolo_model"],
            imgsz=int(settings["yolo_imgsz"]),
            conf=float(settings["yolo_conf"]),
            device=YOLO_DEVICE,
            tracker_config=settings["tracker_config"],
        )

        frames: list[FrameRow] = []
        objects: list[FrameObjectRow] = []
        for record in video_frame_records(
            Path(source_video),
            sample_rate_fps=float(settings["sample_rate_fps"]),
            start_time=datetime.fromisoformat(start_time_text),
            waypoints=PATROL_WAYPOINTS,
            seconds_per_location=PATROL_SECONDS_PER_LOCATION,
            altitude_m=DRONE_ALTITUDE_M,
        ):
            observations = tracker.track(record.image)
            frame, frame_objects = build_frame_index(record, observations)
            frames.append(frame)
            objects.extend(frame_objects)

        store_frame_index(conn, args.run_id, frames, objects)

        after = _protected_snapshot(conn)
        if before != after:
            raise RuntimeError(
                "Backfill changed protected data: "
                f"before={before} after={after}"
            )
        print(f"protected_data_check: PASS run_id={args.run_id}")

        if not _compare_track_ids(conn, args.run_id):
            return 2

        print(
            f"run_id={args.run_id} frames={len(frames)} "
            f"objects={len(objects)} device={tracker.device}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
