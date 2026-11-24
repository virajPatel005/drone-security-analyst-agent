"""Read-only CLI for the frame index: query by time window, class or track.

Examples:
    python scripts/query_frames.py --run 2 --class truck
    python scripts/query_frames.py --run 1 --from 2026-11-23T23:59:55 --to 2026-11-24T00:00:10
    python scripts/query_frames.py --run 2 --track 410
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import DB_PATH
from src.storage import (
    query_frames_by_class,
    query_frames_for_track,
    query_frames_in_time_window,
    query_objects_by_class,
)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only so a query cannot accidentally write to it."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _header(label: str, result: dict) -> None:
    print(label)
    print(
        f"total={result['total']} showing={result['showing']} "
        f"truncated={result['truncated']}"
    )
    if "notice" in result:
        print(result["notice"])


def _print_frames(result: dict) -> None:
    for row in result["rows"]:
        print(
            f"frame={row['frame_index']} timestamp={row['timestamp']} "
            f"zone={row['zone']} detections={row['detection_count']} "
            f"description={row['description']!r}"
        )


def _print_objects(result: dict) -> None:
    for row in result["rows"]:
        print(
            f"frame={row['frame_index']} track={row['track_id']} "
            f"class={row['class_name']} conf={row['confidence']:.3f} "
            f"bbox={row['bbox']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True, help="existing run id")
    parser.add_argument("--class", dest="class_name", help="filter by detection class")
    parser.add_argument("--track", type=int, help="all frames containing this track id")
    parser.add_argument("--from", dest="from_ts", help="inclusive start ISO timestamp")
    parser.add_argument("--to", dest="to_ts", help="inclusive end ISO timestamp")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args(argv)

    modes = [bool(args.class_name), args.track is not None, bool(args.from_ts or args.to_ts)]
    if sum(modes) != 1:
        parser.error("choose exactly one of --class, --track, or --from with --to")
    if bool(args.from_ts) != bool(args.to_ts):
        parser.error("--from and --to must be given together")

    conn = _connect(args.db)
    try:
        if args.class_name:
            frames = query_frames_by_class(
                conn, run_id=args.run, class_name=args.class_name, limit=args.limit
            )
            objects = query_objects_by_class(
                conn, run_id=args.run, class_name=args.class_name, limit=args.limit
            )
            _header(f"frames with class={args.class_name} run={args.run}", frames)
            _print_frames(frames)
            _header(f"detections of class={args.class_name} run={args.run}", objects)
            _print_objects(objects)
            track_ids = sorted(
                {row["track_id"] for row in objects["rows"] if row["track_id"] is not None}
            )
            print(f"distinct_track_ids={track_ids}")
        elif args.track is not None:
            result = query_frames_for_track(
                conn, run_id=args.run, track_id=args.track, limit=args.limit
            )
            _header(f"frames for track={args.track} run={args.run}", result)
            _print_frames(result)
        else:
            result = query_frames_in_time_window(
                conn, run_id=args.run, start=args.from_ts, end=args.to_ts, limit=args.limit
            )
            _header(f"frames in window run={args.run}", result)
            _print_frames(result)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
