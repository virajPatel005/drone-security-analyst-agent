"""Run YOLO + ByteTrack over sampled frames and emit events plus a demo video."""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import (
    ANNOTATED_MAX_WIDTH,
    ANNOTATED_OUTPUT_DIR,
    CLASS_CATEGORIES,
    DB_PATH,
    DRONE_ALTITUDE_M,
    LOST_FRAMES,
    MIN_TRACK_FRAMES,
    PATROL_SECONDS_PER_LOCATION,
    PATROL_WAYPOINTS,
    SAMPLE_RATE_FPS,
    SIM_START_DATETIME,
    VIDEO_PATH,
)
from src.alerts import ALERT_RULES
from src.storage import init_db, store_run, store_tracks_and_events
from src.perception.annotated_video import AnnotatedVideoWriter, annotated_video_path
from src.perception.events import EventBuilder, TrackEvent
from src.perception.frame_sampler import probe_video
from src.perception.tracker import YoloByteTrackTracker
from src.telemetry.video_stream import video_frame_records


@dataclass
class _TrackSummary:
    track_id: int
    class_name: str
    first_timestamp: datetime
    last_timestamp: datetime
    frames_seen: int
    zones: set[str] = field(default_factory=set)


def _print_events(events: list[TrackEvent]) -> None:
    print("EVENTS")
    print(
        "timestamp | event_type | track_id | category | class | zone | conf | dwell_s | flags"
    )
    for event in events:
        dwell = f"{event.dwell_s:.2f}" if event.dwell_s is not None else "-"
        flags = "end_of_video" if event.end_of_video else ""
        print(
            f"{event.timestamp.isoformat(sep=' ', timespec='milliseconds')} | "
            f"{event.event_type:12s} | {event.track_id:8d} | {event.category:8s} | "
            f"{event.class_name:10s} | {event.zone:9s} | {event.confidence:.2f} | "
            f"{dwell:>7s} | {flags}"
        )


def _print_track_summary(
    summaries: dict[int, _TrackSummary],
    dwell_by_track: dict[int, float],
) -> None:
    print("TRACK SUMMARY")
    print(
        "track_id | class | first_seen | last_seen | frames_seen | zones | total_dwell_s"
    )
    for track_id in sorted(summaries):
        summary = summaries[track_id]
        print(
            f"{track_id:8d} | {summary.class_name:10s} | "
            f"{summary.first_timestamp.isoformat(sep=' ', timespec='milliseconds')} | "
            f"{summary.last_timestamp.isoformat(sep=' ', timespec='milliseconds')} | "
            f"{summary.frames_seen:11d} | {', '.join(sorted(summary.zones)):25s} | "
            f"{dwell_by_track.get(track_id, 0.0):.2f}"
        )


def main() -> None:
    metadata = probe_video(VIDEO_PATH)
    tracker = YoloByteTrackTracker()
    builder = EventBuilder(
        min_track_frames=MIN_TRACK_FRAMES,
        lost_frames=LOST_FRAMES,
        class_categories=CLASS_CATEGORIES,
    )
    output_path = annotated_video_path(VIDEO_PATH, ANNOTATED_OUTPUT_DIR)

    # --- storage setup -------------------------------------------------------
    db_path = DB_PATH
    # ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)

    # We'll store the run after the pipeline finishes; use SIM_START_DATETIME
    # as the run's start time and capture the tuning settings from the env.
    tuning_settings = {
        "yolo_model": os.getenv("YOLO_MODEL_NAME", "yolo11s.pt"),
        "yolo_conf": float(os.getenv("YOLO_CONF", "0.10")),
        "yolo_imgsz": int(os.getenv("YOLO_IMGSZ", "1280")),
        "sample_rate_fps": float(os.getenv("SAMPLE_RATE_FPS", "10.0")),
        "min_track_frames": MIN_TRACK_FRAMES,
        "lost_frames": LOST_FRAMES,
        "tracker_config": os.getenv("TRACKER_CONFIG", "configs/bytetrack_0.9.yaml"),
    }
    run_id = store_run(
        conn,
        source_video=str(VIDEO_PATH),
        start_time=SIM_START_DATETIME,
        tuning_settings=tuning_settings,
    )

    # -----------------------------------------------------------------------

    events: list[TrackEvent] = []
    summaries: dict[int, _TrackSummary] = {}
    tracking_seconds = 0.0
    sampled_frames = 0

    with AnnotatedVideoWriter(
        output_path,
        fps=SAMPLE_RATE_FPS,
        max_width=ANNOTATED_MAX_WIDTH,
    ) as writer:
        for record in video_frame_records(
            VIDEO_PATH,
            sample_rate_fps=SAMPLE_RATE_FPS,
            start_time=SIM_START_DATETIME,
            waypoints=PATROL_WAYPOINTS,
            seconds_per_location=PATROL_SECONDS_PER_LOCATION,
            altitude_m=DRONE_ALTITUDE_M,
        ):
            if record.image is None:
                raise RuntimeError("Video frame record unexpectedly has no image")

            started = time.perf_counter()
            observations = tracker.track(record.image)
            tracking_seconds += time.perf_counter() - started
            sampled_frames += 1

            events.extend(builder.process_frame(record, observations))
            writer.write(record.image, observations, record.timestamp, record.location)

            for observation in observations:
                summary = summaries.get(observation.track_id)
                if summary is None:
                    summaries[observation.track_id] = _TrackSummary(
                        track_id=observation.track_id,
                        class_name=observation.class_name,
                        first_timestamp=record.timestamp,
                        last_timestamp=record.timestamp,
                        frames_seen=1,
                        zones={record.location},
                    )
                else:
                    summary.class_name = observation.class_name
                    summary.last_timestamp = record.timestamp
                    summary.frames_seen += 1
                    summary.zones.add(record.location)

    events.extend(builder.finish())

    # ---- store tracks and events in one transaction -------------------------
    store_tracks_and_events(conn, run_id, events, tuning_settings=tuning_settings)

    # ---- derive alerts from the stored rows ----------------------------------
    # Alerts are computed after storage, never from live model output, so the
    # same rules can be re-run later from the database alone. Re-running is
    # safe: the alerts table has UNIQUE (run_id, track_id, rule_name) and
    # store_alert uses INSERT OR IGNORE, so counts never inflate.
    for rule_fn in ALERT_RULES.values():
        rule_fn(conn, run_id=run_id)

    alert_counts = dict(
        conn.execute(
            "SELECT rule_name, COUNT(*) FROM alerts WHERE run_id = ? GROUP BY rule_name",
            (run_id,),
        ).fetchall()
    )
    conn.close()

    # -----------------------------------------------------------------------
    dwell_by_track: dict[int, float] = defaultdict(float)
    for event in events:
        if event.dwell_s is not None:
            dwell_by_track[event.track_id] += event.dwell_s

    _print_events(events)
    print()
    _print_track_summary(summaries, dwell_by_track)
    print()
    print(f"sampled_frames: {sampled_frames}")
    print(f"total_seconds: {metadata.duration_s:.3f}")
    print(f"effective_fps: {sampled_frames / metadata.duration_s:.3f}")
    print(
        "mean_yolo_tracker_ms_per_frame: "
        f"{(tracking_seconds / sampled_frames * 1000) if sampled_frames else 0.0:.2f}"
    )
    print(f"device: {tracker.device}")
    print(f"annotated_video: {output_path.resolve()}")
    print(f"db: {db_path.resolve()}")
    print("alerts:")
    for rule_name in ALERT_RULES:
        print(f"  {rule_name}: {alert_counts.get(rule_name, 0)}")


if __name__ == "__main__":
    main()
