"""Run YOLO + ByteTrack over sampled frames and emit events plus a demo video."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Only the two CLI defaults are loaded here. YOLO/OpenCV/model modules are
# imported inside main() *after* argument parsing, so --help and a refused
# duplicate never pay their import cost or instantiate a model.
from src.config.settings import DB_PATH, VIDEO_PATH  # noqa: E402


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


def normalise_video_path(video: Path | str) -> str:
    """Return the string form written to ``runs.source_video`` by ``store_run``.

    Both the stored value and the CLI check go through this one function, so a
    relative path is compared in exactly the form the pipeline would persist.
    """
    return str(Path(video))


def existing_run_ids(
    db_path: Path,
    source_video: str,
    *,
    force: bool = False,
) -> list[int]:
    """Return ids of runs for *source_video* that must block a new run.

    The check opens the target database read-only: a missing file or a missing
    ``runs`` table means there is nothing to compare against, and the check can
    never create or modify a database. ``--force`` skips the read entirely.
    """
    if force:
        return []
    path = Path(db_path).resolve()
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return []
    try:
        rows = conn.execute(
            "SELECT id FROM runs WHERE source_video = ? ORDER BY id",
            (source_video,),
        ).fetchall()
        return [int(row[0]) for row in rows]
    except sqlite3.OperationalError:
        # A file without the runs table cannot hold a matching source_video.
        return []
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the demo CLI; the defaults reproduce the old no-argument run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO + ByteTrack over sampled frames and emit events plus a demo video."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help=f"video to process (default: {VIDEO_PATH})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"database to write to (default: {DB_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow reprocessing a video already present in the database",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    video_path = Path(args.video)
    db_path = Path(args.db)
    source_video = normalise_video_path(video_path)

    # Refuse duplicates before touching a model, a video or the database.
    blocking_ids = existing_run_ids(db_path, source_video, force=args.force)
    if blocking_ids:
        ids = ", ".join(str(run_id) for run_id in blocking_ids)
        print(
            f"Video already processed in {db_path.resolve()} "
            f"(source_video={source_video}); existing run id(s): {ids}. "
            "Use --force to reprocess it."
        )
        raise SystemExit(2)

    # Deferred until after parsing/the gate: importing ultralytics and OpenCV
    # takes ~2 s and must not happen for --help or a refused duplicate.
    from src.alerts import ALERT_RULES
    from src.config.settings import (
        ANNOTATED_MAX_WIDTH,
        ANNOTATED_OUTPUT_DIR,
        CLASS_CATEGORIES,
        DRONE_ALTITUDE_M,
        LOST_FRAMES,
        MIN_TRACK_FRAMES,
        PATROL_SECONDS_PER_LOCATION,
        PATROL_WAYPOINTS,
        SAMPLE_RATE_FPS,
        SIM_START_DATETIME,
        VLM_CROP_MIN_PAD_PX,
        VLM_CROP_MIN_SHORT_SIDE_PX,
        VLM_CROP_PAD_RATIO,
        VLM_CROP_TARGET_SHORT_SIDE_PX,
        VLM_CROPS_DIR,
        VLM_ENABLED,
    )
    from src.description import (
        CropRecord,
        LmStudioVlmClient,
        describe_run,
        save_track_crop,
    )
    from src.indexing import FrameObjectRow, FrameRow, build_frame_index
    from src.storage import init_db, store_frame_index, store_run, store_tracks_and_events
    from src.perception.annotated_video import AnnotatedVideoWriter, annotated_video_path
    from src.perception.events import EventBuilder, TrackEvent
    from src.perception.frame_sampler import probe_video
    from src.perception.tracker import YoloByteTrackTracker
    from src.telemetry.video_stream import video_frame_records

    metadata = probe_video(video_path)
    tracker = YoloByteTrackTracker()
    builder = EventBuilder(
        min_track_frames=MIN_TRACK_FRAMES,
        lost_frames=LOST_FRAMES,
        class_categories=CLASS_CATEGORIES,
    )
    output_path = annotated_video_path(video_path, ANNOTATED_OUTPUT_DIR)

    # --- storage setup -------------------------------------------------------
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
        source_video=source_video,
        start_time=SIM_START_DATETIME,
        tuning_settings=tuning_settings,
    )

    # -----------------------------------------------------------------------

    events: list[TrackEvent] = []
    summaries: dict[int, _TrackSummary] = {}
    # The frame index is collected here, inside the tracking loop, so every
    # future run is indexed by time and object without a separate backfill.
    frame_rows: list[FrameRow] = []
    frame_object_rows: list[FrameObjectRow] = []
    tracking_seconds = 0.0
    sampled_frames = 0

    # Crops are written during tracking (cheap OpenCV work) and described
    # afterwards, so VLM latency never blocks the frame loop.
    crops_dir = VLM_CROPS_DIR / f"run_{run_id:03d}"
    crop_records: list[CropRecord] = []
    cropped_tracks: set[int] = set()
    if VLM_ENABLED:
        crops_dir.mkdir(parents=True, exist_ok=True)

    with AnnotatedVideoWriter(
        output_path,
        fps=SAMPLE_RATE_FPS,
        max_width=ANNOTATED_MAX_WIDTH,
    ) as writer:
        for record in video_frame_records(
            video_path,
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

            frame, frame_objects = build_frame_index(record, observations)
            frame_rows.append(frame)
            frame_object_rows.extend(frame_objects)

            new_events = builder.process_frame(record, observations)
            events.extend(new_events)

            if VLM_ENABLED:
                for event in new_events:
                    # appeared is emitted on the frame where the track becomes
                    # confirmed, so this is the requested crop frame. Only the
                    # first appearance of a track ID is described; a reused ID
                    # later in the clip must not create a second row.
                    if event.event_type != "appeared":
                        continue
                    if event.track_id in cropped_tracks:
                        continue
                    cropped_tracks.add(event.track_id)

                    crop_path = crops_dir / f"track_{event.track_id:06d}.jpg"
                    too_small, orig_w, orig_h = save_track_crop(
                        record.image,
                        event.bbox,
                        crop_path,
                        pad_ratio=VLM_CROP_PAD_RATIO,
                        min_pad_px=VLM_CROP_MIN_PAD_PX,
                        min_short_side_px=VLM_CROP_MIN_SHORT_SIDE_PX,
                        target_short_side_px=VLM_CROP_TARGET_SHORT_SIDE_PX,
                    )
                    crop_records.append(
                        CropRecord(
                            track_id=event.track_id,
                            class_name=event.class_name,
                            crop_path=None if too_small else crop_path,
                            orig_width=orig_w,
                            orig_height=orig_h,
                            too_small=too_small,
                        )
                    )

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

    # ---- store the frame index for this run ---------------------------------
    # A single replace-write makes re-indexing idempotent; only the two new
    # frame tables are affected, so descriptions/costs/traces stay untouched.
    store_frame_index(conn, run_id, frame_rows, frame_object_rows)

    # ---- describe confirmed tracks from the saved crops ---------------------
    description_summary = None
    if VLM_ENABLED:
        description_summary = describe_run(
            conn,
            run_id=run_id,
            crops=crop_records,
            client=LmStudioVlmClient(),
        )

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
    print(
        "frame_index: "
        f"frames={len(frame_rows)} detections={len(frame_object_rows)}"
    )
    print(f"total_seconds: {metadata.duration_s:.3f}")
    print(f"effective_fps: {sampled_frames / metadata.duration_s:.3f}")
    print(
        "mean_yolo_tracker_ms_per_frame: "
        f"{(tracking_seconds / sampled_frames * 1000) if sampled_frames else 0.0:.2f}"
    )
    print(f"device: {tracker.device}")
    print(f"run_id: {run_id}")
    print(f"annotated_video: {output_path.resolve()}")
    print(f"crops: {crops_dir.resolve()}")
    print(f"db: {db_path.resolve()}")
    if description_summary is None:
        print("vlm: disabled")
    else:
        print("vlm:")
        print(f"  described: {description_summary.described}")
        print(f"  too_small: {description_summary.too_small}")
        print(f"  error: {description_summary.error}")
        print(f"  total_seconds: {description_summary.total_latency_s:.2f}")
    print("alerts:")
    for rule_name in ALERT_RULES:
        print(f"  {rule_name}: {alert_counts.get(rule_name, 0)}")


if __name__ == "__main__":
    main()
