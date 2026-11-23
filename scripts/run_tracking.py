"""Run YOLO + ByteTrack over sampled frames and emit events plus a demo video."""

from __future__ import annotations

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
    DRONE_ALTITUDE_M,
    LOST_FRAMES,
    MIN_TRACK_FRAMES,
    PATROL_SECONDS_PER_LOCATION,
    PATROL_WAYPOINTS,
    SAMPLE_RATE_FPS,
    SIM_START_DATETIME,
    VIDEO_PATH,
)
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


if __name__ == "__main__":
    main()
