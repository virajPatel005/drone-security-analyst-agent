"""Step 10 scalability measurements. Read-only on videos and demo.db.

Every write goes to a scratch database under /tmp/opencode/step10; the real
``data/demo.db`` is opened only through a ``mode=ro`` URI. No OpenAI call and
no model download/install happens here.

Run from the repo root:
    .venv/bin/python scripts/benchmark.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config.settings import (
    CLASS_CATEGORIES,
    DB_PATH,
    DRONE_ALTITUDE_M,
    LOST_FRAMES,
    MIN_TRACK_FRAMES,
    PATROL_SECONDS_PER_LOCATION,
    PATROL_WAYPOINTS,
    SAMPLE_RATE_FPS,
    SIM_START_DATETIME,
    VIDEOS_DIR,
    YOLO_DEVICE,
)
from src.indexing import FrameObjectRow, FrameRow, build_frame_index
from src.perception.events import EventBuilder
from src.perception.frame_sampler import probe_video, sample_video
from src.perception.tracker import YoloByteTrackTracker
from src.storage import (
    init_db,
    query_frames_by_class,
    query_frames_for_track,
    query_frames_in_time_window,
    store_frame_index,
    store_tracks_and_events,
)
from src.telemetry.patrol import telemetry_for_video_time
from src.telemetry.records import FrameRecord

SCRATCH = Path("/tmp/opencode/step10")
DEMO_DB = DB_PATH
CLIPS = ("drone_intersection.mp4", "drone_truck.mp4")


# ---------------------------------------------------------------------------
# Per-stage timing on the real clips
# ---------------------------------------------------------------------------


def _measured_samples(gen, decode_times: list[float]):
    """Wrap sample_video so the time to obtain each frame is isolated.

    The seek and 4K/1080p decode happen inside ``next(gen)``; measuring that
    call is the only way to separate decode from YOLO+ByteTrack without
    touching pipeline code.
    """
    while True:
        started = time.perf_counter()
        try:
            sample = next(gen)
        except StopIteration:
            return
        decode_times.append(time.perf_counter() - started)
        yield sample


def measure_pipeline(video_name: str) -> dict:
    """Time decode, YOLO+ByteTrack, event build, and both DB writes on one clip."""
    video = VIDEOS_DIR / video_name
    meta = probe_video(video)
    scratch_db = SCRATCH / f"pipeline_{video.stem}.db"
    if scratch_db.exists():
        scratch_db.unlink()
    conn = init_db(scratch_db)

    tracker = YoloByteTrackTracker(device=YOLO_DEVICE)
    builder = EventBuilder(
        min_track_frames=MIN_TRACK_FRAMES,
        lost_frames=LOST_FRAMES,
        class_categories=CLASS_CATEGORIES,
    )

    decode_times: list[float] = []
    track_times: list[float] = []
    index_build_times: list[float] = []
    event_build_times: list[float] = []
    frame_rows = []
    frame_object_rows = []
    events = []
    sampled = 0

    # FrameRecord construction mirrors video_frame_records exactly; only the
    # timing wrappers are new, so the work being measured is the pipeline's.
    for sample in _measured_samples(sample_video(video, SAMPLE_RATE_FPS), decode_times):
        telemetry = telemetry_for_video_time(
            sample.video_time_s,
            start_time=SIM_START_DATETIME,
            waypoints=PATROL_WAYPOINTS,
            seconds_per_location=PATROL_SECONDS_PER_LOCATION,
            altitude_m=DRONE_ALTITUDE_M,
        )
        record = FrameRecord(
            frame_index=sample.frame_index,
            video_time_s=sample.video_time_s,
            timestamp=telemetry.timestamp,
            location=telemetry.location,
            altitude_m=telemetry.altitude_m,
            drone_x=telemetry.drone_x,
            drone_y=telemetry.drone_y,
            image=sample.image,
        )

        started = time.perf_counter()
        observations = tracker.track(record.image)
        track_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        frame, objects = build_frame_index(record, observations)
        index_build_times.append(time.perf_counter() - started)
        frame_rows.append(frame)
        frame_object_rows.extend(objects)

        started = time.perf_counter()
        events.extend(builder.process_frame(record, observations))
        event_build_times.append(time.perf_counter() - started)
        sampled += 1

    events.extend(builder.finish())

    started = time.perf_counter()
    store_tracks_and_events(conn, 1, events, tuning_settings={})
    event_write_s = time.perf_counter() - started

    started = time.perf_counter()
    store_frame_index(conn, 1, frame_rows, frame_object_rows)
    frame_write_s = time.perf_counter() - started

    conn.close()

    def stat(name: str, values: list[float]) -> dict:
        return {
            "stage": name,
            "n": len(values),
            "mean_ms": round(statistics.mean(values) * 1000.0, 3) if values else None,
            "median_ms": round(statistics.median(values) * 1000.0, 3) if values else None,
            "first_ms": round(values[0] * 1000.0, 3) if values else None,
            "mean_excl_first_ms": (
                round(statistics.mean(values[1:]) * 1000.0, 3) if len(values) > 1 else None
            ),
        }

    stages = [
        stat("decode (seek+read)", decode_times),
        stat("YOLO+ByteTrack", track_times),
        stat("frame-index build (in-loop)", index_build_times),
        stat("event build (in-loop)", event_build_times),
    ]
    total_measured = sum(sum(v) for v in (decode_times, track_times, index_build_times, event_build_times))
    total_measured += event_write_s + frame_write_s

    return {
        "clip": video_name,
        "video_duration_s": round(meta.duration_s, 3),
        "sampled_frames": sampled,
        "stages": stages,
        "event_write_total_s": round(event_write_s, 6),
        "event_write_per_sampled_frame_ms": round(event_write_s / sampled * 1000.0, 3),
        "frame_index_write_total_s": round(frame_write_s, 6),
        "frame_index_write_per_sampled_frame_ms": round(frame_write_s / sampled * 1000.0, 3),
        "sum_measured_s": round(total_measured, 3),
        "throughput_sampled_frames_per_s": round(sampled / total_measured, 3),
        "ratio_to_realtime": round(meta.duration_s / total_measured, 3),
        "device": tracker.device,
        "scratch_db": str(scratch_db),
        "db_size_bytes": scratch_db.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Decode comparison: frame-index seek vs sequential read (script only)
# ---------------------------------------------------------------------------


def _sample_indices(meta, rate: float) -> list[int]:
    """Source-frame indices used by sample_video, reproduced for the sequential reader."""
    total = max(1, math.ceil(meta.duration_s * rate - 1e-9))
    out: list[int] = []
    for offset in range(total):
        video_time = offset / rate
        if video_time >= meta.duration_s:
            break
        out.append(min(int(round(video_time * meta.fps)), meta.frame_count - 1))
    return out


def compare_decode(video_name: str) -> dict:
    video = VIDEOS_DIR / video_name
    meta = probe_video(video)

    seek_times: list[float] = []
    seek_images = 0
    for _ in _measured_samples(sample_video(video, SAMPLE_RATE_FPS), seek_times):
        seek_images += 1

    targets = _sample_indices(meta, SAMPLE_RATE_FPS)
    capture = cv2.VideoCapture(str(video))
    started = time.perf_counter()
    collected = 0
    next_target = 0
    for frame_no in range(meta.frame_count):
        ok, image = capture.read()
        if not ok or image is None:
            break
        if next_target < len(targets) and frame_no == targets[next_target]:
            next_target += 1
            collected += 1
    sequential_s = time.perf_counter() - started
    capture.release()

    if collected != seek_images:
        raise RuntimeError(
            f"Decode comparison mismatch on {video_name}: seek={seek_images} sequential={collected}"
        )

    return {
        "clip": video_name,
        "source_frames": meta.frame_count,
        "sampled_frames": seek_images,
        "seek_total_s": round(sum(seek_times), 3),
        "seek_ms_per_sampled_frame": round(statistics.mean(seek_times) * 1000.0, 3),
        "sequential_total_s": round(sequential_s, 3),
        "sequential_ms_per_sampled_frame": round(sequential_s / collected * 1000.0, 3),
        "sequential_faster_by_x": round(sum(seek_times) / sequential_s, 3),
    }


# ---------------------------------------------------------------------------
# Frame-index growth from the REAL rows (demo.db read-only -> scratch DB)
# ---------------------------------------------------------------------------


def frame_index_growth() -> dict:
    src = sqlite3.connect(f"file:{DEMO_DB}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT run_id, COUNT(*), SUM(detection_count) FROM frames GROUP BY run_id ORDER BY run_id"
    ).fetchall()
    objects = dict(src.execute("SELECT run_id, COUNT(*) FROM frame_objects GROUP BY run_id").fetchall())

    scratch = SCRATCH / "growth_scratch.db"
    if scratch.exists():
        scratch.unlink()
    conn = init_db(scratch)
    baseline = scratch.stat().st_size

    per_run = {}
    for run_id, frames, det_sum in rows:
        frame_data = src.execute(
            "SELECT frame_index,timestamp,zone,detection_count,description FROM frames WHERE run_id=? ORDER BY frame_index",
            (run_id,),
        ).fetchall()
        object_data = src.execute(
            "SELECT frame_index,track_id,class_name,confidence,bbox FROM frame_objects WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
        # Reuse the production write path so row layout and indexes match.
        # The read-only rows are tuples, so map them back to the index dataclasses.
        frames_typed = [
            FrameRow(
                frame_index=r[0],
                timestamp=r[1],
                zone=r[2],
                detection_count=r[3],
                description=r[4],
            )
            for r in frame_data
        ]
        objects_typed = [
            FrameObjectRow(
                frame_index=r[0],
                track_id=r[1],
                class_name=r[2],
                confidence=r[3],
                bbox=r[4],
            )
            for r in object_data
        ]
        from src.storage import store_frame_index as _sfi

        _sfi(conn, run_id, frames_typed, objects_typed)
        size = scratch.stat().st_size
        per_run[run_id] = {
            "frames": frames,
            "frame_objects": objects[run_id],
            "declared_detection_sum": det_sum,
            "rows_per_sampled_frame": round(objects[run_id] / frames, 3),
            "db_bytes_after_run": size,
            "incremental_bytes": size - baseline,
            "bytes_per_sampled_frame": round((size - baseline) / frames, 1),
            "bytes_per_object": round((size - baseline) / objects[run_id], 1),
        }
        baseline = size

    conn.close()
    src.close()
    return {"source": "data/demo.db (read-only); rows re-inserted into scratch DB", "runs": per_run}


# ---------------------------------------------------------------------------
# Synthetic query latency at 10k / 100k / 1M frames
# ---------------------------------------------------------------------------

_FRAME_DDL = """
CREATE TABLE frames (
    run_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    zone TEXT,
    detection_count INTEGER NOT NULL,
    description TEXT,
    PRIMARY KEY (run_id, frame_index)
);
CREATE TABLE frame_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    track_id INTEGER,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox TEXT NOT NULL
);
"""
_FRAME_INDEXES = """
CREATE INDEX idx_frames_run_timestamp ON frames(run_id, timestamp);
CREATE INDEX idx_frame_objects_run_class ON frame_objects(run_id, class_name);
CREATE INDEX idx_frame_objects_run_track ON frame_objects(run_id, track_id);
"""
_CLASSES = ("car", "truck", "person", "bus", "bicycle")
_ZONES = ("gate", "garage", "driveway", "perimeter")
_BATCH = 50000


def _populate(conn, n_frames: int, objects_per_frame: int) -> None:
    base = datetime(2026, 11, 24, 0, 0, 0)
    for start in range(1, n_frames + 1, _BATCH):
        stop = min(start + _BATCH, n_frames + 1)
        frows = []
        orows = []
        for i in range(start, stop):
            ts = (base + timedelta(seconds=(i - 1) / SAMPLE_RATE_FPS)).isoformat(timespec="seconds")
            frows.append((1, i, ts, _ZONES[i % 4], objects_per_frame, None))
            for k in range(objects_per_frame):
                track = 1 + ((i * 7 + k * 13) % 50)
                cls = _CLASSES[(i + k) % 5]
                orows.append((1, i, track, cls, 0.5 + (k % 5) * 0.05, "[0,0,10,10]"))
        conn.executemany("INSERT INTO frames VALUES (?,?,?,?,?,?)", frows)
        conn.executemany(
            "INSERT INTO frame_objects (run_id,frame_index,track_id,class_name,confidence,bbox) "
            "VALUES (?,?,?,?,?,?)",
            orows,
        )
        conn.commit()


def _median_query_ms(fn, repeats: int = 5) -> float:
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000.0)
    return round(statistics.median(times), 3)


def synthetic_query_latency(n_frames: int) -> dict:
    path = SCRATCH / f"synthetic_{n_frames}.db"
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(_FRAME_DDL)
    objects_per_frame = 5
    started = time.perf_counter()
    _populate(conn, n_frames, objects_per_frame)
    build_s = time.perf_counter() - started

    # A 10-second window near the middle of the synthetic timeline.
    mid = n_frames // 2
    base = datetime(2026, 11, 24, 0, 0, 0)
    win_start = (base + timedelta(seconds=(mid - 5) / SAMPLE_RATE_FPS)).isoformat(timespec="seconds")
    win_end = (base + timedelta(seconds=(mid + 5) / SAMPLE_RATE_FPS)).isoformat(timespec="seconds")

    def queries(label: str) -> dict:
        return {
            "label": label,
            "time_window_ms": _median_query_ms(
                lambda: query_frames_in_time_window(
                    conn, run_id=1, start=win_start, end=win_end, limit=50
                )
            ),
            "class_truck_ms": _median_query_ms(
                lambda: query_frames_by_class(conn, run_id=1, class_name="truck", limit=50)
            ),
            "track_42_ms": _median_query_ms(
                lambda: query_frames_for_track(conn, run_id=1, track_id=42, limit=50)
            ),
        }

    # Warm the page cache once, then time WITHOUT the frame indexes.
    queries("warmup")
    no_index = queries("without_indexes")

    conn.executescript(_FRAME_INDEXES)
    conn.execute("ANALYZE")
    conn.commit()
    with_index = queries("with_indexes")

    result = {
        "label": "SYNTHETIC (not measured on real footage)",
        "frames": n_frames,
        "frame_objects": n_frames * objects_per_frame,
        "objects_per_frame": objects_per_frame,
        "db_bytes": path.stat().st_size,
        "build_s": round(build_s, 2),
        "without_indexes": no_index,
        "with_indexes": with_index,
    }
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Stored-text facts: VLM cost scaling and agent cost (demo.db read-only)
# ---------------------------------------------------------------------------


def stored_scaling() -> dict:
    conn = sqlite3.connect(f"file:{DEMO_DB}?mode=ro", uri=True)
    vlm = []
    for run_id, tracks in conn.execute("SELECT run_id, COUNT(*) FROM tracks GROUP BY run_id ORDER BY run_id"):
        counts = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM descriptions WHERE run_id=? GROUP BY status", (run_id,)
            ).fetchall()
        )
        described = counts.get("described", 0)
        errors = counts.get("error", 0)
        too_small = counts.get("too_small", 0)
        total_s = conn.execute(
            "SELECT COALESCE(SUM(latency_ms),0.0)/1000.0 FROM descriptions WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        sampled = conn.execute("SELECT COUNT(*) FROM frames WHERE run_id=?", (run_id,)).fetchone()[0]
        vlm.append(
            {
                "run": run_id,
                "confirmed_tracks": tracks,
                "sampled_frames": sampled,
                "vlm_calls_attempted": described + errors,
                "vlm_calls_success": described,
                "too_small_skipped": too_small,
                "calls_per_track": round((described + errors) / tracks, 3),
                "calls_per_100_sampled_frames": round((described + errors) / sampled * 100, 2),
                "total_vlm_s": round(total_s, 2),
                "mean_s_per_call": round(total_s / (described + errors), 3)
                if (described + errors)
                else None,
            }
        )

    def aggregates(table: str, mode_col: str | None = None) -> dict:
        where = ""
        if mode_col:
            where = f" WHERE {mode_col}='ask'"
        row = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            f"COALESCE(SUM(estimated_cost_usd),0), COALESCE(SUM(latency_s),0) FROM {table}{where}"
        ).fetchone()
        n = row[0]
        return {
            "rows": n,
            "sum_prompt_tokens": row[1],
            "sum_completion_tokens": row[2],
            "sum_cost_usd": round(row[3], 8),
            "sum_latency_s": round(row[4], 3),
            "avg_prompt_tokens": round(row[1] / n, 1) if n else None,
            "avg_completion_tokens": round(row[2] / n, 1) if n else None,
            "avg_cost_usd": round(row[3] / n, 8) if n else None,
            "avg_latency_s": round(row[4] / n, 3) if n else None,
        }

    cost_all = aggregates("agent_costs")
    cost_ask = aggregates("agent_costs", "command")
    trace_all = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
        "COALESCE(SUM(estimated_cost_usd),0), COALESCE(SUM(latency_ms),0) FROM agent_traces"
    ).fetchone()
    trace_ask = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
        "COALESCE(SUM(estimated_cost_usd),0), COALESCE(SUM(latency_ms),0) "
        "FROM agent_traces WHERE mode='ask' AND cached=0"
    ).fetchone()

    def tr(row) -> dict:
        n = row[0]
        return {
            "rows": n,
            "avg_prompt_tokens": round(row[1] / n, 1) if n else None,
            "avg_completion_tokens": round(row[2] / n, 1) if n else None,
            "avg_cost_usd": round(row[3] / n, 8) if n else None,
            "avg_latency_ms": round(row[4] / n, 1) if n else None,
        }

    result = {
        "vlm_per_run": vlm,
        "agent_costs_all_calls": cost_all,
        "agent_costs_ask_only": cost_ask,
        "agent_traces_all_answers": tr(trace_all),
        "agent_traces_ask_uncached": tr(trace_ask),
    }
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    output: dict = {}
    for clip in CLIPS:
        output.setdefault("pipeline", []).append(measure_pipeline(clip))
    for clip in CLIPS:
        output.setdefault("decode_comparison", []).append(compare_decode(clip))
    output["frame_index_growth"] = frame_index_growth()
    output["synthetic_queries"] = [synthetic_query_latency(n) for n in (10_000, 100_000, 1_000_000)]
    output["stored_scaling"] = stored_scaling()

    out_path = SCRATCH / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"WROTE {out_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
