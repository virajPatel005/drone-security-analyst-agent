"""Preview the first sampled frames with their simulated clock and telemetry."""

from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import (
    DRONE_ALTITUDE_M,
    PATROL_SECONDS_PER_LOCATION,
    PATROL_WAYPOINTS,
    SAMPLE_RATE_FPS,
    SIM_START_DATETIME,
    VIDEO_PATH,
)
from src.telemetry.video_stream import video_frame_records


def main() -> None:
    records = video_frame_records(
        VIDEO_PATH,
        sample_rate_fps=SAMPLE_RATE_FPS,
        start_time=SIM_START_DATETIME,
        waypoints=PATROL_WAYPOINTS,
        seconds_per_location=PATROL_SECONDS_PER_LOCATION,
        altitude_m=DRONE_ALTITUDE_M,
    )
    print("index | video_time_s | timestamp | location | altitude_m")
    for record in islice(records, 10):
        print(
            f"{record.frame_index:5d} | {record.video_time_s:12.3f} | "
            f"{record.timestamp.isoformat(sep=' ')} | "
            f"{record.location:9s} | {record.altitude_m:.1f}"
        )


if __name__ == "__main__":
    main()
