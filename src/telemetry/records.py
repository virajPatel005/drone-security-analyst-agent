"""One frame record shared by video and text simulation modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(slots=True)
class FrameRecord:
    """A sampled moment with its simulated clock and telemetry attached.

    Keeping one record type means later stages never branch on whether the
    source was a video or a text script. ``image`` is None for text mode and
    ``description`` is only populated by scripted scenes. ``frame_index`` is
    1-based because scripted scenes use "Frame 1" and previews read naturally.
    """

    frame_index: int
    video_time_s: float
    timestamp: datetime
    location: str
    altitude_m: float
    drone_x: float
    drone_y: float
    image: np.ndarray | None = None
    description: str | None = None
