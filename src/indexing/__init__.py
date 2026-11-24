"""Pure frame-index records shared by the live pipeline and the backfill script.

Nothing in this module imports YOLO, OpenCV, SQLite, OpenAI or the VLM. Keeping
the mapping from a frame record + its observations into index rows pure means it
can be tested with dataclasses alone and reused unchanged by backfilling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.perception.observations import BBox, Observation
from src.telemetry.records import FrameRecord


@dataclass(frozen=True, slots=True)
class FrameRow:
    """One row of the ``frames`` table, before persistence."""

    frame_index: int
    timestamp: datetime
    zone: str
    detection_count: int
    description: str | None = None


@dataclass(frozen=True, slots=True)
class FrameObjectRow:
    """One row of the ``frame_objects`` table, before persistence."""

    frame_index: int
    track_id: int | None
    class_name: str
    confidence: float
    bbox: BBox


def build_frame_index(
    record: FrameRecord, observations: Sequence[Observation]
) -> tuple[FrameRow, list[FrameObjectRow]]:
    """Map one sampled frame and its detections into index rows.

    ``detection_count`` is deliberately derived from the observation list, so
    it always equals the number of ``frame_objects`` rows written for the frame.
    That equality is what makes the index self-consistent when queried later.
    """
    objects = [
        FrameObjectRow(
            frame_index=record.frame_index,
            track_id=o.track_id,
            class_name=o.class_name,
            confidence=o.confidence,
            bbox=o.bbox,
        )
        for o in observations
    ]
    frame = FrameRow(
        frame_index=record.frame_index,
        timestamp=record.timestamp,
        zone=record.location,
        detection_count=len(objects),
        description=record.description,
    )
    return frame, objects
