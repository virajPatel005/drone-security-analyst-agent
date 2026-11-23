"""Model-independent observation types shared by trackers and event logic."""

from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Observation:
    """One tracked object in one sampled frame.

    Keeping this type free of ultralytics lets the event builder be tested
    with a fake tracker and no GPU or model downloads.
    """

    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
