"""Ultralytics YOLO + ByteTrack adapter behind a small tracker interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    CLASS_CATEGORIES,
    TRACKER_CONFIG,
    YOLO_CONF,
    YOLO_DEVICE,
    YOLO_IMGSZ,
    YOLO_MODEL_NAME,
)
from src.perception.observations import Observation


class FrameTracker(Protocol):
    """Interface the event pipeline depends on, so tests can use a fake."""

    @property
    def device(self) -> str: ...

    def track(self, frame: np.ndarray) -> list[Observation]: ...


class YoloByteTrackTracker:
    """Persist YOLO detections and ByteTrack IDs across the sampled stream."""

    def __init__(
        self,
        *,
        model_name: str = YOLO_MODEL_NAME,
        imgsz: int = YOLO_IMGSZ,
        conf: float = YOLO_CONF,
        device: str = YOLO_DEVICE,
        tracker_config: str = TRACKER_CONFIG,
        class_categories: Mapping[str, str] = CLASS_CATEGORIES,
    ) -> None:
        self.model = YOLO(model_name)
        self.model.to(device)
        self._device = str(next(self.model.parameters()).device)
        self._imgsz = imgsz
        self._conf = conf
        self._tracker_config = tracker_config
        self._class_categories = dict(class_categories)

    @property
    def device(self) -> str:
        return self._device

    def track(self, frame: np.ndarray) -> list[Observation]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self._tracker_config,
            imgsz=self._imgsz,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        observations: list[Observation] = []
        for track_id, class_id, confidence, bbox in zip(
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
        ):
            class_name = self.model.names[int(class_id)]
            if class_name not in self._class_categories:
                continue
            observations.append(
                Observation(
                    track_id=int(track_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )
        return observations
