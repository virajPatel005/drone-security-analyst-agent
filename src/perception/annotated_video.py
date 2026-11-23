"""Write a downscaled annotated demo video of tracked sampled frames."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.perception.observations import Observation


def annotated_video_path(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"annotated_{video_path.stem}.mp4"


class AnnotatedVideoWriter:
    """Draw boxes/track IDs and an overlay, then write at the sample rate.

    Downscaling first keeps the 4K portrait source cheap to review and makes
    drawing coordinates simple to scale from the original frame.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        max_width: int,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if max_width <= 0:
            raise ValueError("max_width must be positive")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self._fps = fps
        self._max_width = max_width
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._scale = 1.0

    def write(
        self,
        frame: np.ndarray,
        observations: Sequence[Observation],
        timestamp: datetime,
        zone: str,
    ) -> None:
        prepared = self._prepare_frame(frame)

        for observation in observations:
            x1, y1, x2, y2 = self._scaled_bbox(observation.bbox)
            color = (0, 255, 0) if observation.class_name == "person" else (0, 165, 255)
            cv2.rectangle(prepared, (x1, y1), (x2, y2), color, 2)
            label = (
                f"{observation.class_name} id={observation.track_id} "
                f"{observation.confidence:.2f}"
            )
            self._draw_label(prepared, label, x1, y1, color)

        overlay = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | zone={zone}"
        cv2.rectangle(prepared, (0, 0), (prepared.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(
            prepared,
            overlay,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        assert self._writer is not None
        self._writer.write(prepared)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> AnnotatedVideoWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if self._size is None:
            scale = min(1.0, self._max_width / width)
            self._scale = scale
            self._size = (int(round(width * scale)), int(round(height * scale)))
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                self._size,
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open annotated video for writing: {self.output_path}")

        if (width, height) == self._size:
            return frame
        return cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)

    def _scaled_bbox(self, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        return tuple(int(round(value * self._scale)) for value in bbox)  # type: ignore[return-value]

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )
        label_y = max(y, text_height + 4)
        cv2.rectangle(
            frame,
            (x, label_y - text_height - 4),
            (x + text_width + 4, label_y),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x + 2, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
