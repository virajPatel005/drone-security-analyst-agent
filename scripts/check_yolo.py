"""Time YOLO inference on one sample image using the configured device."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from src.config.settings import TEST_IMAGE_PATH, YOLO_CHECK_RUNS, YOLO_DEVICE, YOLO_MODEL_NAME


def main() -> None:
    if not TEST_IMAGE_PATH.is_file():
        raise FileNotFoundError(TEST_IMAGE_PATH)

    model = YOLO(YOLO_MODEL_NAME)
    model.to(YOLO_DEVICE)
    device = next(model.parameters()).device

    print(f"model: {YOLO_MODEL_NAME}")
    print(f"image: {TEST_IMAGE_PATH}")
    print(f"device_requested: {YOLO_DEVICE}")
    print(f"device_used: {device}")

    timings_ms: list[float] = []
    results = None
    for run_number in range(1, YOLO_CHECK_RUNS + 1):
        started = time.perf_counter()
        results = model.predict(source=str(TEST_IMAGE_PATH), device=YOLO_DEVICE, verbose=False)
        timings_ms.append((time.perf_counter() - started) * 1000)
        print(f"run_{run_number}_inference_ms: {timings_ms[-1]:.2f}")

    assert results is not None
    boxes = results[0].boxes
    names = model.names
    print(f"detections: {len(boxes) if boxes is not None else 0}")
    if boxes is not None:
        for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            print(f"  {names[int(class_id)]}: {confidence:.3f}")
    print(f"warmed_up_inference_ms: {timings_ms[-1]:.2f}")


if __name__ == "__main__":
    main()
