"""Shared settings for the prototype, loaded from .env when present."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://172.19.144.1:1234/v1")
LM_STUDIO_FALLBACK_BASE_URLS = ("http://localhost:1234/v1",)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-e4b")
LM_STUDIO_TEMPERATURE = 0.0
LM_STUDIO_TIMEOUT_S = 120.0
LM_STUDIO_TEXT_PROMPT = "Reply with exactly: OK"
LM_STUDIO_IMAGE_PROMPT = "Describe this image in one sentence."
LM_STUDIO_TEXT_MAX_TOKENS = 16
LM_STUDIO_IMAGE_MAX_TOKENS = 512

YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolo11n.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")
YOLO_CHECK_RUNS = 2

TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test_images"
TEST_IMAGE_NAME = os.getenv("TEST_IMAGE_NAME", "multiple_cars.jpg")
TEST_IMAGE_PATH = TEST_IMAGES_DIR / TEST_IMAGE_NAME
