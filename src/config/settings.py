"""Shared settings for the prototype, loaded from .env when present."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# LM Studio
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://172.19.144.1:1234/v1")
LM_STUDIO_FALLBACK_BASE_URLS = ("http://localhost:1234/v1",)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-e4b")
LM_STUDIO_TEMPERATURE = 0.0
LM_STUDIO_TIMEOUT_S = 120.0
LM_STUDIO_TEXT_PROMPT = "Reply with exactly: OK"
LM_STUDIO_IMAGE_PROMPT = "Describe this image in one sentence."
LM_STUDIO_TEXT_MAX_TOKENS = 16
LM_STUDIO_IMAGE_MAX_TOKENS = 512

# YOLO
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolo11s.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.10"))
YOLO_CHECK_RUNS = 2

# Tracker
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "configs/bytetrack_0.9.yaml")
TRACKED_CLASSES = ("person", "car", "motorcycle", "bus", "truck", "bicycle")
CLASS_CATEGORIES = {
    class_name: "person" if class_name == "person" else "vehicle"
    for class_name in TRACKED_CLASSES
}
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES", "2"))
LOST_FRAMES = int(os.getenv("LOST_FRAMES", "10"))
ANNOTATED_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
ANNOTATED_MAX_WIDTH = int(os.getenv("ANNOTATED_MAX_WIDTH", "720"))

# VLM
VLM_ENABLED = os.getenv("VLM_ENABLED", "1") == "1"
VLM_CROPS_DIR = PROJECT_ROOT / "data" / "crops"
VLM_CROP_PAD_RATIO = float(os.getenv("VLM_CROP_PAD_RATIO", "0.5"))
VLM_CROP_MIN_PAD_PX = int(os.getenv("VLM_CROP_MIN_PAD_PX", "4"))
VLM_CROP_MIN_SHORT_SIDE_PX = int(os.getenv("VLM_CROP_MIN_SHORT_SIDE_PX", "20"))
VLM_CROP_TARGET_SHORT_SIDE_PX = int(os.getenv("VLM_CROP_TARGET_SHORT_SIDE_PX", "336"))
VLM_PROMPT = (
    "You are given a small, low-resolution crop of an object from an aerial "
    "drone video. It may be blurry or only a few pixels wide. Do not guess "
    "details you cannot actually see; if unsure, say so in the description. "
    "Reply with JSON only: {\"object_type\": string, \"main_colour\": string, "
    "\"description\": one short sentence}."
)
VLM_JSON_SCHEMA = {
    "name": "track_description",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "object_type": {"type": "string"},
            "main_colour": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["object_type", "main_colour", "description"],
        "additionalProperties": False,
    },
}

# OpenAI reasoning layer. These settings are deliberately separate from LM
# Studio: the local VLM serves descriptions, the OpenAI model reads text only.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")
OPENAI_COST_CAP_USD = float(os.getenv("OPENAI_COST_CAP_USD", "1.00"))
OPENAI_WARNING_USD = float(os.getenv("OPENAI_WARNING_USD", "0.50"))
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2000"))
OPENAI_MAX_ROUNDS = int(os.getenv("OPENAI_MAX_ROUNDS", "4"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
# Row cap for each JSON list returned by agent tools. 50 is a floor, not a
# hard context limit: callers see a total and an explicit truncated flag.
OPENAI_TOOL_ROW_LIMIT = int(os.getenv("OPENAI_TOOL_ROW_LIMIT", "50"))
# Assumed prices in USD per million tokens, used for the cost guard only.
OPENAI_INPUT_COST_PER_MILLION = float(
    os.getenv("OPENAI_INPUT_COST_PER_MILLION", "0.75")
)
OPENAI_OUTPUT_COST_PER_MILLION = float(
    os.getenv("OPENAI_OUTPUT_COST_PER_MILLION", "4.50")
)

# Video
VIDEO_FILE_NAME = os.getenv("VIDEO_FILE_NAME", "16647865_2160_3840_30fps.mp4")
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
VIDEO_PATH = VIDEOS_DIR / VIDEO_FILE_NAME
SAMPLE_RATE_FPS = float(os.getenv("SAMPLE_RATE_FPS", "10.0"))
SIM_START_DATETIME = datetime.fromisoformat(
    os.getenv("SIM_START_DATETIME", "2026-11-23T23:55:00")
)
TEXT_SCRIPTS_DIR = PROJECT_ROOT / "data" / "text_scripts"
TEXT_SECONDS_PER_FRAME = float(os.getenv("TEXT_SECONDS_PER_FRAME", "60.0"))

# Patrol
PATROL_WAYPOINTS = (
    ("gate", 0.0, 0.0),
    ("garage", 120.0, 40.0),
    ("driveway", 60.0, 90.0),
    ("perimeter", -40.0, 160.0),
)
PATROL_LOCATIONS = tuple(label for label, _, _ in PATROL_WAYPOINTS)
PATROL_SECONDS_PER_LOCATION = float(os.getenv("PATROL_SECONDS_PER_LOCATION", "3.0"))
DRONE_ALTITUDE_M = float(os.getenv("DRONE_ALTITUDE_M", "30.0"))
DB_PATH = PROJECT_ROOT / "data" / "demo.db"