"""Stateless VLM descriptions for confirmed tracks, one call per track crop.

The perception layer saves one crop per confirmed track while it is tracking.
This module then runs *after* tracking, reads those saved files and asks the
local VLM for a structured description. Keeping the model calls out of the
tracking loop means a slow VLM cannot lower the effective frame rate.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from src.config.settings import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_IMAGE_MAX_TOKENS,
    LM_STUDIO_MODEL,
    LM_STUDIO_TIMEOUT_S,
    VLM_CROP_MIN_PAD_PX,
    VLM_CROP_MIN_SHORT_SIDE_PX,
    VLM_CROP_PAD_RATIO,
    VLM_CROP_TARGET_SHORT_SIDE_PX,
    VLM_JSON_SCHEMA,
    VLM_PROMPT,
)
from src.storage import query_descriptions, store_description

STATUS_DESCRIBED = "described"
STATUS_TOO_SMALL = "too_small"
STATUS_ERROR = "error"


@dataclass(frozen=True, slots=True)
class CropRecord:
    """One confirmed track and the crop saved for it during tracking."""

    track_id: int
    class_name: str
    crop_path: Path | None
    orig_width: int
    orig_height: int
    too_small: bool


@dataclass(frozen=True, slots=True)
class DescriptionSummary:
    """Status counts and total VLM latency read back from the database."""

    described: int
    too_small: int
    error: int
    total_latency_s: float


class VlmClient(Protocol):
    """Small interface so tests can replace LM Studio with a fake."""

    model_name: str

    def describe_crop(self, crop_path: Path) -> str: ...


class LmStudioVlmClient:
    """One stateless chat-completions call per crop against LM Studio."""

    def __init__(
        self,
        *,
        base_url: str = LM_STUDIO_BASE_URL,
        model_name: str = LM_STUDIO_MODEL,
        max_tokens: int = LM_STUDIO_IMAGE_MAX_TOKENS,
        timeout_s: float = LM_STUDIO_TIMEOUT_S,
    ) -> None:
        # Imported here so importing this module (and its tests) does not
        # depend on the OpenAI SDK being installed.
        from openai import OpenAI

        self.model_name = model_name
        self._max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=timeout_s,
        )

    def describe_crop(self, crop_path: Path) -> str:
        """Send one crop and return the raw assistant text.

        There is deliberately no message history: every call is independent.
        The image is base64-encoded into the user message, so the crop never
        leaves the local LM Studio server.
        """
        image_base64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
            # Keep >= 512: 128 was consumed by reasoning and returned empty.
            max_tokens=self._max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": VLM_JSON_SCHEMA,
            },
        )
        return response.choices[0].message.content or ""


def save_track_crop(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    *,
    pad_ratio: float = VLM_CROP_PAD_RATIO,
    min_pad_px: int = VLM_CROP_MIN_PAD_PX,
    min_short_side_px: int = VLM_CROP_MIN_SHORT_SIDE_PX,
    target_short_side_px: int = VLM_CROP_TARGET_SHORT_SIDE_PX,
) -> tuple[bool, int, int]:
    """Save a padded, enlarged crop and return ``(too_small, width, height)``.

    ``width``/``height`` are the *original* box dimensions in source pixels,
    before padding, so the report can count how many tracks were too small.
    A box whose shorter side is below ``min_short_side_px`` is not written.
    Padding is a fraction of the box on every side with a pixel floor, then
    clamped to the frame. The crop is upscaled (never downscaled) so its
    shorter side reaches ``target_short_side_px`` while keeping the aspect
    ratio.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    orig_w = int(round(box_w))
    orig_h = int(round(box_h))

    if orig_w <= 0 or orig_h <= 0 or min(orig_w, orig_h) < min_short_side_px:
        return True, orig_w, orig_h

    pad_x = max(int(round(box_w * pad_ratio)), min_pad_px)
    pad_y = max(int(round(box_h * pad_ratio)), min_pad_px)

    left = max(0, int(round(x1)) - pad_x)
    top = max(0, int(round(y1)) - pad_y)
    right = min(width, int(round(x2)) + pad_x)
    bottom = min(height, int(round(y2)) + pad_y)

    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return True, orig_w, orig_h

    short_side = min(crop.shape[:2])
    if short_side < target_short_side_px:
        scale = target_short_side_px / short_side
        new_width = max(1, int(round(crop.shape[1] * scale)))
        new_height = max(1, int(round(crop.shape[0] * scale)))
        crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), crop):
        raise RuntimeError(f"Could not write crop to {out_path}")
    return False, orig_w, orig_h


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_description(raw: str) -> dict[str, str] | None:
    """Parse the model's JSON defensively; return ``None`` if unusable.

    Accepts a bare object, an object inside a Markdown code fence, or an
    object surrounded by prose. Requires non-empty ``object_type``,
    ``description`` and a colour field. ``main_color`` is normalised to
    ``main_colour`` so the stored schema stays consistent.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    fenced = _FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    object_type = parsed.get("object_type")
    description = parsed.get("description")
    colour = parsed.get("main_colour", parsed.get("main_color"))

    if not all(isinstance(value, str) and value.strip() for value in (object_type, description, colour)):
        return None

    return {
        "object_type": object_type.strip(),
        "main_colour": colour.strip(),
        "description": description.strip(),
    }


def describe_run(
    conn,
    *,
    run_id: int,
    crops: Sequence[CropRecord],
    client: VlmClient,
) -> DescriptionSummary:
    """Describe every confirmed, not-yet-described track exactly once.

    Only track IDs present in the ``tracks`` table are considered, so an
    unconfirmed observation can never reach the VLM. Tracks that already have
    a ``descriptions`` row are skipped, which makes a second pass free. Each
    row is committed by ``store_description`` as soon as it is produced.
    """
    confirmed = {
        row[0]
        for row in conn.execute(
            "SELECT track_id FROM tracks WHERE run_id = ?", (run_id,)
        )
    }
    already_stored = {
        row[0]
        for row in conn.execute(
            "SELECT track_id FROM descriptions WHERE run_id = ?", (run_id,)
        )
    }

    seen: set[int] = set()
    for crop in crops:
        if crop.track_id in seen:
            continue
        seen.add(crop.track_id)

        if crop.track_id not in confirmed:
            continue
        if crop.track_id in already_stored:
            continue

        if crop.too_small:
            store_description(
                conn,
                run_id=run_id,
                track_id=crop.track_id,
                status=STATUS_TOO_SMALL,
                description=None,
                attributes=None,
                crop_path=None,
                orig_width=crop.orig_width,
                orig_height=crop.orig_height,
                model_name=client.model_name,
                latency_ms=None,
            )
            continue

        if crop.crop_path is None:
            store_description(
                conn,
                run_id=run_id,
                track_id=crop.track_id,
                status=STATUS_ERROR,
                description="missing crop path",
                attributes=None,
                crop_path=None,
                orig_width=crop.orig_width,
                orig_height=crop.orig_height,
                model_name=client.model_name,
                latency_ms=None,
            )
            continue

        started = time.perf_counter()
        try:
            raw = client.describe_crop(crop.crop_path)
        except Exception as exc:  # noqa: BLE001 - stored, not swallowed silently
            latency_ms = (time.perf_counter() - started) * 1000.0
            store_description(
                conn,
                run_id=run_id,
                track_id=crop.track_id,
                status=STATUS_ERROR,
                description=f"{type(exc).__name__}: {exc}",
                attributes=None,
                crop_path=str(crop.crop_path),
                orig_width=crop.orig_width,
                orig_height=crop.orig_height,
                model_name=client.model_name,
                latency_ms=latency_ms,
            )
            continue

        latency_ms = (time.perf_counter() - started) * 1000.0
        attributes = parse_description(raw)
        if attributes is None:
            store_description(
                conn,
                run_id=run_id,
                track_id=crop.track_id,
                status=STATUS_ERROR,
                description=raw if raw else "<empty response>",
                attributes=None,
                crop_path=str(crop.crop_path),
                orig_width=crop.orig_width,
                orig_height=crop.orig_height,
                model_name=client.model_name,
                latency_ms=latency_ms,
            )
        else:
            store_description(
                conn,
                run_id=run_id,
                track_id=crop.track_id,
                status=STATUS_DESCRIBED,
                description=attributes["description"],
                attributes=attributes,
                crop_path=str(crop.crop_path),
                orig_width=crop.orig_width,
                orig_height=crop.orig_height,
                model_name=client.model_name,
                latency_ms=latency_ms,
            )

    return _summary(conn, run_id=run_id)


def _summary(conn, *, run_id: int) -> DescriptionSummary:
    rows = query_descriptions(conn, run_id=run_id)
    counts = Counter(row["status"] for row in rows)
    total_latency_ms = sum(row["latency_ms"] or 0.0 for row in rows)
    return DescriptionSummary(
        described=counts.get(STATUS_DESCRIBED, 0),
        too_small=counts.get(STATUS_TOO_SMALL, 0),
        error=counts.get(STATUS_ERROR, 0),
        total_latency_s=total_latency_ms / 1000.0,
    )
