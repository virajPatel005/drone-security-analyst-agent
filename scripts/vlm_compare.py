"""Step 10 VLM comparison on a fixed set of stored crops.

Lists LM Studio model names, probes which candidates actually accept an image
and return valid JSON, then compares every confirmed vision model on the same
10 run-2 crops. Also times 5 calls with the configured (chosen) model.

No model is downloaded or installed. No OpenAI call is made. Results go to a
scratch JSON file under /tmp/opencode/step10.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from src.config.settings import (
    DB_PATH,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
    LM_STUDIO_TIMEOUT_S,
)
from src.description import LmStudioVlmClient, parse_description

SCRATCH = Path("/tmp/opencode/step10")
# All six run-2 truck tracks plus four car tracks; includes 103, 326, 410.
FIXED_TRACKS = (2, 36, 64, 103, 106, 233, 326, 330, 353, 410)
# Plausibly-multimodal models by name; the remaining listings are coder,
# reasoning or embedding models and are not image candidates.
PROBE_CANDIDATES = (
    "google/gemma-4-e4b",
    "google/gemma-4-e2b",
    "google/gemma-3-12b",
    "qwen/qwen3.5-9b",
    "qwen/qwen2.5-vl-7b",
)


def model_names() -> list[str]:
    response = httpx.get(
        f"{LM_STUDIO_BASE_URL.rstrip('/')}/models", timeout=LM_STUDIO_TIMEOUT_S
    )
    response.raise_for_status()
    return [item["id"] for item in response.json()["data"]]


def fixed_crops() -> dict[int, Path]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT track_id, crop_path FROM descriptions "
        "WHERE run_id=2 AND status='described' ORDER BY track_id"
    ).fetchall()
    conn.close()
    available = {tid: Path(path) for tid, path in rows}
    result = {}
    for tid in FIXED_TRACKS:
        if tid not in available:
            raise RuntimeError(f"Fixed track {tid} has no described crop in demo.db")
        path = available[tid]
        if not path.is_file():
            raise FileNotFoundError(f"Fixed crop for track {tid} missing: {path}")
        result[tid] = path
    return result


def call_once(model: str, crop: Path) -> dict:
    client = LmStudioVlmClient(model_name=model)
    started = time.perf_counter()
    error = None
    raw = ""
    try:
        raw = client.describe_crop(crop)
    except Exception as exc:  # recorded, not silently swallowed
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
    attrs = parse_description(raw)
    return {
        "latency_ms": latency_ms,
        "valid_json": attrs is not None,
        "description": attrs["description"] if attrs else (raw[:300] or "<empty>"),
        "error": error,
    }


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    names = model_names()
    crops = fixed_crops()
    probe_crop = crops[FIXED_TRACKS[0]]

    probes = []
    for name in PROBE_CANDIDATES:
        if name not in names:
            probes.append({"model": name, "available": False, "vision_ok": False})
            continue
        result = call_once(name, probe_crop)
        probes.append(
            {
                "model": name,
                "available": True,
                "vision_ok": result["valid_json"],
                "probe_latency_ms": result["latency_ms"],
                "probe_error": result["error"],
                "probe_description": result["description"],
            }
        )

    vision_models = [p["model"] for p in probes if p["vision_ok"]]
    if LM_STUDIO_MODEL not in vision_models:
        # A failure of the configured/model-in-use VLM is a real failure here:
        # the fixed-set comparison cannot be built reliably. Stop and report.
        raise RuntimeError(
            f"Configured model {LM_STUDIO_MODEL} did not return valid JSON; "
            f"vision_ok models: {vision_models}"
        )

    # Explicit 5-call timing with the configured (chosen) model.
    five = []
    for tid in FIXED_TRACKS[:5]:
        result = call_once(LM_STUDIO_MODEL, crops[tid])
        five.append({"track": tid, **result})

    per_model = {}
    for name in vision_models:
        per_track = {}
        for tid, path in crops.items():
            result = call_once(name, path)
            per_track[tid] = result
        latencies = [r["latency_ms"] for r in per_track.values()]
        per_model[name] = {
            "calls": len(per_track),
            "mean_latency_ms": round(statistics.mean(latencies), 1),
            "median_latency_ms": round(statistics.median(latencies), 1),
            "valid_json": f"{sum(r['valid_json'] for r in per_track.values())}/{len(per_track)}",
            "per_track": per_track,
        }

    output = {
        "all_model_names": names,
        "fixed_tracks": list(FIXED_TRACKS),
        "probe": probes,
        "vision_models_compared": vision_models,
        "five_call_timing": {
            "model": LM_STUDIO_MODEL,
            "calls": five,
            "mean_latency_ms": round(statistics.mean([c["latency_ms"] for c in five]), 1),
            "valid_json": f"{sum(c['valid_json'] for c in five)}/{len(five)}",
        },
        "per_model": per_model,
        "note": "LM Studio returned no ground truth; crop files exist for all 10 fixed tracks.",
    }
    out_path = SCRATCH / "vlm_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"WROTE {out_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
