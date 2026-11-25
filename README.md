# Drone Security Analyst Agent

Prototype of a drone security analyst agent: simulated telemetry plus video frames, object/event logging, rule-based security alerts, a frame-by-frame SQLite index, and an agent that summarizes video and answers follow-up questions.

## Overview

The system watches sampled drone footage, logs what it sees into SQLite, raises rule-based alerts, and lets an agent summarize a run or answer follow-up questions from stored text only. Pipeline in one line: YOLO11s + ByteTrack → Gemma 4 E4B via LM Studio on new-track crops → an OpenAI mini model over stored text, with SQLite storage and a Streamlit UI. No image is ever sent to OpenAI.

## Feature spec (short)

Value to a property owner: an auditable, text-only account of what a patrol drone saw, with alerts and answers traceable to stored rows rather than a fresh model guess.

- Index every sampled frame by time, object class and track ID in SQLite.
- Raise one alert per confirmed track per rule, with a stored severity and reason.
- Summarize a run and answer follow-up questions over stored text, with token/cost rows and per-answer traces.

## Architecture

Three layers: local perception (YOLO11s + ByteTrack on sampled frames), local description (one stateless Gemma call per newly confirmed track crop), and cloud reasoning (OpenAI mini model reading stored text only). Alerts and the UI also read the database. Full detail: [docs/architecture.md](docs/architecture.md). Canonical Mermaid sources: [architecture.mmd](docs/diagrams/architecture.mmd), [data_flow.mmd](docs/diagrams/data_flow.mmd), [database_schema.mmd](docs/diagrams/database_schema.mmd), [alert_rule_flow.mmd](docs/diagrams/alert_rule_flow.mmd), [text_simulation.mmd](docs/diagrams/text_simulation.mmd).

## Setup (WSL Ubuntu)

Requires Python 3.12 and an NVIDIA driver with CUDA 13 support (tested on RTX 5050 in WSL2 Ubuntu).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`: set `OPENAI_API_KEY` and, if the Windows host address changed, `LM_STUDIO_BASE_URL`.

- CUDA wheels (`torch==2.14.1+cu130`) are pinned in `requirements.txt`; confirm PyTorch really uses CUDA before trusting any speed number (`.venv/bin/python scripts/check_gpu.py`).
- LM Studio runs on Windows and serves an OpenAI-compatible API. WSL cannot reach Windows `localhost:1234`, so `LM_STUDIO_BASE_URL` uses the host IP (example `http://172.19.144.1:1234/v1`); update it if that address changes.
- Secrets stay in `.env` (gitignored). Never print, commit or paste the key into docs or commands.

## Telemetry and clock

Simulated telemetry supplies altitude, patrol zones (gate, garage, driveway, perimeter) and a configurable clock (`SIM_START_DATETIME`, `TEXT_SECONDS_PER_FRAME`) so a run can cross midnight and demonstrate the night rules; a zone is the drone's location, not an image region.

## Running

Quick start with no models or network (writes only to `data/text_demo.db`):

```bash
.venv/bin/python scripts/run_text_demo.py --scenario data/scenarios/midnight_gate.jsonl --db data/text_demo.db
```

The other committed scenarios are `daytime_truck.jsonl` and `quiet_daytime.jsonl`. This path uses hand-labelled JSONL frames, not vision.

Full video pipeline (GPU and LM Studio; writes `data/demo.db` and an annotated video):

```bash
.venv/bin/python scripts/run_tracking.py
```

Agent over stored text (needs a valid key; a cached summary makes no model call, `--force` regenerates it):

```bash
.venv/bin/python scripts/ask_agent.py summary 1
.venv/bin/python scripts/ask_agent.py summary 1 --force
.venv/bin/python scripts/ask_agent.py ask 1 "Were any vehicles seen after midnight? Give times and track ids."
.venv/bin/python scripts/ask_agent.py traces 1
```

Streamlit UI (read-only, from the repo root):

```bash
streamlit run app/streamlit_app.py
```

## What is in the repo vs what you generate

From a fresh clone, the text-simulation demo works with no models and no video: the scenarios are committed in `data/scenarios/`, and the script creates `data/text_demo.db` itself.

The full video pipeline needs the two clips in `data/videos/`, a CUDA GPU and LM Studio; it creates `data/demo.db`. The Streamlit UI and the agent examples (`summary`, `ask`, `traces` and `query_frames.py --run 2`) need a generated `data/demo.db`, and the agent also needs an OpenAI key in `.env`. `.db` files and videos are gitignored, so neither is in the repo. The demo videos (to be added) show the full pipeline.

Source of the two sample clips: downloaded from Pexels (pexels.com, aerial video search). The individual clip page links were not recorded.

## Sample outputs

Real lines from stored data (nothing invented):

```text
# frame index row, data/text_demo.db
frame=3 timestamp=2026-11-24T00:00:00 zone=gate detections=1 description='A person in a dark jacket is now standing at the main gate.'
# alert row, data/demo.db
vehicle_at_night | warning | 2026-11-24T00:00:01 | vehicle track 62 first seen at 2026-11-24T00:00:01 in night window (00:00-05:00)
```

## Alert rules

| Rule | Fires when | Severity |
|------|-----------|----------|
| `vehicle_at_night` | confirmed car/truck/bus/motorcycle track first seen in the night window | warning |
| `person_at_night` | confirmed person track first seen in the night window | warning |
| `loitering` | any confirmed track of any class: `last_seen - first_seen` from the track's own first and last seen times exceeds 5 s, not zone dwell | warning |
| `heavy_vehicle_logged` | track classed `truck` or `bus`, any hour | info |

Night-window and loiter thresholds are constants in `src/alerts/__init__.py` (`NIGHT_WINDOW_START=0`, `NIGHT_WINDOW_END=5`, `LOITERING_THRESHOLD_S=5`); a window whose start hour is greater than its end hour wraps midnight. The simulated start time and clock step are settings in `src/config/settings.py` / `.env` (`SIM_START_DATETIME`, `TEXT_SECONDS_PER_FRAME`). `UNIQUE (run_id, track_id, rule_name)` keeps alert generation idempotent; alerts count tracker tracks, not physical objects.

## Querying the frame index

```bash
.venv/bin/python scripts/query_frames.py --run 2 --class truck --limit 1000
.venv/bin/python scripts/query_frames.py --db data/text_demo.db --run 1 --from 2026-11-23T23:59:00 --to 2026-11-24T00:03:00 --limit 1000
```

The first lists frames and detections of a class with distinct track IDs; the second is a time window crossing midnight. `--track N` selects all frames for one track. Queries open SQLite read-only.

## Bonus features

- One-sentence video summary: `.venv/bin/python scripts/ask_agent.py summary 1` returns the cached stored summary without a model call; add `--force` for a fresh one.
- Follow-up Q&A: `.venv/bin/python scripts/ask_agent.py ask 2 "Which alerts look questionable and why?"` can call read-only tools to fetch stored rows; `.venv/bin/python scripts/ask_agent.py traces 2` shows what was used.

## Testing

```bash
.venv/bin/python -m pytest -q
```

Recorded result: `99 passed`. Named scenario tests in `tests/test_agent.py` cover the **midnight alert** (`test_scenario_midnight_vehicle_and_fake_agent`) and the **truck logged correctly** (`test_scenario_truck_logged_and_fake_agent_cites_track`), both with fake model clients. The text-simulation tests (`tests/test_text_demo.py`, `tests/test_text_simulation.py`) and the scenario tests run with no model call and no network; the suite uses fakes and temporary databases. Full measurements: [docs/testing.md](docs/testing.md).

## Design Decisions

- Python 3.12 venv: system 3.14 lacked `ensurepip`.
- Pinned `torch`/`torchvision` cu130 for the RTX 5050 (compute capability 12.0); CUDA was verified by a matmul.
- LM Studio URL lives in `.env`/config because WSL cannot reach Windows localhost.
- Gemma 4 E4B with 512 max tokens: 128 tokens produced empty text (spent on reasoning).
- One stateless VLM call per new confirmed track crop; OpenAI settings are separate and the reasoning layer reads stored text only.
- Simulated clock is a pure `start + timedelta(video_time)` function, so midnight rollover is deterministic.
- SQLite holds alerts, summaries, costs and traces; alert writes are idempotent and frame-ID checks report extras instead of forcing a match.

Full list and reasons: [docs/design-decisions.md](docs/design-decisions.md).

## Results

Only numbers already recorded in [docs/testing.md](docs/testing.md); no new measurements here.

- End-to-end about **0.20 s per sampled frame**, **5.16 / 4.76 sampled frames/s** (0.51× / 0.47× real time) on the two real clips; frame-index seek decode dominates.
- Sequential decode measured **42–54× faster** than seeking in the benchmark, but it is not implemented in the pipeline.
- VLM calls scale with new tracks, not frames: 33 calls for 33 tracks on run 1, 11 calls for 23 tracks on run 2 (12 crops too small), about 1.2 s per call.
- Agent average: ~3.8k prompt + 305 completion tokens, **$0.0042 and 2.5 s per question**; total OpenAI spend $0.09634725.
- SYNTHETIC, NOT VIDEO: the `(run_id, timestamp)` index keeps time-window queries at ~0.03 ms even at 1 M frames (130 ms without).
- EXTRAPOLATION, NOT MEASURED: one hour of footage ≈ 2.0 h as written, or ~22 min with sequential decoding; ten drones ≈ 20 GPU-equivalents at current throughput.
- YOLO supplies counts and classes; VLM text is fallible context, not ground truth.

## Limitations

- ByteTrack switches IDs in crowded/occluded scenes, so alerts count tracks, not verified objects.
- The chosen VLM under-identifies large trucks: only **4 of 10** crops were described as vehicles (a keyword reading of stored text, not an accuracy score).
- Frame-index seeking makes the pipeline about 0.5× real time; decode, not YOLO, dominates.
- Clock, zones and telemetry are simulated. 12 of the 23 run-2 tracks had crops too small for the VLM and were skipped (recorded in docs/testing.md), and small objects from altitude are often missed (an observation, not a measured rate).

## Future work

- Apply sequential decode in the pipeline: it measured **42–54× faster** in the benchmark, but is **not implemented**.
- Fine-tune a small vision-language model or the detector on labelled aerial crops to improve recognition of small, blurry and large vehicles: **not implemented or measured**.

## AI Tools Used

The opencode assistant wrote code, tests and docs, ran no-model checks and interpreted failures; an external assistant helped plan steps and tuning; six installed design skills guided the Step 9B UI work; the human made scope decisions. Nothing was trusted without a real run. Per-step record: [docs/ai-tools-log.md](docs/ai-tools-log.md).

## Demo videos and PDF report

To be added.
