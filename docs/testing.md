# Testing

## Test Cases

| Case | Command | Result |
|------|---------|--------|
| GPU and CUDA initialisation | `.venv/bin/python scripts/check_gpu.py` | Pass: `cuda_available: True`, `cuda_matmul_ok: True` |
| YOLO inference on GPU | `.venv/bin/python scripts/check_yolo.py` run twice | Pass: 23 detections, `device_used: cuda:0` |
| LM Studio text and image calls | `.venv/bin/python scripts/check_lmstudio.py` | Pass on `http://172.19.144.1:1234/v1` |
| Config loading without services | `.venv/bin/python -m pytest -q` | Pass: `2 passed in 0.01s` |
| Fresh-machine installability | `pip install --dry-run -r requirements.txt` in throwaway venv at `/tmp/opencode/step1-verify-venv` (Python 3.12.13) | Pass: `Would install ... torch-2.14.1+cu130 torchvision-0.29.1+cu130 ...` |
| Clock mapping incl. midnight rollover | `.venv/bin/python -m pytest -q` | Pass |
| Synthetic-video sampler count and metadata | `.venv/bin/python -m pytest -q` | Pass: tiny OpenCV video in `tmp_path` |
| Telemetry determinism and route cycling | `.venv/bin/python -m pytest -q` | Pass |
| Text-mode parsing and midnight script | `.venv/bin/python -m pytest -q` | Pass: example `midnight_crossing.txt` crosses midnight |
| Track events: confirmation, flicker, zone change, loss, end-of-video, ID reuse, class filter | `.venv/bin/python -m pytest -q` | Pass: 7 new event tests, no model/GPU needed; total `24 passed in 0.16s` |

## Measurements

| Metric | Value |
|--------|-------|
| System Python | `Python 3.14.4` (no `ensurepip`) |
| Venv Python | `Python 3.12.13` |
| NVIDIA driver / CUDA UMD | `NVIDIA-SMI 610.53`, `CUDA UMD Version: 13.3` |
| GPU | `NVIDIA GeForce RTX 5050 Laptop GPU`, 8151 MiB, compute capability `(12, 0)` |
| Torch | `2.14.1+cu130`, `torch_cuda_build: 13.0` |
| YOLO model / image | `yolo11n.pt`, `data/test_images/multiple_cars.jpg` |
| YOLO first run, inference 1 / 2 | 2171.23 ms / 12.43 ms |
| YOLO second run, inference 1 / 2 | 1264.62 ms / 12.95 ms |
| YOLO warmed-up time (reported) | 12.95 ms per image |
| YOLO detections | 23 total: 22 cars, 1 stop sign; top car confidence 0.824 |
| LM Studio working base URL | `http://172.19.144.1:1234/v1` |
| LM Studio localhost attempt | Failed: `ConnectError [Errno 111] Connection refused` |
| LM Studio models found | 11; first listed `google/gemma-4-e4b` |
| LM Studio text reply / latency | `OK` / 0.29 s |
| LM Studio image reply / latency | One-sentence traffic description / 7.63 s |
| Image call with 128 max tokens | Failed to produce visible text; `content` empty, 125 reasoning tokens used |

## YOLO Comparison: multiple_cars.jpg

| Model | imgsz | Cars detected | Warmed-up ms/image |
|-------|-------|---------------|--------------------|
| yolo11n.pt | 640 | 22 | 15.22 |
| yolo11n.pt | 1280 | 51 | 19.13 |
| yolo11s.pt | 640 | 13 | 9.46 |
| yolo11s.pt | 1280 | 35 | 21.68 |

Note: single image, single timed run each, conf=0.25. Ground-truth count: not available. The scene is dense and the boxes overlap heavily, so an exact manual count was not feasible. Qualitative review by the developer: yolo11n at imgsz 1280 captured the most visible cars; the distant traffic remains partly undetected at every setting. Detection counts are therefore the model's own and are not validated against ground truth.

## Step 2: Video Sampling Measurements

| Metric | Value |
|--------|-------|
| Demo video | `data/videos/16647865_2160_3840_30fps.mp4` |
| Video resolution | 2160 x 3840 (portrait) |
| Video fps | 29.97003 |
| Video frame count | 420 |
| Video duration | 14.014 s |
| Sample rate | 5.0 frames per video second |
| Sampled frames | 71 |
| Full read + sample run 1 | 25.451 s (358.47 ms/frame) |
| Full read + sample run 2 | 27.318 s (384.76 ms/frame) |
| First sampled frame | index 1, 0.000 s, `2026-11-23 23:55:00`, gate |
| Last sampled frame | index 71, 14.000 s, `2026-11-23 23:55:14`, gate |
| Test suite | `17 passed in 0.18s` |

Note: the video is 2160x3840 portrait. At 1 fps the pass took about 0.34 s per sampled frame (5.149 s / 15 frames), mostly 4K decode time. At 5 fps it sampled 71 frames in 25.451 s / 27.318 s, about 0.358 s / 0.385 s per sampled frame.

## Step 3: Tracking and Event Measurements

Command: `.venv/bin/python scripts/run_tracking.py` on `data/videos/16647865_2160_3840_30fps.mp4`.

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Sampled frames | 71 | 71 |
| Video duration | 14.014 s | 14.014 s |
| Sampling rate (71 / 14.014), not throughput | 5.066 fps | 5.066 fps |
| YOLO + tracker time per frame, excl. decode/drawing | 84.82 ms (includes warm-up) | 64.38 ms |
| Implied YOLO + tracker rate, excl. decode/drawing | about 11.8 fps | about 15.5 fps |
| Device | `cuda:0` | `cuda:0` |
| Event counts | 39 `appeared`, 91 `entered_zone`, 39 `left` | 39 `appeared`, 91 `entered_zone`, 39 `left` |
| Observed track summaries | 52 | 52 |

Note: `5.066 fps` is only the sampled-frame rate (`71 / 14.014 s`), not end-to-end throughput. YOLO + tracker processing was about **64 ms per frame (about 15 fps)** on the second run, excluding 4K video decode and annotation drawing. The first run's **84.8 ms** includes model/tracker warm-up.

| Track class | Count |
|-------------|-------|
| car | 29 |
| person | 19 |
| motorcycle | 3 |
| truck | 1 |

| Annotated output | Value |
|------------------|-------|
| Path | `data/outputs/annotated_16647865_2160_3840_30fps.mp4` |
| Resolution | 720 x 1280 |
| FPS | 5.0 |
| Frame count | 71 |
| File size | 4,751,009 bytes (about 4.75 MB) |

### Step 3 observations and issues

- 52 distinct tracker IDs were observed, but only 39 appearances were confirmed. The 13 unconfirmed IDs were mostly one-frame flickers and were suppressed by `MIN_TRACK_FRAMES=2`.
- Class assignment was unstable for at least one track: track 14 changed between `car` and `motorcycle`. Ultralytics track IDs also persisted across the simulated zone changes because the zone comes from patrol telemetry, not image position.
- The 91 `entered_zone` events include the initial zone entry for each confirmed appearance plus repeated entries for long-lived tracks each time the 3 s patrol route changed. This is expected from the current definition of zone, but it is noisy.
- The demo clip is a dense street scene, and visual review of a contact sheet showed heavily overlapping labels in the busiest part. Some boxes sat on bright foreground lamps/barriers and look suspect. Ground-truth validation is still pending, so detection accuracy is not validated.
- On the first Step 3 run, Ultralytics automatically installed `lap==0.5.14` while initialising ByteTrack, printing `Ultralytics requirement ['lap>=0.5.12'] not found, attempting AutoUpdate`. ByteTrack needs `lap`, so `lap==0.5.14` is now pinned in `requirements.txt` after confirming the installed version with `pip freeze`.
- The event-builder tests use fake observations only; they do not exercise CUDA, YOLO, ByteTrack, or the video writer.

## Step 3 tuning experiments (end of Step 3)

### 5 fps / 10 fps / 15 fps with default bytetrack.yaml

| rate fps | sampled frames | unique IDs | detections | confirmed | reapp(event) | gap IDs | gap events | mean fr/track | ≤3 fr | mean dur (s) | <0.5 s | track ms/fr | decode ms/fr | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 71 | 52 | 813 | 39 | 1 | 20 | 52 | 15.63 | 22 | 3.75 | 19 | 43.4 | 341.8 | 28.1 |
| 10 | 141 | 58 | 1732 | 59 | 7 | 32 | 89 | 29.86 | 17 | 3.60 | 19 | 42.1 | 356.2 | 57.9 |
| 15 | 211 | 70 | 2677 | 84 | 16 | 44 | 128 | 38.24 | 14 | 3.05 | 23 | 44.3 | 354.2 | 86.7 |

**Finding:** Higher fps did **not** reduce fragmentation; unique IDs increased 52 → 58 → 70, and mean track duration fell 3.75 → 3.60 → 3.05 s. Decode dominates (~350 ms/sampled frame, 8–10× the YOLO time).

### match_thresh 0.8 vs 0.9 (at 10 fps)

| config | unique IDs | mean dur (s) | ≤3 fr | <0.5 s | gap IDs | gap events | van frames/141 | low conf <0.35 |
|---|---|---|---|---|---|---|---|---|
| match 0.8 (baseline) | 58 | 3.60 | 17 | 19 | 32 | 89 | 141 (conf 0.873, class car) | n/a |
| **match 0.9** | **48** | **4.50** | **11** | **11** | **29** | **102** | **141 (conf 0.873, class car)** | n/a |

**Finding:** Looser association (0.9) cuts unique IDs 58 → 48, raises mean duration 3.60 → 4.50 s, halves sub-0.5 s tracks (19 → 11). The van is fully detected at conf 0.873.

### Confidence / model combos (A, B, C) at 10 fps, match 0.9

| run | model | conf | uniq IDs | detections | confirmed | gap IDs | gap events | mean dur (s) | <0.5 s | low conf <0.35 | short (<=5 fr) | van frames/141 | van class votes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | yolo11n | 0.10 | **45** | 2331 | 56 | **18** | **36** | **5.68** | **7** | **7** | **4** (IDs 15,17,19,233) | 109 / 141 | `{'truck': 101, 'bus': 8}` (max conf **0.341**) |
| **B** | yolo11s | 0.25 | 57 | 2561 | 64 | 40 | 144 | 5.41 | 8 | 4 (IDs 15,96) | 2 | 141 / 141 | `{'car': 141}` (max conf **0.873**) |
| **C** | yolo11s | **0.10** | 54 | 3059 | 56 | 26 | 47 | 5.96 | 5 | **4** (ID 15) | 1 | **141 / 141** | `{'car': 134}` (max conf **0.873**) |

**Findings:**
- **Higher conf (0.25) + yolo11s (B)** gives the van full 141/141 coverage at high confidence (0.873), but creates more flicker/IDs (57, 144 gap events).
- **Lower conf (0.10) + yolo11n (A)** reduces IDs to 45 and gap events to 36, mean duration 5.68 s, but the van only reaches 109/141 with max conf 0.341 and class votes truck/bus.
- **Lower conf (0.10) + yolo11s (C)** is the middle ground: 54 IDs, 47 gap events, mean duration 5.96 s, 4 low-conf IDs (only ID 15 ≤5 frames), van at 141/141 with class car at conf 0.873.
- **No ground truth exists** for this dense night scene; detection counts are the model's own.
- **Not every parked bike gets a box** in some configs; the "parked bikes/scooters" row is an aggregate of several distinct tracks.
- **Distant traffic still switches IDs** frequently — this is a fundamental ByteTrack limitation on crowded scenes.
- **Decoding by frame-index seek costs ~350 ms per sampled frame**, about 8–10× the YOLO+tracker time (~37–46 ms). This dominates end-to-end throughput.
- **`SAMPLE_RATE_FPS=10`** with `LOST_FRAMES=10` gives a loss window of ~1.0 s (10 sampled frames ≈ 1 s at 10 fps); at 5 fps the window was ~2 s.

## Step 4: SQLite Storage Measurements

Command: `.venv/bin/python scripts/run_tracking.py` on `data/videos/16647865_2160_3840_30fps.mp4` (with storage enabled).

| Metric | Value |
|--------|-------|
| Database path | `data/demo.db` |
| Runs table rows | 1 |
| Tracks table rows | 52 |
| Events table rows | 261 |

### Query: events between 23:59:55 and 00:00:09

All 261 events are returned, because the entire demo video (simulated start `2026-11-23T23:59:55`, duration 14.014 s) falls within this window, which crosses midnight.

### Query: events for track 1

| timestamp | event_type | zone | dwell_s |
|-----------|-----------|------|---------|
| 2026-11-23T23:59:55 | appeared | gate | 0.0 |
| 2026-11-23T23:59:55 | entered_zone | gate | 0.0 |
| 2026-11-23T23:59:58 | entered_zone | garage | 2.9 |
| 2026-11-24T00:00:01 | entered_zone | driveway | 2.9 |
| 2026-11-24T00:00:04 | entered_zone | perimeter | 2.9 |
| 2026-11-24T00:00:07 | entered_zone | gate | 2.9 |
| 2026-11-24T00:00:09 | left | gate | 2.0 |

### Counts by event type

| event_type | count |
|-----------|-------|
| appeared | 56 |
| entered_zone | 149 |
| left | 56 |

### Tracks by class

| class | count |
|-------|-------|
| car | 23 |
| person | 18 |
| motorcycle | 7 |
| bicycle | 3 |
| truck | 1 |

Note: the midnight-crossing window query works without special handling because timestamps are stored as full ISO 8601 date‑times (`2026-11-23T23:59:55`), enabling ordinary range comparisons.

The new defaults are: **yolo11s, conf 0.10, imgsz 1280, 10 fps sampling, ByteTrack match_thresh 0.9**.

## Step 5: Rule-based Alerts Measurements

Two real video clips were run through the full pipeline with the default configuration (yolo11s, conf 0.10, imgsz 1280, 10 fps sampling) and both runs were appended to the same `data/demo.db`. `scripts/run_tracking.py` now generates the alerts itself after storing tracks and events, so the numbers below come from the pipeline with no manual rule calls.

### drone_intersection.mp4
- Simulated start: 2026-11-23T23:59:55 (the clip crosses midnight, so tracks split across the night window)
- Resolution: 1080x1920, fps 30.0, duration 16.633 s
- Sampled frames: 167
- Unique tracks: 33 (all car class)
- Person tracks: 0
- Truck/bus tracks: 0
- appeared events: 44
- Alert results:
  - vehicle_at_night: 12 (car tracks first-seen in 00:00-05:00 window)
  - person_at_night: 0
  - loitering: 10 (tracks with last_seen - first_seen > 5 s)
  - heavy_vehicle_logged: 0
- TODO: source and licence to be added if provided

### drone_truck.mp4
- Simulated start: 2026-11-24T14:00:00 (daytime, so the night rules do not apply)
- Resolution: 1920x1080, fps 24.0, duration 18.333 s
- Sampled frames: 184
- Unique tracks: 23 (17 car + 6 truck)
- Person tracks: 0
- Truck/bus tracks: 6
- appeared events: 25
- Alert results:
  - vehicle_at_night: 0 (run starts at 14:00; the earlier "16" figure came from a different simulated start and is not reproducible from this stored run)
  - person_at_night: 0
  - loitering: 5 (tracks with last_seen - first_seen > 5 s)
  - heavy_vehicle_logged: 6 (truck tracks, severity info)
- The six heavy-vehicle alerts correspond to track IDs 2, 103, 326, 330, 360 and 410. Each has an `appeared` event, so each is confirmed. The `tracks` table has no `confirmed` column; confirmation is implied by having an `appeared` event.
- Caveat: `tracks.frames_seen` is currently `len(track events)` (appeared + zone changes + left), not a frame count. The name is misleading and it should not be used as a detection count.
- TODO: source and licence to be added if provided

### Rule test with fake data (no models)
Each rule was tested with injected track data:
- `vehicle_at_night`: fires for tracks with first-seen hour in [00:00, 05:00), does not fire for 23:59:58
- `person_at_night`: does not fire when no person tracks exist
- `loitering`: fires when last_seen - first_seen > 5 s, does not fire for shorter tracks
- `heavy_vehicle_logged`: fires for truck/bus tracks regardless of hour, severity info
- Once-per-track dedup verified via UNIQUE (run_id, track_id, rule_name) constraint
- Unconfirmed tracks (not in tracks table) ignored by all rules

### Midnight boundary verification
- Track at 23:59:58: hour 23 not in night window → no alert
- Track at 00:00:03: hour 0 in night window → alert fires

### YOLO-only diagnostic on drone_truck.mp4 (no tracker)
Temporary script (`/tmp/opencode/truck_probe.py`, not committed), yolo11s, conf 0.10, 10 fps sampling, no ByteTrack. Annotated start/middle/end frames were written to `/tmp/opencode/truck_probe/`.

| imgsz | sampled frames | mean vehicle dets/frame | frames with >=1 truck/bus | car | truck | bus | train | person | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| 1280 | 184 | 17.011 | 184 (100%) | 2106 | 988 | 36 | 244 | 51 | 27.35 |
| 1920 | 184 | 26.734 | 184 (100%) | 3263 | 1633 | 23 | 338 | 43 | 33.51 |

Interpretation:
- Detection counts rose ~57% at 1920 for ~23% more time per frame, so 1920 does not look like a cheap win on this GPU.
- 100% of frames contain a truck/bus label, but the annotated frames show several high-confidence false positives on static structures (a building labelled `truck 0.22`, the road labelled `train`). Detection presence alone is not evidence that a real truck is present, which is why `heavy_vehicle_logged` must keep severity `info` and why per-track confirmation matters.
- The probe sees ~11 cars per frame but the stored tracker run produces only 23 tracks. That gap is unexplained; a tracker-level count (with IDs) is needed before drawing conclusions about fragmentation.

### Known limits of the alert layer
- **Alerts count tracks, not physical objects.** One truck split across several ByteTrack IDs becomes several `heavy_vehicle_logged` alerts, and two objects sharing an ID become one. The counts are "confirmed tracks that matched a rule".
- **Small objects from altitude are often missed.** The raw YOLO probe found roughly 11 car detections per frame, while the tracker stored only 23 tracks for the whole clip. Distant vehicles are a few pixels wide and disappear between sampled frames.
- **`person_at_night` is tested on fake data only.** Neither demo clip produced a confirmed person track, so the rule's behaviour on real footage is unverified.
- **Tiled inference is untried future work.** Running YOLO on overlapping tiles (SAHI-style) is the obvious way to recover small objects, but it has not been implemented or measured.

### Test count reconciliation
- Step 4 commit (8b2f7c7) collects **31** tests, not 30 (verified in a clean worktree at that commit).
- Current tree collects **40** tests: net **+9**.
- `tests/test_storage.py` went from 7 to 16 tests.
- Tests in `test_storage.py` that changed relative to Step 4:
  - **9 alert tests added**, including one that runs `ALERT_RULES` twice and proves the stored alert count does not grow.
  - `test_night_window_22_00_to_05_00_both_sides_midnight` was replaced by `test_night_window_wraps_midnight_22_to_05`, which patches the window to 22‑5 and checks real tracks at 23:30/04:00 (fire) and 12:00/21:59/05:00 (no fire). The old test only asserted the constants.
  - `test_store_and_reopen_file_db` and `test_per_track_query_with_id_switching_note` were restored; the file-DB test now writes runs, tracks, events and an alert with `tmp_path` and reopens the connection.
- The earlier "13 new alert tests" claim was wrong: 13 was the then-total for `test_storage.py`, not the number added.

### VLM description measurements (real runs)
Two clips were processed with the full pipeline (yolo11s, conf 0.10, 10 fps sampling). VLM calls used `google/gemma-4-e4b` against the LM Studio server at `http://172.19.144.1:1234/v1`. Each confirmed track produced exactly one crop (the `appeared` frame) and one VLM call; too-small crops were skipped. Results:

| Run | Start | Track count | Described | Too small | Error | Total VLM seconds |
|-----|-------|------------|-----------|-----------|-------|-------------------|
| 1   | 23:59:55 | 33 | 33 | 0 | 0 | 42.37 |
| 2   | 14:00:00 | 23 | 11 | 12 | 0 | 14.59 |

### Run 2 truck-class tracks and `heavy_vehicle_logged` false-positive analysis
The six truck-class tracks in run 2 (ids 2, 103, 326, 330, 360, 410) each received a `heavy_vehicle_logged` alert (severity `info`). The VLM descriptions of their crops are:

| track | VLM description | Looks like a truck? |
|-------|----------------|----------------------|
| 2 | A rectangular white object is visible, but its full nature and details are unclear. | uncertain |
| 103 | A blurry rectangular object that appears to be a storage container. | no |
| 326 | There are three small, rectangular blocks of blue and red colors. | no |
| 330 | A small red vehicle is visible on a dusty surface. | unclear (a small vehicle, not described as a truck) |
| 360 | A large, rectangular white object is visible on the ground. | no |
| 410 | A blurry view of a brick building facade with visible windows and architectural details. | no |

The VLM descriptions show that 4 of the 6 truck-class tracks are described as non‑trucks, 1 (track 330) is unclear — a small vehicle not described as a truck — and 1 (track 2) is uncertain.  Together this means only track 330 could reasonably be considered a genuine truck; the other five `heavy_vehicle_logged` alerts fire on tracks whose crops do not depict trucks, so the alerts are effectively false positives from the rule's class‑only perspective.

### Test count reconciliation
- The earlier "13 new alert tests" claim was wrong: 13 was the then-total for `test_storage.py`, not the number added.

## Step 7: OpenAI Agent (real runs, 24 Nov 2026)

Key setup was checked with only the two allowed commands: `grep -c OPENAI_API_KEY .env` returned `1`, and the shell-variable/`.env` comparison returned `same`. No key, prefix or masked value was printed anywhere.

Model listing through `OpenAIClient().list_model_names()` succeeded and printed model IDs only. The configured default `gpt-5.4-mini` is present in the list, so no substitute model was needed.

Database state before the runs (read-only check, **no rebuild performed**): runs 1 and 2 present; alerts run 1 = 12 `vehicle_at_night` + 10 `loitering`, run 2 = 6 `heavy_vehicle_logged` + 5 `loitering`; descriptions run 1 = 33, run 2 = 23. There were already 2 real `agent_costs` rows ($0.016543) and 2 `run_summaries` rows from earlier real calls, so the running totals below include that earlier spend. Because cached summaries existed, both summaries were forced with `--force` to guarantee real model calls.

Fake-client tests: `.venv/bin/python -m pytest -q` → `57 passed in 0.65s` (above the required 49; Step 6 had 49, `tests/test_agent.py` adds 8).

Real commands, one at a time, all succeeded. Raw stats lines:

| Command | model | prompt tokens | completion tokens | latency s | cost USD | running total USD |
|---------|-------|---------------|-------------------|-----------|----------|-------------------|
| `summary 1 --force` | gpt-5.4-mini | 3663 | 1698 | 5.721 | 0.010388 | 0.026932 |
| `summary 2 --force` | gpt-5.4-mini | 3467 | 794 | 3.422 | 0.006173 | 0.033105 |
| `ask 1 "Were any vehicles seen after midnight? Give times and track ids."` | gpt-5.4-mini | 3600 | 267 | 2.189 | 0.003902 | 0.037006 |
| `ask 2 "Show all truck events and tell me whether they look reliable."` | gpt-5.4-mini | 3403 | 1451 | 4.877 | 0.009082 | 0.046088 |
| `ask 2 "Which alerts look questionable and why?"` | gpt-5.4-mini | 3398 | 582 | 2.905 | 0.005168 | 0.051256 |
| `ask 1 "What colour was the driver's jacket?"` | gpt-5.4-mini | 3594 | 7 | 1.242 | 0.002727 | 0.053983 |

Final `agent_costs`: 8 rows, $0.053983 total, stored in SQLite (not memory). No warning ($0.50) or cap ($1.00) was reached.

### Answer checks against data/demo.db

- `ask 1` returned 12 vehicle-at-night events with track IDs 62, 67, 68, 70, 76, 79, 85, 87, 88, 96, 109, 110 and their timestamps. This matches `alerts` for run 1 exactly (12 rows) — no wrong or invented IDs.
- `ask 2` listed truck tracks 2, 103, 326, 330, 360, 410 and "6 tracks / 33 events". The `tracks` and `events` tables confirm exactly those 6 truck tracks and 33 events. Descriptions were quoted accurately except track 410: the agent wrote "multi-story building" where the stored text says "brick building facade" — same meaning, not invented. Its reliability labels were cautious rather than wrong, but it called tracks 103 and 360 "somewhat reliable" despite non-truck descriptions (see below).
- `ask 3` flagged tracks 103, 326, 410 and 2 as questionable `heavy_vehicle_logged` alerts, plus loitering tracks 47, 196 and 367 whose descriptions have status `too_small`. All IDs and quotes exist in the DB. It did **not** flag track 360 although its stored description ("A large, rectangular white object is visible on the ground") is also non-truck — an omission, not an invented claim. It also said "run not given" because the preloaded tool context contains no run ID; that is accurate.
- `ask 4` answered exactly "the data cannot tell", as required.
- Summary 1 reported "22 warning alerts" (= 12 vehicle_at_night + 10 loitering) and "no truck alerts stored" for run 1 — both true; its `questionable` IDs were 76 and 96. Summary 2 reported "six truck tracks" — true — and `questionable` IDs 103, 360, 410, omitting 326. Every track ID cited in both summaries (21 in run 1, 9 in run 2) was verified to exist in that run's `tracks` table: no invented IDs.
- Expected VLM non-truck descriptions for 103, 326, 360 and 410 were confirmed directly from the `descriptions` table.

### Failures and limits observed this step
- No command failed in this final run. Earlier in the Step 7 repair, real summaries had been run against the wrong base URL (LM Studio) and contained invented track IDs; those are not the rows used here because both summaries were regenerated with `--force`.
- The preloaded context for `ask`/`summary` does not include `run_id`, so the model cannot cite a run number in answers; this is a known presentation gap, not a data error.
- The agent sometimes under-flags a questionable truck track (326 in summary 2, 360 in ask 3). This is model judgment, not a code failure; the underlying stored text is correct.

### Additional Step 7 result notes
1. Summary 2 flagged tracks 103, 360 and 410 as questionable and missed 326; ask question (c) ("Which alerts look questionable and why?") missed 360. Both are partial results, not data errors — the stored descriptions for all four tracks are correct in `data/demo.db`.
2. One summary recommendation misreads simulated zone changes as "patrol or dwell behavior". Zone is the drone's telemetry location, not the object's position, so repeated `entered_zone` events do not show the object patrolling or dwelling.
3. Answers do not cite the run ID because the preloaded tool context omits it. This is a context/presentation gap, not a database error.
4. The summary is stored as JSON in `run_summaries.text`; the UI should render it (parse and display the `summary`, `notable_alerts`, `questionable` and `recommendations` fields) rather than printing the raw JSON string.

## Step 8: Scenario Tests and Agent Trace Table (24 Nov 2026)

### How the agent actually calls tools (Part 1 answer)
It is a **mix**, not one mechanism:
- The model genuinely chooses tools in a loop. `run_conversation` (`src/agent/__init__.py:517-563`) calls `client.call(..., tools=TOOL_DEFINITIONS)`, reads `result["tool_calls"]`, runs each via `_run_tool`, appends the result as a `role: "tool"` message with `tool_call_id`, and repeats up to `OPENAI_MAX_ROUNDS` (4). Tool choice is `auto`.
- Before that loop, `cmd_summary`/`cmd_ask` call `_build_tool_results(db, run_id)` and inject the text via `{tool_results}` in the user prompt. The preloaded blocks are: **alerts** (rule_name, severity, timestamp, reason, track_id), **truck tracks** (track_id, class_name, first_seen, last_seen), **truck events** (track_id, event_type, timestamp), **all descriptions** (track_id, status, description, main_colour), and **event counts**, each with `compact_rows` totals/truncation flags.
- So the digest is fixed preloaded context; additional tool calls are model-chosen. At the time of that check the `ask` template omitted `{run_id}`, which is why those answers did not cite a run ID; the follow-up fix adds it. Traces 1 and 2 below have `tools=[none]` — those answers came entirely from the preloaded digest.

### Scenario tests (fakes only, no network or GPU, real file-backed SQLite rows)
Four tests were added to `tests/test_agent.py`, reusing `ScriptedClient` and `_EventRow` with temporary `.db` files:
1. **Midnight** — a car track first seen at `23:59:58` fires no `vehicle_at_night` alert, one at `00:00:03` does; a `23:59:55`→`00:00:09` window query returns events from both sides; the fake agent's tool result and answer cite only track 2 and `2026-11-24T00:00:03`.
2. **Truck logged** — a truck track produces exactly one `heavy_vehicle_logged` alert with severity `info`; running all rules twice does not duplicate it; the fake agent calls `get_alerts` and cites track ID 5.
3. **Questionable alert** — a truck track with a stored "brick building facade" description appears in `_build_tool_results` together with its `heavy_vehicle_logged` alert, so the digest supports qualification.
4. **Traces** — one ask writes exactly one trace row with the tools called; a cached summary writes a second row marked `cached` with zero tokens; no stored row contains the key, `Authorization` or `Bearer`.

All tests: `.venv/bin/python -m pytest -q` → `61 passed in 0.71s` (previous total 57, +4).

### Real trace check (Part 4, raw output)
`.venv/bin/python scripts/ask_agent.py ask 1 "Were any vehicles seen after midnight? Give times and track ids."` returned the 12 vehicle-at-night IDs/times as before, with `stats: model=gpt-5.4-mini prompt_tokens=3600 completion_tokens=270 latency_s=2.611 cost_usd=0.003915 running_total_usd=0.057898`.

`.venv/bin/python scripts/ask_agent.py traces 1`:
```
trace 1 time=2026-11-24T09:30:13 mode=ask cached=0 question='Were any vehicles seen after midnight? Give times and track ids.' tools=[none] cost_usd=0.003915
```
`tools=[none]` was accurate for that answer: it came from the preloaded digest, with no model-chosen tool call. The `agent_costs` running total went from $0.053983 to $0.057898. This was later found to be a systemic gap: on track 47 (a car, whose events are not in the digest) the model still made no tool call, listed only 2 items (one an alert, not an event) and said "the data cannot tell" although the DB has 10 events — see the follow-up below.
`.venv/bin/python scripts/ask_agent.py ask 2 "List every event for track 103, with timestamps and types."` returned the six stored track-103 events (appeared/entered_zone x4/left, 14:00:05-14:00:14) with `stats: model=gpt-5.4-mini prompt_tokens=3404 completion_tokens=133 latency_s=1.583 cost_usd=0.003151 running_total_usd=0.061049`; `traces 2` → `trace 2 time=2026-11-24T09:36:24 mode=ask cached=0 question='List every event for track 103, with timestamps and types.' tools=[none] cost_usd=0.003151` — no tool call, kept as evidence of the gap.
After the prompt-only follow-up fix: `ask 2 "List every event for track 47, with timestamps and types."` returned all 10 events with `stats: model=gpt-5.4-mini prompt_tokens=7588 completion_tokens=259 latency_s=2.977 cost_usd=0.006856 running_total_usd=0.070949`, and `traces 2` shows `trace 4 ... tools=[get_events_for_track]` — the model now calls the tool. All 10 event types/timestamps match `SELECT event_type, timestamp FROM events WHERE run_id=2 AND track_id=47 ORDER BY timestamp` exactly. `ask 1 "What colour was the driver's jacket?"` still answered "Run 1: ... the data cannot tell" with `trace 5 ... tools=[list_runs]`, $0.005843, inventing nothing.
Track 47 gap finding: the stored events show `left` at `14:00:10` followed by a fresh `appeared` at `14:00:12`, so the 15.0 s `loitering` alert (first_seen 14:00:03, last_seen 14:00:18) spans two track visits and includes a 2 s gap — the dwell figure is not continuous presence.

## Step 9A: Streamlit app smoke and real start (24 Nov 2026)

### Dependency
`.venv/bin/python -m pip index versions streamlit` → latest `1.67.0`. Streamlit was missing, so it was installed and pinned as `streamlit==1.67.0` in `requirements.txt`. Pip also installed streamlit's own transitive packages (altair, pandas, pyarrow, pydeck, protobuf, watchdog, uvicorn and others); no other direct dependency was added.

### Tests (no network, no OpenAI)
`.venv/bin/python -m pytest -q tests/test_app_data.py tests/test_app_ui.py` → `6 passed in 0.83s`. Five data-layer tests use a temporary database with fake rows: three tracks, eight events with a 23:59:58 → 00:00:03 midnight crossing, a `vehicle_at_night` alert only for the post-midnight track, a `heavy_vehicle_logged` info alert whose VLM description says "brick building facade", a stored summary flagging that track as questionable, costs and two traces. They assert typed rows, filters, provenance, the summary-only badge, SQLite-backed cost state, cap blocking and that the UI connection rejects writes.

The UI smoke test patches both captured `DB_PATH` values to the temp DB, removes the OpenAI key from the environment, replaces `agent_ui.generate_summary` and `agent_ui.ask` with functions that raise, then renders Overview, Alerts, Ask and Traces through `AppTest`. Each page must have no exception and its own subheader. No agent function was called.

Full suite: `.venv/bin/python -m pytest -q` → `70 passed in 1.19s`. The six new tests account for the rise from 64 to 70 (the Step 8 log separately records 61 at the end of that step).

One failure during this step: the first `AppTest.from_file("app/streamlit_app.py")` call raised `FileNotFoundError` because AppTest resolves relative paths against the calling test file, not the repo root. Fixed by passing an absolute path built from `Path(__file__).resolve().parents[1]`; reran and all tests passed.

### Real server check
Command: `.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false`, started from the repo root with output redirected to `/tmp/opencode/streamlit_step9.log`. `curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8501/_stcore/health` → `HTTP 200`. The process was killed by its PID; a subsequent socket connect to `127.0.0.1:8501` printed `port 8501 free`. No browser session was used and no OpenAI call was made.

### Not run in Step 9A
- The Ask page against real OpenAI (no question was submitted).
- The Overview "Generate summary" button (no new model call).
- Step 9B (styling / extra presentation work) was not started.

### Run command
From the repo root: `streamlit run app/streamlit_app.py`. To bind explicitly: `streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501`.

## Step 9B: visual polish, themes and screenshots (24 Nov 2026)

### Run command (final)
From the repo root: `streamlit run app/streamlit_app.py`. The app binds to `127.0.0.1:8501` in the checks below via `--server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false`. No new dependency was added in 9B; Streamlit 1.67.0 from 9A is unchanged.

### Theme mechanism
Installed Streamlit is `1.67.0`. Its native `[theme] base` setting is read at process start only, so an in-app toggle cannot use it. The implemented toggle injects one stylesheet through a same-origin component script and sets `data-theme` on `<html>`; all tokens resolve from that attribute. Plain HTML tables (`st.table`) replaced `st.dataframe` because the latter is canvas-backed and paints its own light-mode text. Native Streamlit radio dots, switch thumb and expand/collapse icons keep their own drawn glyph colours; their surrounding surfaces, borders and focus rings follow the tokens.

### Measured contrast (better-colors, WCAG 2)
Ratios were computed from the declared token pairs with a local script; AA thresholds are 4.5:1 for body text and 3:1 for UI components.

| Pair | Dark | Light |
|------|------|-------|
| Body text on page | `#E6EDF0` / `#0F1418` = 15.65:1 | `#162228` / `#F4F7F8` = 15.08:1 |
| Secondary text on panel | `#A9BBC2` / `#151C21` = 8.67:1 | `#3E525B` / `#FBFCFC` = 7.97:1 |
| Muted text on panel | `#95A9B1` / `#151C21` = 7.04:1 | `#5B7079` / `#FBFCFC` = 5.06:1 |
| Text on primary button | `#071115` / `#5BA6B8` = 6.90:1 | `#F7FAFA` / `#176C7B` = 5.77:1 |
| Warning chip | `#F0B75E` / `#332613` = 8.15:1 | `#7A4E08` / `#FFF3DB` = 6.54:1 |
| Info chip | `#A9CBE9` / `#162734` = 9.03:1 | `#1E4E76` / `#E8F1F9` = 7.64:1 |
| Success/status chip | `#9AD3B2` / `#143026` = 8.33:1 | `#1B603E` / `#E6F4EB` = 6.63:1 |
| Control border on page (UI) | `#5A707A` / `#0F1418` = 3.56:1 | `#71868E` / `#F4F7F8` = 3.54:1 |
| Focus ring on page (UI) | `#8FD3E2` / `#0F1418` = 11.10:1 | `#0F6676` / `#F4F7F8` = 6.13:1 |

All pairs pass AA. Colours were not repainted to chase a failing pair; border tokens were chosen to meet the 3:1 UI threshold before the final measurement.

### Screenshots
32 final PNGs in `docs/screenshots/`: every page (Overview, Alerts, Ask, Traces) in dark and light, with the sidebar open and collapsed, at 1440×900 and 820×900. File pattern `<page>-<theme>-<width>-<state>.png`. Largest file is 131 KB, all under the 400 KB limit. Two pre-redesign captures are kept for comparison: `before-overview-dark-1440-open.png` and `before-alerts-dark-1440-open.png`; their after images are `overview-dark-1440-open.png` and `alerts-dark-1440-open.png`. Playwright MCP drove `http://127.0.0.1:8501` headlessly on loopback only; no external site was visited.

### Tests
`.venv/bin/python -m pytest -q` → `70 passed in 1.67s` after the final code edits. The AppTest smoke test now renders all four pages twice, once with `theme_light=False` and once with `True`, still with the OpenAI key absent and both agent entry points replaced by raising fakes. No network or OpenAI call was made.

### Real failures encountered and fixed
1. `impeccable context` reported no product record; init was completed with the human's corrected product context before any UI edit.
2. First AppTest path used a repo-relative path; AppTest resolves against the test file. (9A regression check; fixed there and unchanged.)
3. The first design CSS hid `stHeader`, which also hid the sidebar expand button, making a collapsed sidebar impossible to reopen. Only toolbar actions are hidden now; collapse/re-expand was re-tested in the browser.
4. Context `<code>` chips inherited a white browser background in dark mode, making values unreadable. They are transparent now.
5. `st.dataframe` cells and then `st.table` cells kept Streamlit's dark-on-light text, leaving tables visually empty in dark mode. Tables were switched to `st.table` with token-driven cell colours and verified by screenshot.
6. Several screenshot batches captured Streamlit's rerender skeleton (empty table area) because waits matched stale cells. Final captures use fixed post-rerender waits and were visually inspected.
7. Streamlit watches the entry script, not imported modules; `app/styles.py` edits required a server restart before screenshots. Early captures were stale and were discarded.

### Not run in Step 9B
- The Ask page against real OpenAI, and the Overview Generate-summary button.
- No commit was made; screenshots and style files remain uncommitted pending approval.

### Skills actually used
`impeccable` (context, init/shape, critique, audit, polish rules), `design-taste-frontend` (Design Read and anti-slop dials), `frontend-design` (plan → self-critique, component vocabulary), `better-colors` (token ramps and measured contrast), `building-components` (tokens, states, keyboard/focus accessibility), `web-design-guidelines` (guidelines re-fetched from source and applied to the final files). The installed impeccable skill folder contains no README; its init reference was used for the ephemeral `.gitignore` entries instead, and that substitution is reported to the human.

### Impeccable critique (Assessment A, design review)
Method: two isolated sub-assessments were requested; Assessment A is complete. It scored 27/40 on the Nielsen heuristics ("Acceptable"): visibility 3, match 3, control 3, consistency 3, prevention 2, recognition 3, flexibility 2, aesthetics 4, recovery 2, help 2. Strengths recorded: on-page simulation disclosure, a real token system with AA contrast, and the alert→track→event→VLM→crop evidence chain.

**[P0] Not fixed — outside 9B scope.** `app/data.py` searches only `vehicle`/`person` track classes, while `data/demo.db` stores `car` (50) and `truck` (6). Overview therefore shows `Tracks 0` beside 22 alerts, and the Alerts provenance panel shows "No stored track row for this alert" for every alert. This is a pre-existing 9A data-layer defect, not a visual defect; fixing it requires changing `app/data.py`, which the human explicitly forbade in this step. It is reported as a blocker for the next data step.

**P1 fixes applied in presentation code only:** money-spending buttons now use a two-step confirm plus `st.spinner`, Ask failures include a next step, the alerts table drops an all-empty zone column and shows clock times in the scan table, and tuning settings moved into an expander. The critique's other P1 (bounding "Notable alerts" and shortening raw timestamps in the summary table) was left unchanged to preserve the full stored-summary data required by Step 9A.

### Impeccable critique (Assessment B, deterministic evidence)
Method: dual-agent run completed; Assessment B used the bundled detector and Playwright MCP in a fresh tab. Source scan `impeccable detect --json app/streamlit_app.py app/styles.py app/pages/*.py` returned `[]`, exit 0 — a coverage-limited clean result because the detector regex-scans source and does not parse CSS inside Python strings. URL scanning was unavailable: the host has no Chrome/Chromium/Edge/Brave binary, and the tool exited 1 with that exact error; nothing was installed. Browser evidence: both themes apply to table headers/cells, warning/status chips, timeline dots/text, crop image borders and context strip; sidebar collapse/re-expand works with the run still shown in the main area; focus-visible rings were observed on combobox, text input, switch and radio; document scrollWidth equals clientWidth at 1440 and 820 with no horizontal overflow; console errors 0. Warnings are framework-owned (Chromium permissions-policy notices and the component iframe sandbox notice).
**[P1] Fixed:** the "Flagged questionable by the stored summary" badge emitted by `st.badge()` carries inline light-mode colours and no `data-testid`, so it ignored both themes. CSS now targets `.stMarkdownBadge` with `!important` semantic tokens; table header type also now follows `--sans`. Both were verified by the final screenshot pass.
**[P3] Not fixed:** native Streamlit radio dots, switch thumb and expand/collapse icons keep their framework-drawn colours; the header toolbar button has no visible focus ring (framework-owned). The detector's `[]` is not claimed as proof of a clean UI.

### Impeccable audit (code-level)
| Dimension | Score | Key finding |
|-----------|-------|-------------|
| Accessibility | 3 | AA contrast and focus rings pass; heading levels skip h2 and crop image has no alt text. |
| Performance | 4 | No layout reads, no heavy animation, small static assets. |
| Theming | 4 | One token set drives both themes; runtime toggle needs CSS injection on Streamlit 1.67. |
| Responsive design | 3 | Usable and overflow-free at 1440 and 820; no layout work below 820 was requested. |
| Implementation integrity | 3 | Single styles module; native Streamlit glyphs and the component iframe remain outside full token control. |
| **Total** | **17/20** | **Good** |

Audit P2 findings left open: h1→h3 heading jump, missing crop alt text, session-only (not URL) page/theme/run state, and the `st.badge` orange glyph bypassing the token palette. None blocks the reviewer task.

### Unresolved findings from web-design-guidelines review
`app/pages/*.py`: headings skip h2 where `st.subheader` renders h3; `st.image` has a caption but no alt text; the Ask placeholder does not end with `…`; the Ask error text now has a next step; page, theme and run state stay in `session_state` rather than the URL; and meta strings use `·` separators as a dense ops-console convention. These are P2/P3 and were not changed in this visual-polish step.

## Frame index (frame-by-frame index) — 24 Nov 2026

### What was built
`frames` (one row per sampled frame) and `frame_objects` (one row per detection) plus three indexes, a self-committing idempotent `store_frame_index`, collection inside the tracking loop, a backfill script that reuses each run's stored tuning, four read-only row-limited queries, the `query_frames` agent tool and `scripts/query_frames.py`. No Streamlit page, no OpenAI/VLM call.

### First failure (honest record)
The first version of the backfill check required set equality between frame-index track IDs and `tracks`. Run 1 failed:

```
track_id_check: FAIL run_id=1
  only_in_frame_index=[113, 131]
  only_in_tracks=[]
  Refusing to force a match; inspect the run before using its index.
```

Both IDs had exactly one detection (frame 139 conf 0.331, frame 167 conf 0.463) and zero events: short-lived ByteTrack detections that never became confirmed tracks. Storing every detection is deliberate, so forcing equality would mean deleting real detections. The human chose option (a): require only that every confirmed ID is present, and report extras as `unconfirmed_detections`. The check was changed; no index rows were filtered.

### Result after the change (raw output)
```
DB_SIZE_BEFORE=389120
protected_data_check: PASS run_id=1
track_id_check: PASS run_id=1 confirmed_tracks_present=33 frame_ids=35
  unconfirmed_detections: count=2 ids=[113, 131]
run_id=1 frames=167 objects=1287 device=cuda:0
real 0m38.209s

protected_data_check: PASS run_id=2
track_id_check: PASS run_id=2 confirmed_tracks_present=23 frame_ids=25
  unconfirmed_detections: count=2 ids=[5, 592]
run_id=2 frames=184 objects=868 device=cuda:0
real 0m47.324s
```
The run-1 extras are the same two IDs; run 2 also passes.

### Row counts and protected data (read-only check)
| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| frames rows | 167 | 184 |
| frame_objects rows | 1287 | 868 |
| tracks | 33 | 23 |
| events | 166 | 105 |
| alerts | 22 | 11 |
| descriptions | 33 | 23 |
| run_summaries | 1 | 1 |
| agent_costs rows / traces rows (all runs) | 19 / 7 | — |

OpenAI cost total before and after: **$0.09634725** (19 rows) — unchanged. `runs` = 2 throughout; no run was created, moved or deleted.

Database size: **180224 bytes** before any backfill → **389120 bytes** after run 1 → **516096 bytes** after run 2.

### Query evidence (raw output)
- `query_frames.py --run 2 --class truck --limit 1000`: `frames with class=truck run=2 / total=176 showing=176 truncated=False`; `detections of class=truck run=2 / total=386 showing=386 truncated=False`; `distinct_track_ids=[2, 47, 103, 196, 326, 330, 360, 410, 472]`. All six expected confirmed truck IDs (2, 103, 326, 330, 360, 410) are present. The three extras (47, 196, 472) are **confirmed tracks stored with class `car`** that received at least one truck-class frame vote — the known per-frame class instability, not invented IDs.
- `query_frames.py --run 1 --from 2026-11-23T23:59:55 --to 2026-11-24T00:00:10`: `total=160`; with `--limit 1000`, 50 rows on 2026-11-23 and 110 on 2026-11-24; the first post-midnight row is `frame=51 timestamp=2026-11-24T00:00:00`, last is `frame=160 timestamp=2026-11-24T00:00:10`. The window crosses midnight with no special handling.
- `query_frames.py --run 2 --track 103`: `total=85 showing=50 truncated=True`, `showing 50 of 85`; frames 56–106, 14:00:05–14:00:10 across garage/driveway/perimeter.

### Tests
`.venv/bin/python -m pytest -q` → **83 passed in 4.27s** (was 81; +2 backfill-ID-check tests). Two tests cover the new rule: extras allowed and reported; a missing confirmed track fails the check.

### Limits observed
- The ID check cannot distinguish one object split across IDs from two unconfirmed detections; it only guarantees no confirmed track is missing.
- Per-frame class votes can disagree with a track's stored class, so a class query is a detection-level query, not a confirmed-track query.
- The backfill uses the run's stored YOLO/tracking settings; patrol waypoints, dwell time and altitude are not stored in `tuning_settings`, so current config defaults were used.

## Text-simulation demo (no models, no network)

Honesty note: this demo tests the **alert and index logic only, not vision**. No YOLO, ByteTrack, VLM, LM Studio or OpenAI call is made; objects are hand-labelled in JSONL.

### Tests
`.venv/bin/python -m pytest -q` → **96 passed in 4.05s** (was 83; +13 in `tests/test_text_demo.py`). Coverage: parser valid/invalid input (missing `time`, unknown class, bad time format, out-of-order frames), one track per label, midnight and truck alerts, quiet negative control, idempotency, frame queries by window/class/track, stored description, and parsing of the three committed scenario files.

### Expected versus actual alerts
| Scenario | Expected | Actual |
|---|---|---|
| midnight_gate | `person_at_night` 1, `loitering` 1, others 0 | identical |
| daytime_truck | `heavy_vehicle_logged` 1, two appearances of one truck track | identical; `loitering` also fired (see limits) |
| quiet_daytime | no alerts | no alerts |

Raw output from `scripts/run_text_demo.py --db data/text_demo.db`:
```
run_id: 1 source: text-simulated:midnight_gate frames=6 objects=4 tracks=1 events=3
  track=1 class=person first_seen=2026-11-24T00:00:00 last_seen=2026-11-24T00:03:00
  alerts: vehicle_at_night 0, person_at_night 1, loitering 1, heavy_vehicle_logged 0
run_id: 2 source: text-simulated:daytime_truck frames=6 objects=5 tracks=1 events=6
  track=1 class=truck first_seen=2026-11-24T12:00:00 last_seen=2026-11-24T16:01:00
  alerts: vehicle_at_night 0, person_at_night 0, loitering 1, heavy_vehicle_logged 1
run_id: 3 source: text-simulated:quiet_daytime frames=3 objects=2 tracks=1 events=3
  track=1 class=person first_seen=2026-11-24T12:00:00 last_seen=2026-11-24T12:00:03
  alerts: vehicle_at_night 0, person_at_night 0, loitering 0, heavy_vehicle_logged 0
```
Event types: run 1 = 1 appeared / 1 entered_zone / 1 left; run 2 = 2 / 2 / 2; run 3 = 1 / 1 / 1. Text DB totals: runs 3, tracks 3, events 12, frames 15, frame_objects 11, alerts 4.

### Frame queries (raw output)
`scripts/query_frames.py --db data/text_demo.db --run 1 --class person --limit 1000` → `total=4 showing=4 truncated=False`, frames 3–6, `distinct_track_ids=[1]`.
`scripts/query_frames.py --db data/text_demo.db --run 1 --from 2026-11-23T23:59:00 --to 2026-11-24T00:03:00 --limit 1000` → `total=6 showing=6 truncated=False`, frames 1–6, crossing midnight without special handling. Example stored frame row: `frame=3 timestamp=2026-11-24T00:00:00 zone=gate detections=1 description='A person in a dark jacket is now standing at the main gate.'`

### Idempotency (real DB)
Before re-running `midnight_gate`: `runs 3, tracks 3, events 12, frames 15, frame_objects 11, alerts 4`; after: identical, and the script again reported `run_id: 1`.

### Protected data (read-only check)
`data/demo.db`: runs 2; tracks 33/23; alerts 22/11; frames 167/184; cost total `$0.09634725`. Unchanged from the baseline measured before this work.

### Failure found before implementation (shown, not worked around)
Probing the unmodified rule on a temporary in-memory DB printed:
```
night_window: 0 5
stored first_seen: ('2026-11-23T23:59:00', '2026-11-24T00:03:00')
person_at_night alerts for 23:59->00:03 person: 0
```
`person_at_night` checks only the hour of `first_seen`, so a person present from 23:59 cannot fire it. The human chose to keep the gate empty until 23:59:59 and first see the person at 00:00:00. Future work: check presence anywhere in the night window instead of first-seen hour.

### Other limits
- The truck's `loitering` alert spans both visits (12:00→16:01) including the four-hour gap; the existing rule measures `last_seen - first_seen` across all events, so this is not continuous presence. Left unchanged.
- Hand-written labels are certainties (`confidence=1.0`) and boxes are placeholders `(0,0,0,0)`; neither is a real detection. No crop or VLM call is made.
- The Streamlit app reads only `demo.db`, so these text runs are not visible there in this step. Step 10 and Step 11 were not started.

## Step 10: Scalability measurements (24 Nov 2026)

Machine (measured this session): `NVIDIA GeForce RTX 5050 Laptop GPU`, **8151 MiB** VRAM (4258 MiB in use at measurement), driver **610.74**, compute capability (12,0) from earlier checks; WSL-visible RAM **7.6 GiB** total, 4.1 GiB free. Venv Python 3.12.13. Device reported by the tracker: `cuda:0`.

Settings used (unchanged defaults from `src/config/settings.py`): `yolo11s.pt`, `imgsz=1280`, `conf=0.10`, `SAMPLE_RATE_FPS=10`, tracker `configs/bytetrack_0.9.yaml`, `MIN_TRACK_FRAMES=2`, `LOST_FRAMES=10`. No setting, tracker, YOLO, alert, VLM or agent code was changed.

Command: `.venv/bin/python scripts/benchmark.py`. Raw output: `/tmp/opencode/step10/results.json`. All writes went to scratch DBs under `/tmp/opencode/step10`; `data/demo.db` was opened only with `mode=ro`.

### Per-stage timing on the two real clips

Mean seconds per sampled frame, with median and warm-up note. "First" includes model/tracker load.

| Clip | Sampled frames | Decode (seek+read) mean / median / first | YOLO+ByteTrack mean / median / first | Frame-index build (in-loop) | Event build (in-loop) |
|------|---------------|------------------------------------------|--------------------------------------|-----------------------------|-----------------------|
| drone_intersection.mp4 | 167 | 152.925 / 150.858 / 195.281 ms | 40.848 / 21.033 / **3209.722** ms | 0.010 ms | 0.013 ms |
| drone_truck.mp4 | 184 | 186.017 / 181.213 / 147.497 ms | 24.007 / 21.459 / 93.098 ms | 0.010 ms | 0.010 ms |

Steady-state YOLO+ByteTrack (excluding the first frame): **21.758 ms** (intersection) and **23.629 ms** (truck) — consistent with the earlier ~64 ms figure once that earlier number's warm-up and denser first run are accounted for. Decode dominates by roughly 7–8×.

Note: the decode means here (152.9 / 186.0 ms per sampled frame) and the seek figures in the decode-comparison table below (176.0 / 202.6 ms) come from two separate benchmark passes in the same session, which is why they differ. The one-hour extrapolation uses about 170 ms per frame, which sits between the two sets.

DB write stages, measured as one transaction per run and divided by sampled frames:

| Clip | Event write total / per frame | Frame-index write total / per frame | Scratch DB size |
|------|-------------------------------|-------------------------------------|-----------------|
| drone_intersection.mp4 | 0.002196 s / 0.013 ms | 0.006192 s / 0.037 ms | 335872 bytes |
| drone_truck.mp4 | 0.002632 s / 0.014 ms | 0.004235 s / 0.023 ms | 266240 bytes |

The in-loop build and the single end-of-run DB writes are effectively free (<0.05 ms/frame); decode and YOLO are the whole cost. Annotated-video drawing was deliberately not timed (it is presentation, not a pipeline stage).

### Throughput and ratio to real time

Sum of measured stages (decode + YOLO + both builds + both DB writes):

| Clip | Video duration | Measured total | Throughput | Ratio to real time |
|------|---------------|----------------|------------|--------------------|
| drone_intersection.mp4 | 16.633 s | 32.372 s | 5.159 sampled frames/s | 0.514 (about 2× slower than real time) |
| drone_truck.mp4 | 18.333 s | 38.655 s | 4.760 sampled frames/s | 0.474 (about 2.1× slower than real time) |

### Decode comparison: frame-index seek vs sequential reading

Sequential reading was possible in the benchmark script only; no pipeline code changed. Both methods collected the identical set of sampled frames (167 / 184), asserted equal.

| Clip | Source frames | Seek total / ms per sampled frame | Sequential total / ms per sampled frame | Sequential faster by |
|------|--------------|-----------------------------------|------------------------------------------|----------------------|
| drone_intersection.mp4 | 499 | 29.386 s / 175.962 ms | 0.707 s / 4.232 ms | 41.6× |
| drone_truck.mp4 | 440 | 37.275 s / 202.580 ms | 0.685 s / 3.724 ms | 54.4× |

Finding: repeated `CAP_PROP_POS_FRAMES` seeks are the bottleneck; decoding every source frame once and keeping only the sampled indices is 40–54× faster. The pipeline uses seeking, so this is a measured, unimplemented improvement — **not** applied here.

### Frame-index growth from the real rows

Read-only from `data/demo.db`; the same rows were re-inserted into a scratch DB through the production `store_frame_index` write path so row layout and indexes match. Bytes are incremental file-size deltas (includes index pages).

| Run | Frames | frame_objects | Rows / sampled frame | Incremental bytes | Bytes / sampled frame | Bytes / object |
|-----|--------|---------------|----------------------|-------------------|-----------------------|----------------|
| 1 | 167 | 1287 | 7.707 | 184320 | 1103.7 | 143.2 |
| 2 | 184 | 868 | 4.717 | 126976 | 690.1 | 146.3 |

`SUM(detection_count)` equalled the `frame_objects` row count for both runs (1287, 868), so the index is self-consistent.

### Synthetic query latency (NOT real footage)

**SYNTHETIC, not measured on video.** Generated scratch DBs with 5 detections per frame, 50 cyclic track IDs, 5 classes and 10 fps timestamps; DB sizes 4.1 MB / 42.5 MB / 430.8 MB. Each query is the existing `src.storage` function, median of 5 warmed calls, limit 50.

| Frames (objects) | Time window no idx / idx | Class=truck no idx / idx | Track=42 no idx / idx |
|------------------|--------------------------|--------------------------|------------------------|
| 10k (50k) | 1.224 / **0.030** ms | 10.464 / 9.774 ms | 3.895 / **1.534** ms |
| 100k (500k) | 12.972 / **0.030** ms | 112.764 / 111.281 ms | 39.350 / **16.171** ms |
| 1M (5M) | 130.182 / **0.029** ms | 1161.863 / 1155.677 ms | 389.911 / **173.145** ms |

Honest reading: the `(run_id, timestamp)` index makes time windows flat (~0.03 ms); the track index roughly halves per-track lookups. The class query barely improves because `COUNT(DISTINCT frame_index)` over ~1M matching objects dominates regardless of the index. No 10M-frame run was attempted.

### VLM cost scaling (stored numbers)

| Run | Confirmed tracks | Sampled frames | VLM calls attempted | Too-small skipped | Calls / track | Calls / 100 frames | Stored latency total / mean per call |
|-----|------------------|----------------|---------------------|-------------------|---------------|--------------------|--------------------------------------|
| 1 | 33 | 167 | 33 | 0 | 1.000 | 19.76 | 40.61 s / 1.230 s |
| 2 | 23 | 184 | 11 | 12 | 0.478 | 5.98 | 13.72 s / 1.247 s |

The stored per-call latency sums (40.61 / 13.72 s) are slightly lower than the wall totals recorded earlier in Step 6 (42.37 / 14.59 s); the difference is DB-write and parse overhead outside `latency_ms`, not a contradiction.

Fragmentation multiplier: one VLM call per *new confirmed track ID*. Run 1's 33 IDs produced 33 calls; run 2's 23 IDs produced at most 23, of which 11 were made and 12 skipped as too small. A tracker split that turns one physical object into *k* IDs therefore multiplies VLM calls by *k* (arithmetic: 1 object → 1 ID → 1 call; 1 object → 3 IDs → 3 calls), while doubling the sample rate does not add calls for an already-described track.

Five calls timed live on stored crops with the configured `google/gemma-4-e4b` (LM Studio reachable): latencies 10471.1, 1630.5, 1618.9, 1462.8, 1610.7 ms — mean **3358.8 ms**, median 1618.9 ms, valid JSON **5/5**. The first call includes a cold model reload; steady state is ~1.5–1.6 s, close to the stored ~1.23 s mean.

### Agent cost per question (stored tables, read-only)

| Source | Rows | Avg prompt tokens | Avg completion tokens | Avg cost USD | Avg latency |
|--------|------|-------------------|-----------------------|--------------|-------------|
| `agent_costs` all calls | 19 | 3741.9 | 503.2 | 0.00507091 | 2.982 s |
| `agent_costs` command=`ask` | 15 | 3789.1 | 305.4 | 0.00421615 | 2.512 s |
| `agent_traces` all answers | 7 | 6120.3 | 324.9 | 0.00605207 | 3780.3 ms |
| `agent_traces` mode=`ask`, uncached | 7 | 6120.3 | 324.9 | 0.00605207 | 3780.3 ms |

Totals: 71097 prompt + 9561 completion tokens, **$0.09634725** across 19 cost rows — unchanged. The two tables answer different questions: `agent_costs` counts model calls, `agent_traces` counts user-visible answers (its higher prompt average reflects the tool-call rounds in the tracked answers).

### Extrapolations — EXTRAPOLATION, NOT MEASURED

All arithmetic below uses the measured numbers above; none of it was run.

- **One hour at 10 fps sampling = 36000 sampled frames.** Mean measured cost/frame = (32.372 + 38.655) / (167 + 184) = 0.2020 s. 36000 × 0.2020 = **7272 s ≈ 2.02 h** of CPU/GPU wall time (ratio 3600/7272 = 0.50, i.e. ~2× slower than real time). The per-frame average includes the 3.2 s first-frame model warm-up, so this estimate is slightly pessimistic. With the *unimplemented* sequential decode (~4 ms instead of ~170–186 ms): per frame ≈ 0.2020 − 0.170 + 0.004 = 0.036 s → 36000 × 0.036 = **1296 s ≈ 21.6 min**.
- **VLM calls for that hour, by observed density:** run-1 rate 19.76 calls/100 frames → 36000 × 0.1976 = 7114 calls × 1.230 s = 8750 s ≈ 2.43 h; run-2 rate 5.98/100 → 2153 calls × 1.247 s = 2685 s ≈ 45 min. The large spread is the real message: call count depends on track fragmentation and crop size, not on frame count.
- **Index size for that hour:** 36000 frames × 1103.7 B/frame (run-1 density) = 39.7 MB; at run-2 density 36000 × 690.1 = 24.8 MB. Both exclude alerts/descriptions.
- **100 agent questions:** 100 × $0.00421615 = **$0.42** (ask-only average) with 100 × 2.512 s = 251 s of model latency, assuming the bounded digest keeps prompt size roughly flat as the video grows — an assumption, not a measurement.
- **Ten drones, each 10 fps sampled = 100 sampled frames/s demanded.** Measured single-stream capacity ≈ 4.9 frames/s → 100 / 4.9 ≈ **20.4 GPU-equivalents** on this hardware. With sequential decode (~27.8 frames/s/stream) → 100 / 27.8 ≈ **3.6 GPU-equivalents**. Serving ten streams on one GPU is unmeasured and would contend for VRAM (8151 MiB total).

### Lift-ready: Scalability

On an RTX 5050 Laptop (8 GB VRAM), the prototype processes a sampled frame in about **0.20 s end to end**: 0.15–0.19 s is 4K/1080p frame decode by seeking, 0.021–0.024 s is steady-state YOLO+ByteTrack, and SQLite writes are under 0.05 ms per frame. End-to-end throughput is **5.16 and 4.76 sampled frames/s** (0.51× and 0.47× real time). Seeking is pathological: reading the video sequentially and discarding unneeded frames is **42–54× faster** (0.707/0.685 s versus 29.4/37.3 s per clip) — the single biggest unimplemented win. The index adds ~0.7–1.1 KB per sampled frame; the `(run_id, timestamp)` index keeps synthetic time-window queries at ~0.03 ms even at 1 M frames (130 ms without it), while a class count over 5 M detections stays around 1.16 s with or without indexes. VLM cost scales with *new tracks*, not frames: 33 calls for 33 tracks versus 11 calls for 23 tracks (12 crops too small), at ~1.2 s per call. The agent averages 3.8k prompt + 305 completion tokens, **$0.0042 and 2.5 s per question**; total OpenAI spend remains $0.09634725. One hour of footage would take about 2.0 h to process as written, or ~22 min if sequential decoding were adopted; ten simultaneous drones would need roughly 20 GPUs at current throughput. All hour/ten-drone figures are extrapolations, not measured.

## Step 10: VLM comparison (24 Nov 2026)

LM Studio was reachable and returned 11 model names (names only, no install/download):

- `google/gemma-4-e4b`, `qwen/qwen2.5-coder-14b`, `qwen/qwen3.5-9b`, `google/gemma-3-12b`, `nvidia/nemotron-3-nano-4b`, `qwen2.5-coder-7b-instruct`, `qwen/qwen2.5-vl-7b`, `google/gemma-4-e2b`, `mistralai/ministral-3-3b`, `deepseek/deepseek-r1-0528-qwen3-8b`, `text-embedding-nomic-embed-text-v1.5`.

Each was probed with one stored crop; five returned valid JSON and count as available vision models: `google/gemma-4-e4b`, `google/gemma-4-e2b`, `google/gemma-3-12b`, `qwen/qwen3.5-9b`, `qwen/qwen2.5-vl-7b`. The other six are coder/reasoning/embedding listings and were not compared. No accuracy score is claimed — LM Studio provides no ground truth.

Fixed set: 10 run-2 tracks — **2, 36, 64, 103, 106, 233, 326, 330, 353, 410** — five YOLO truck tracks (2, 103, 326, 330, 410) and five car tracks, including the required 103/326/410. All 10 crop files exist; the "By eye" column was first drafted by the assistant from the crop files, then checked by the human (the project owner) against the crops. It is one person's visual reading, not independent ground truth, and no accuracy score is claimed.

Per-model summary (10 calls each, same crops, stateless, temperature 0, JSON schema):

| Model | Mean latency | Median latency | Valid JSON |
|-------|--------------|----------------|------------|
| google/gemma-4-e4b | 1542.5 ms | 1459.7 ms | 10/10 |
| google/gemma-4-e2b | 2667.0 ms | 1202.5 ms | 10/10 |
| google/gemma-3-12b | 60157.0 ms | 41251.2 ms | 10/10 |
| qwen/qwen3.5-9b | 15721.3 ms | 12613.8 ms | 10/10 |
| qwen/qwen2.5-vl-7b | 3286.0 ms | 1240.6 ms | 10/10 |

Latency includes model loading on the first call of each model; that is why the means for the larger models are far above their medians (for example gemma-3-12b's first call took 171405 ms). All five produced valid JSON for all ten crops.

Per-track descriptions (verbatim, truncated only for table width):

| Track | YOLO class | gemma-4-e4b (chosen) | gemma-4-e2b | gemma-3-12b | qwen3.5-9b | qwen2.5-vl-7b | By eye (no ground truth) |
|-------|-----------|----------------------|-------------|-------------|------------|---------------|--------------------------|
| 2 | truck | rectangular white object, nature unclear | blurry large white rectangular object | white-roofed building with writing/artwork | white-roofed building with blue mural | large building with blue graffiti | white box truck with blue cab, "Linda" branding — clearly a truck |
| 36 | car | indistinct blue/grey patch among foliage | small body of blue water | small area of blue water in greenery | blue/grey vehicle on grass | small bird with blue patch | blurry dark object in foliage with a blue patch; vehicle not certain |
| 64 | car | banner featuring text | cylindrical structure with graffiti | cylinder covered in graffiti | cylindrical object with multicolour graffiti | water tank with blue/white text | painted cylindrical tank/drum with text — not a car |
| 103 | truck | blurry storage container | red truck | red trailer with markings | red food trailer | red truck | red-and-white box body on wheels — truck/trailer |
| 106 | car | shape that appears to be a vehicle | vehicle on a road | dark-coloured vehicle on paved surface | dark shape, appears to be a car | blurred vehicle on road | blurry white/blue vehicle on road — likely a car |
| 233 | car | blurry white vehicle on paved surface | white van on road | white van parked on a street | white van on road | white van parked on street | white van/minibus on road |
| 326 | truck | three small blue/red blocks | two rectangular blue/red blocks | several blue/orange rectangular containers | four shipping containers | stacked shipping containers | four coloured rectangular containers — not a truck |
| 330 | truck | small red vehicle on dusty surface | red truck | blurry red truck with blue bed and equipment | red pickup carrying small excavator | red truck with blue cargo area | red compact flatbed/pickup carrying a blue object — small truck |
| 353 | car | blurry blue vehicle on paved surface | blue vehicle | blurry blue vehicle next to white vehicle | blue vehicle | small blue vehicle | blue minivan/car on road |
| 410 | truck | multi-storey building with windows | brick building facade | blurry grey vehicle (wrong) | brown building structure | building section with windows | narrow alley between brick buildings — not a truck |

Gemma-4-e4b versus YOLO class label, per track (keyword rule stated explicitly, not an accuracy metric: "agreement" = the stored description says vehicle/truck/van/car for a YOLO vehicle class; silence or a non-vehicle noun = "disagreement"):

| Track | YOLO | Stored Gemma description says vehicle? | Result |
|-------|------|----------------------------------------|--------|
| 2 | truck | no (rectangular object) | disagreement |
| 36 | car | no (patch among foliage) | disagreement |
| 64 | car | no (banner) | disagreement |
| 103 | truck | no (storage container) | disagreement |
| 106 | car | yes ("appears to be a vehicle") | agreement |
| 233 | car | yes ("white vehicle") | agreement |
| 326 | truck | no (blocks) | disagreement |
| 330 | truck | yes ("small red vehicle") | agreement at vehicle level |
| 353 | car | yes ("blue vehicle") | agreement |
| 410 | truck | no (building) | disagreement |

Four of ten agree at the vehicle-vs-non-vehicle level; the chosen VLM under-identifies large trucks as buildings/containers. This is a qualitative keyword reading of stored text, **not** a measured accuracy number.

### Lift-ready: VLM comparison

Five vision models were already available in LM Studio and were compared on the same 10 stored crops. All returned valid JSON 10/10 times. Three models are close in steady state (about 1.2–1.5 s median: `gemma-4-e4b` 1.46 s, `gemma-4-e2b` 1.20 s, `qwen2.5-vl-7b` 1.24 s), while `qwen3.5-9b` (12.6 s) and `gemma-3-12b` (41.3 s) are far slower; the larger models' means are inflated by one-time loading. By eye, Gemma-4-e4b called only 4/10 crops vehicles, missing the clearly blue-cabbed truck (track 2) and the red box truck (103) as vehicles, and correctly rejected the shipping containers (326) and the building (410). Qwen3.5-9b and Qwen2.5-VL-7b named track 103 a truck and track 326 containers, but Qwen2.5-VL-7b called the truck crop 2 a "building" and track 36 a "bird"; gemma-3-12b also read track 2 as a building and uniquely called the building crop 410 a "grey vehicle". No ground truth exists, so these are observations rather than accuracy scores; the operational conclusion is unchanged: YOLO supplies counts/classes and VLM text must be treated as fallible context.

### Tests and protected data

- `.venv/bin/python -m pytest -q` → **98 passed** (was 96; +2 in `tests/test_benchmark.py`). Two no-model tests assert the sequential decoder targets exactly the same frames as the seek sampler and never exceeds the clip duration.
- `data/demo.db` read-only check after all measurements: runs **2**; tracks **33/23**; alerts **22/11**; frames **167/184**; `agent_costs` **19** rows, total **$0.09634725**. Unchanged from baseline.
- No OpenAI call was made; no `.db`, crop or video was written inside the repo; no dependency was added; no external site was visited.

## UI cleanup before Step 11 (24 Nov 2026)

### What changed
- `app/data.py`: `_TRACK_CLASSES` is now `settings.TRACKED_CLASSES` (concrete tracker classes), replacing the abstract `vehicle`/`person` names that never matched `tracks.class_name`.
- `app/styles.py`: the warning `<p>` inherits Streamlit's alert colour; the primary-button `<p>` uses `--on-accent`; both sidebar arrow icons use `--text`, and the stale `stSidebarCollapseButton` selector now also covers Streamlit 1.67's real collapse button.
- `tests/conftest.py` / `tests/test_app_data.py`: the shared fixture stores realistic `car`/`truck` classes and one new regression test pins concrete-class counting and provenance lookup.

### Verification (no model calls)
- `.venv/bin/python -m pytest -q` → **99 passed in 4.00s** (was 98; +1 regression test). No OpenAI key was needed; agent entry points were not called.
- Read-only `app.data.run_counts` after the fix: `run 1 -> RunCounts(tracks=33, events=166, alerts=22)`, `run 2 -> RunCounts(tracks=23, events=105, alerts=11)`.
- Browser check on the read-only server (dummy key only for button gating; **Confirm question** and **Confirm spend** were never clicked): Overview run 1 shows Tracks 33 / Events 166 / Alerts 22; run 2 shows Tracks 23 / Events 105 / Alerts 11.
- Computed colours after the fix, with the Step 9B WCAG method:
  - Ask label: light `#F7FAFA` on `#176C7B` = **5.77:1**, dark `#071115` on `#5BA6B8` = **6.90:1** (was 1.35:1 / 1.39:1).
  - Confirmation line inherits Streamlit's own warning text: with `prefers-color-scheme: dark`, measured `#FBD54F` on `#302917` = **10.12:1**; the declared light pair `#956B0F` on `#FFFAE5` = **4.57:1**. Both clear AA 4.5:1.
  - Sidebar arrows use `--text`: light `#162228` on `#F4F7F8` = **15.08:1**, dark `#E6EDF0` on `#0F1418` = **15.65:1**. Measured icon colours matched those tokens in both injected themes, including the previously failing light-injected/dark-base case (was 1.06:1).
- `data/demo.db` mtime and size were unchanged (516096 bytes, 14:45:49 UTC); the temporary server was stopped and port 8501 was free afterwards.

### Not run
- No demo video, PDF report, README or other Step 11 work; no commit. Changes are left for human inspection.

## Demo CLI duplicate guard (25 Nov 2026)

### What changed
- `scripts/run_tracking.py`: `--video`, `--db`, `--force`; read-only duplicate check before any model/video/DB work; model imports deferred until after parsing; `run_id` and the annotated-video, crops and DB paths printed at the end.
- `tests/test_run_tracking_cli.py`: three no-model tests using a temporary DB.

### Verification (no model calls)
- Import timing of the unmodified script: **2.216 s** (`python -c "import run_tracking"`).
- `/usr/bin/time -f 'elapsed_seconds: %e' python scripts/run_tracking.py --help` → usage with `--video`, `--db`, `--force`, exit 0, **0.05 s**. The test also passes explicit temp `--db`/`--video` paths with `--help` and asserts neither file is created.
- `.venv/bin/python -m pytest -q` → **102 passed in 3.79s** (was 99; +3).
- Read-only `data/demo.db` before and after: **524288 bytes**, runs **2**, tracks **33/23**. Gate on the two stored `source_video` values returned `[1]` and `[2]`; on a new filename it returned `[]`. Size unchanged after every check.
- `git diff --stat` at the end: `scripts/run_tracking.py`, `docs/ai-tools-log.md`, `docs/design-decisions.md`, `docs/testing.md` changed; new untracked `tests/test_run_tracking_cli.py`.

### Not run
- No pipeline run, no benchmark, no OpenAI/LM Studio call, no write to `data/demo.db`, no commit. The untracked `docs/report.md` and `docs/report_assets/` were left untouched.

