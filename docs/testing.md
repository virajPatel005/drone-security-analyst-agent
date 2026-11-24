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
