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
