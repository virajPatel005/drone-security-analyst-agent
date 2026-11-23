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
