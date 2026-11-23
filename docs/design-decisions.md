# Design Decisions

## Decisions

| Decision | Reason |
|----------|--------|
| Use Python 3.12.13 for `.venv` instead of system Python 3.14.4 | `python3 -m venv` failed because `ensurepip` is missing for 3.14.4; 3.12.13 created a working venv. |
| Pin `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130` from the PyTorch CUDA index | `nvidia-smi` reports CUDA UMD 13.3 and the RTX 5050 is compute capability 12.0 (Blackwell); cu130 supports this and was verified by a CUDA matmul. |
| Keep the LM Studio URL in `.env`, `.env.example`, and `src/config/settings.py` | WSL cannot reach Windows `localhost:1234`; the Windows host IP `172.19.144.1` worked, and the URL may change after a restart. |
| Try `localhost:1234` only as a configured fallback | It fails from WSL with connection refused in this environment, but keeping the fallback makes the check portable. |
| Use `google/gemma-4-e4b` with 512 max tokens for image descriptions | With 128 tokens the reply was empty because the model spent them on `reasoning_content`; 512 produced a valid one-sentence description. |
| Auto-download `yolo11n.pt` rather than committing it | `*.pt` is gitignored, and Ultralytics downloads the weights on first run. |
| Default to `yolo11n` at `imgsz=1280`, `conf=0.25` | On `multiple_cars.jpg` it found the most cars for about 4 ms extra over `imgsz=640` (19.13 ms vs 15.22 ms) and looked most complete in visual review; `yolo11s` was not better on this image. To be re-checked on the real demo video in Step 3 against a manually counted short clip. |
| Default sampling rate: 5 frames per second of video | Tracking needs consecutive frames close together; 1 fps risks broken track IDs. The VLM still runs only on new tracks, so the extra frames do not increase VLM calls. Configurable via `SAMPLE_RATE_FPS`. |
| Simulated clock is a pure `start + timedelta(video_time)` function | No wall-clock dependence makes tests deterministic, and midnight rollover happens naturally when the offset crosses a day. |
| Use one shared `FrameRecord` for video and text modes | Downstream stages never branch on source; text mode keeps `image=None` and fills `description`. |
| Text scripts advance 60 s per `Frame N` and can name a location | A 7-line script crosses midnight while still reusing the clock, altitude and waypoint coordinates; a named location overrides the cycling route. |
| Default patrol dwell: 3 s per location | The 14 s demo then visibly cycles through gate, garage, driveway and perimeter instead of staying on gate. |
| Use Ultralytics ByteTrack through `model.track(persist=True)` | ByteTrack is already shipped with Ultralytics, so no separate tracking service is needed. `persist=True` keeps tracker IDs across the sampled frames. |
| Require 2 consecutive hits before emitting `appeared` | The real video produces many one-frame detections. A two-frame minimum removes those flickers from the event stream without delaying real objects much at 5 fps. |
| Expire tracks after 10 missing sampled frames (2 s at 5 fps) | This tolerates brief detection gaps. If the same tracker ID reappears after expiry, it becomes a fresh appearance rather than being merged with the earlier visit. |
| Treat `zone` as the drone telemetry location, not an image region | The assignment asks for events in drone patrol zones. The consequence is that every long-lived track emits a new `entered_zone` when the simulated patrol route changes, even if the object has not physically moved between image regions. |
| Map `person` to `person` and road vehicles to `vehicle`; ignore other COCO classes | These are the two object categories the analyst needs. Stop signs and other classes are filtered out before event generation. |
| Timestamp `left` at the last seen time and measure dwell to that time | Dwell should end when the object was last observed, not when the loss threshold happened to expire, so the gap is not counted as presence. |
| Write annotated video at 5 fps, maximum width 720 | The source is 2160x3840 portrait and awkward to review. 720 px wide keeps the demo small and readable, and 5 fps matches the sampled input. |
| Keep all tracked objects in the inspection summary, but only confirmed tracks in events | Flicker IDs stay visible for debugging without polluting the event stream that later stages will consume. |
| Pin `lap==0.5.14` for ByteTrack | ByteTrack needs the `lap` linear-assignment package. Ultralytics auto-installed `lap==0.5.14` on the first Step 3 run, so it is now pinned for reproducible installs instead of relying on that side effect. |
| Treat zone dwell as resetting on every patrol change | The simulated drone location changes every 3 s with the current patrol dwell, so `zone_entered_at` resets each time even when the object has not moved. Step 5 should define loitering from the track's time in its current zone, and the demo should use a longer patrol dwell to reduce churn. |
