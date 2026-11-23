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
