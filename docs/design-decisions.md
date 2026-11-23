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
