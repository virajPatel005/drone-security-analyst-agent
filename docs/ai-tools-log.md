# AI Tools Used

## Tool Usage

| Step | Tool | Purpose | Result |
|------|------|---------|--------|
| Step 1 | opencode assistant | Selected CUDA wheel, wrote check scripts and config test, interpreted failures | CUDA/YOLO/LM Studio checks passed; 2 tests passed |
| Step 2 | opencode assistant | Wrote sampler, clock, telemetry, shared record, text parser, tests, docs | 17 tests passed; sampled 15 frames from the real 14 s video |
| Step 3 | opencode assistant | Wrote YOLO/ByteTrack adapter, event state machine, annotated video writer, runner script and event tests; reviewed a contact sheet | 24 tests passed; 71 frames tracked on `cuda:0`; 52 observed tracks, 39 confirmed appearances; annotated demo written |
| Step 3 tuning | opencode + external assistant | Ran 6 benchmark variants (5/10/15 fps × conf 0.10/0.25 + yolo11n/yolo11s with match 0.8/0.9); measured ID fragmentation, track duration, van coverage, low-conf IDs; logged results in docs/testing.md | New defaults: yolo11s conf 0.10, 10 fps, match 0.9; van fully detected at high conf; decode dominates at ~350 ms/frame |
