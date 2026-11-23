# AI Tools Used

## Tool Usage

| Step | Tool | Purpose | Result |
|------|------|---------|--------|
| Step 1 | opencode assistant | Selected CUDA wheel, wrote check scripts and config test, interpreted failures | CUDA/YOLO/LM Studio checks passed; 2 tests passed |
| Step 2 | opencode assistant | Wrote sampler, clock, telemetry, shared record, text parser, tests, docs | 17 tests passed; sampled 15 frames from the real 14 s video |
| Step 3 | opencode assistant | Wrote YOLO/ByteTrack adapter, event state machine, annotated video writer, runner script and event tests; reviewed a contact sheet | 24 tests passed; 71 frames tracked on `cuda:0`; 52 observed tracks, 39 confirmed appearances; annotated demo written |
