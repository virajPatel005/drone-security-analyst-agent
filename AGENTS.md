# AGENTS.md - Drone Security Analyst Agent

## Mission
Build a working prototype of a Drone Security Analyst Agent for a hiring assignment.
Hard deadline: submission by 25 Nov 2026. Speed matters, but a working simple system beats an ambitious broken one.

## Read first, every session
1. docs/PROJECT_CONTEXT.md - what we are building and why
2. docs/BUILD_PLAN.md - step list and current status

## How we work
- The human (Viraj) plans each step with an external assistant and pastes prompts to you. Do ONLY the step requested. Do not build ahead.
- Ask before adding any new dependency, and say why it is needed.
- After finishing a step, stop and report using the format below. Do not start the next step.
- If something is ambiguous or a requirement conflicts with the docs, ask instead of guessing.
- Never mark something as working unless you actually ran it. Say what you ran and what the output was.

## Report format after each step
1. What changed (files created or edited)
2. What you ran to verify, and the real output
3. What failed or is uncertain (be honest)
4. Assumptions you made
5. Suggested next step (do not do it)

## Architecture rules
- Three layers only: Perception (YOLO + ByteTrack), Description (local VLM), Reasoning (OpenAI).
- The OpenAI model reads stored TEXT only. Never send images to OpenAI in the main pipeline.
- The VLM is called only when a NEW track appears (on that object's crop), never on every frame. Each VLM call is stateless: no chat history.
- Everything is stored in SQLite. Agent answers must come from stored data, not from re-asking the VLM.
- Core logic must run without GPU, LM Studio or OpenAI when tests use fakes. Keep model calls behind small interfaces so they can be mocked.

## Code conventions
- Python, type hints, small modules, one responsibility each.
- Docstrings and comments explain WHY a decision was made, not only what the code does.
- All settings (paths, URLs, thresholds, model names, simulated start time) live in src/config, never hardcoded.
- Secrets only in .env (gitignored). Never print or commit keys or tokens.
- Use logging, not print, in library code.

## Environment facts
- Development is inside WSL Ubuntu. Repo lives in the Linux filesystem, not /mnt/c.
- LM Studio runs on Windows and serves an OpenAI-compatible API. WSL may not reach it at localhost, so the base URL must be configurable and tested early.
- GPU is an RTX 5050 laptop GPU (about 8 GB VRAM), a very new card. Confirm PyTorch actually uses CUDA before trusting any speed numbers.
- OpenAI budget is about 8 USD total. Keep prompts compact and use a cheap mini model.

## Documentation duty (graded)
- Log every decision and its reason in docs/design-decisions.md.
- Log how AI tools helped in docs/ai-tools-log.md.
- Record real measurements (seconds per frame, counts, failures) in docs/testing.md as we go. Honest failures are valuable.
- Do not commit the assignment PDF or any confidential assignment text.

## Definition of done for a step
Code runs, has at least one test, is documented in the relevant docs file, and is committed with a clear message.