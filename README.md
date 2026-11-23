# Drone Security Analyst Agent

Prototype of a drone security analyst agent: simulated telemetry plus video frames, object/event logging, rule-based security alerts, a frame-by-frame SQLite index, and an agent that summarizes video and answers follow-up questions.

## Overview

## Architecture

## Setup (WSL Ubuntu)

Requires Python 3.12 and an NVIDIA driver with CUDA 13 support (tested on RTX 5050 in WSL2 Ubuntu).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`: set `OPENAI_API_KEY` and, if the Windows host address changed, `LM_STUDIO_BASE_URL`.

## Running

## Testing

## AI Tools Used

## Design Decisions
