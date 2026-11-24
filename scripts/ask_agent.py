"""ask_agent.py - CLI for OpenAI agent about run data."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import cmd_ask, cmd_summary, cmd_traces


def main() -> None:
    if len(sys.argv) < 2:
        print("ask_agent.py <summary|ask|traces> <run-id> [args]", file=sys.stderr)
        sys.exit(2)
    command = sys.argv[1]
    if command == "summary":
        sys.exit(cmd_summary(sys.argv[2:]))
    if command == "ask":
        sys.exit(cmd_ask(sys.argv[2:]))
    if command == "traces":
        sys.exit(cmd_traces(sys.argv[2:]))
    print(f"Unknown command: {command}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
