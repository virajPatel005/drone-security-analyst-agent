"""Thin UI adapter around the existing agent commands.

The Streamlit app must not reimplement prompts, tool loops, tracing or the cost
guard.  These helpers call ``src.agent.cmd_summary`` / ``cmd_ask`` (the same
functions the CLI uses) and then read back the stored answer from SQLite.
"""

from __future__ import annotations

from typing import Any

from src import agent
from src.config import settings

from app import data


def expected_summary_cost_usd() -> float:
    """Upper estimate for one fresh summary, shown before the user clicks.

    Real digests were about 3.4k-3.7k prompt tokens in earlier runs; pairing a
    4k prompt allowance with the configured output ceiling keeps the stated
    figure conservative without making a model call.
    """
    return agent.estimate_cost(
        prompt_tokens=4_000, completion_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS
    )


def generate_summary(run_id: int) -> None:
    """Generate a cached summary through the existing CLI command logic."""
    result = agent.cmd_summary([str(run_id)])
    if result != 0:
        raise RuntimeError(f"Summary command failed for run {run_id}")


def ask(run_id: int, question: str) -> dict[str, Any]:
    """Ask through the existing command, then read back the audit trace."""
    result = agent.cmd_ask([str(run_id), question])
    if result != 0:
        raise RuntimeError(f"Ask command failed for run {run_id}")
    conn = data.connect_read_only()
    try:
        traces = [t for t in data.list_traces(conn, run_id) if t.mode == "ask"]
        if not traces:
            raise RuntimeError("Ask command did not leave a stored trace")
        trace = traces[-1]
        return {
            "question": trace.question,
            "answer": trace.answer,
            "tool_names": trace.tool_names,
            "cost_usd": trace.estimated_cost_usd,
            "model": trace.model_name,
            "running_total_usd": data.agent_state(conn).running_total_usd,
            "cap_usd": float(settings.OPENAI_COST_CAP_USD),
        }
    finally:
        conn.close()
