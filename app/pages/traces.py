"""Traces page: the stored answer audit table, with expandable answers."""

from __future__ import annotations

import streamlit as st

from app import data


def render(run: data.RunInfo) -> None:
    st.subheader(f"Agent traces — run {run.id}")
    conn = data.connect_read_only()
    try:
        traces = data.list_traces(conn, run.id)
    finally:
        conn.close()

    if not traces:
        st.info("No agent traces are stored for this run.")
        return

    st.dataframe(
        [
            {
                "id": trace.id,
                "time": trace.created_time,
                "mode": trace.mode,
                "question": trace.question or "-",
                "tools": ", ".join(trace.tool_names) or "none",
                "cached": "yes" if trace.cached else "no",
                "cost_usd": round(trace.estimated_cost_usd, 6),
            }
            for trace in traces
        ],
        hide_index=True,
    )

    for trace in traces:
        label = f"{trace.id} · {trace.mode} · {trace.created_time}"
        with st.expander(label):
            st.markdown(f"**Cached:** {'yes' if trace.cached else 'no'}")
            st.markdown(f"**Question:** {trace.question or '-'}")
            st.markdown(f"**Tools:** {', '.join(trace.tool_names) or 'none'}")
            st.caption(
                f"Model: {trace.model_name} · tokens: {trace.prompt_tokens}/"
                f"{trace.completion_tokens} · cost: ${trace.estimated_cost_usd:.6f} · "
                f"latency: {trace.latency_ms:.0f} ms"
            )
            st.markdown("**Answer**")
            st.write(trace.answer)
