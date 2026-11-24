"""Streamlit entry point: run from the repo root with
``streamlit run app/streamlit_app.py``.

The repo-root import fix mirrors scripts/ask_agent.py so the app imports `src`
and `app` without a PYTHONPATH setting.  Pages read SQLite; only the agent
buttons write, through the existing agent commands.  Visual rules live in
app/styles.py, never in the pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import data, styles
from app.pages import alerts, ask, overview, traces

# Must run before any other Streamlit command.
styles.configure_page()

PAGES = {
    "Overview": overview.render,
    "Alerts": alerts.render,
    "Ask": ask.render,
    "Traces": traces.render,
}


def main() -> None:
    # The toggle writes session_state on click; the following rerun applies it.
    styles.apply(bool(st.session_state.get("theme_light", False)))

    st.title("Drone Security Analyst")
    st.caption("Stored patrol evidence · read-only review console")

    conn = data.connect_read_only()
    try:
        runs = data.list_runs(conn)
    finally:
        conn.close()
    if not runs:
        st.warning("No runs in the database. Run the pipeline first.")
        return
    by_id = {run.id: run for run in runs}

    with st.sidebar:
        preview_id = int(st.session_state.get("run_id", runs[0].id))
        styles.brand(by_id.get(preview_id, runs[0]))
        page = st.radio("Page", tuple(PAGES), key="page")
        run_id = st.selectbox(
            "Run",
            options=[run.id for run in runs],
            format_func=lambda rid: f"{rid} — {by_id[rid].video_name}",
            key="run_id",
        )
        light = st.toggle("Light theme", key="theme_light", value=False)
        st.caption(f"Theme: {'Light' if light else 'Dark'}")
        st.caption("Simulated clock and zones · prototype data")

    run = by_id[int(run_id)]
    styles.context_strip(run)
    PAGES[page](run)
    styles.limits()


main()
