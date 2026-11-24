"""Streamlit entry point: run from the repo root with
``streamlit run app/streamlit_app.py``.

The repo-root import fix mirrors scripts/ask_agent.py so the app imports `src`
and `app` without a PYTHONPATH setting.  All pages read SQLite; only the agent
buttons write, through the existing agent commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import data
from app.pages import alerts, ask, overview, traces

PAGES = {
    "Overview": overview.render,
    "Alerts": alerts.render,
    "Ask": ask.render,
    "Traces": traces.render,
}


def main() -> None:
    st.title("Drone Security Analyst")
    st.caption(
        "Prototype demo over stored SQLite data. Time, zones and patrol route "
        "are simulated; detection and descriptions were produced offline."
    )

    with st.sidebar:
        page = st.radio("Page", tuple(PAGES), key="page")
        conn = data.connect_read_only()
        try:
            runs = data.list_runs(conn)
        finally:
            conn.close()
        if not runs:
            st.warning("No runs in the database. Run the pipeline first.")
            return
        by_id = {run.id: run for run in runs}
        run_id = st.selectbox(
            "Run",
            options=[run.id for run in runs],
            format_func=lambda rid: f"{rid} — {by_id[rid].video_name}",
            key="run_id",
        )
        run = by_id[int(run_id)]
        st.markdown(f"**Source video:** {run.video_name}")
        st.markdown(f"**Simulated start time:** {run.start_time}")

    PAGES[page](run)


main()
