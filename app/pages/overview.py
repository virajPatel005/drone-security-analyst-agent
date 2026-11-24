"""Overview page: one run's counts and its stored agent summary."""

from __future__ import annotations

import streamlit as st

from src.config import settings

from app import agent_ui, data, styles


def render(run: data.RunInfo) -> None:
    st.subheader(f"Overview — run {run.id}")
    st.caption(
        "The clock, zones and patrol route are simulated. Alerts count tracked "
        "IDs, not necessarily distinct physical objects."
    )

    conn = data.connect_read_only()
    try:
        counts = data.run_counts(conn, run.id)
        summary = data.get_summary(conn, run.id)
        state = data.agent_state(conn)
    finally:
        conn.close()

    left, middle, right = st.columns(3)
    left.metric("Tracks", counts.tracks)
    middle.metric("Events", counts.events)
    right.metric("Alerts", counts.alerts)

    if summary is None:
        st.info("No summary is stored for this run.")
        expected = agent_ui.expected_summary_cost_usd()
        st.caption(
            f"A fresh summary uses {settings.OPENAI_MODEL_NAME} on stored text only. "
            f"Expected cost: about ${expected:.6f} (upper estimate). "
            f"Current spend ${state.running_total_usd:.6f} of ${state.cap_usd:.2f} cap."
        )
        if not state.can_call:
            st.warning(f"Summary generation disabled. {state.blocked_reason}")
        elif st.session_state.get("confirm_summary"):
            st.warning(
                f"Confirm one {settings.OPENAI_MODEL_NAME} call at about ${expected:.6f}. "
                "Only stored text is sent."
            )
            yes, no = st.columns(2)
            if styles.primary_button("Confirm spend"):
                with st.spinner("Generating stored-text summary…"):
                    agent_ui.generate_summary(run.id)
                st.session_state.pop("confirm_summary", None)
                st.rerun()
            if no.button("Cancel"):
                st.session_state.pop("confirm_summary", None)
                st.rerun()
        elif styles.primary_button("Generate summary"):
            st.session_state["confirm_summary"] = True
            st.rerun()
        return

    st.markdown("#### Stored summary")
    st.caption(
        f"Model: {summary.model_name} · prompt tokens: {summary.prompt_tokens} · "
        f"completion tokens: {summary.completion_tokens} · "
        f"cost: ${summary.estimated_cost_usd:.6f} · created: {summary.created_time}"
    )
    parsed = summary.parsed
    if parsed is None:
        st.write(summary.text)
        return

    st.write(parsed.get("summary", ""))

    notable = parsed.get("notable_alerts", [])
    st.markdown("**Notable alerts**")
    if isinstance(notable, list) and notable:
        rows = [
            {
                "track_id": item.get("track_id"),
                "rule_name": item.get("rule_name"),
                "severity": item.get("severity"),
                "timestamp": item.get("timestamp"),
                "reason": item.get("reason"),
            }
            for item in notable
            if isinstance(item, dict)
        ]
        st.table(rows)
    else:
        st.caption("None listed in the stored summary.")

    st.markdown("**Questionable tracks**")
    questionable = summary.questionable
    st.write(", ".join(str(track) for track in questionable) if questionable else "None")

    st.markdown("**Recommendations**")
    recommendations = parsed.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        for item in recommendations:
            st.markdown(f"- {item}")
    else:
        st.caption("None listed in the stored summary.")
