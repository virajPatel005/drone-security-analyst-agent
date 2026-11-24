"""Ask page: questions answered by the existing agent over stored text."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import agent_ui, data, styles

EXAMPLE_QUESTIONS = (
    "Were any vehicles seen after midnight? Give times and track ids.",
    "Show all truck events and tell me whether they look reliable.",
    "Which alerts look questionable and why?",
)


def render(run: data.RunInfo) -> None:
    st.subheader(f"Ask — run {run.id}")
    conn = data.connect_read_only()
    try:
        state = data.agent_state(conn)
    finally:
        conn.close()

    st.caption(f"Source video: {run.video_name} · simulated start: {run.start_time}")
    st.markdown("**Example questions**")
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(
            question,
            key=f"example_{index}",
            disabled=not state.can_call,
            help="Example question",
        ):
            st.session_state["ask_question"] = question
            st.rerun()

    if not state.can_call:
        st.warning(
            "Asking is disabled. " + state.blocked_reason
            + f" Running total ${state.running_total_usd:.6f} of ${state.cap_usd:.2f} cap."
        )

    question = st.text_input(
        "Question",
        key="ask_question",
        disabled=not state.can_call,
        placeholder="Type a question about the stored events and alerts",
    )
    pending = st.session_state.get("confirm_ask")
    if pending and pending == question.strip():
        st.warning("Confirm one stored-text call for this question. Only stored text is sent.")
        yes, no = st.columns(2)
        if styles.primary_button("Confirm question"):
            with st.spinner("Asking the stored-text agent…"):
                try:
                    result = agent_ui.ask(run.id, question.strip())
                except Exception as exc:
                    # Show the failure and a next step; never a key or header.
                    st.error(
                        f"Ask failed: {type(exc).__name__}: {exc}. "
                        "Check the key and cost cap, then retry or inspect Traces."
                    )
                else:
                    st.session_state.pop("confirm_ask", None)
                    _show_result(run, result)
        if no.button("Cancel"):
            st.session_state.pop("confirm_ask", None)
            st.rerun()
    elif styles.primary_button("Ask", disabled=not state.can_call or not question.strip()):
        st.session_state["confirm_ask"] = question.strip()
        st.rerun()


def _show_result(run: data.RunInfo, result: dict[str, Any]) -> None:
    st.markdown(f"**Answer — run {run.id} ({run.video_name})**")
    st.write(result["answer"])
    tools = ", ".join(result["tool_names"]) or "none"
    st.caption(f"Tools used: {tools}")
    st.caption(
        f"Model: {result['model']} · cost of this question: ${result['cost_usd']:.6f} · "
        f"running total: ${result['running_total_usd']:.6f} of "
        f"${result['cap_usd']:.2f} cap"
    )
