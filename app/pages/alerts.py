"""Alerts page: filtered alert table plus stored-data provenance for one alert."""

from __future__ import annotations

from html import escape

import streamlit as st

from app import data, styles


def render(run: data.RunInfo) -> None:
    st.subheader(f"Alerts — run {run.id}")
    conn = data.connect_read_only()
    try:
        all_alerts = data.list_alerts(conn, run.id)
        rules = sorted({alert.rule_name for alert in all_alerts})
        severities = sorted({alert.severity for alert in all_alerts})
    finally:
        conn.close()

    rule_filter = st.selectbox("Rule", ["All"] + rules, key="alert_rule_filter")
    severity_filter = st.selectbox(
        "Severity", ["All"] + severities, key="alert_severity_filter"
    )
    conn = data.connect_read_only()
    try:
        alerts = data.list_alerts(
            conn,
            run.id,
            rule_name=None if rule_filter == "All" else rule_filter,
            severity=None if severity_filter == "All" else severity_filter,
        )
        # An all-empty zone column wastes width on this data set; keep the full
        # ISO value in provenance, but show the clock time in the scan table.
        show_zone = any(alert.zone for alert in alerts)
        table = [
            {
                "id": alert.id,
                "time": alert.timestamp[11:19] if len(alert.timestamp) >= 19 else alert.timestamp,
                "rule": alert.rule_name,
                "severity": alert.severity,
                "track_id": alert.track_id,
                **({"zone": alert.zone or ""} if show_zone else {}),
                "reason": alert.reason,
            }
            for alert in alerts
        ]
        st.table(table)
        if not alerts:
            st.caption("No alerts match these filters.")
            return

        labels = [
            f"{alert.id} · {alert.rule_name} · track {alert.track_id} · {alert.timestamp}"
            for alert in alerts
        ]
        chosen = st.selectbox("Select an alert", labels, key="alert_choice")
        selected = alerts[labels.index(chosen)]
        provenance = data.get_provenance(conn, selected)
    finally:
        conn.close()

    st.markdown("#### Provenance")
    st.markdown(f"**Alert reason:** {escape(provenance.alert.reason)}")
    st.markdown(f"**Rule:** {escape(provenance.alert.rule_name)} · **Time:** {escape(provenance.alert.timestamp)}")
    styles.severity_badge(provenance.alert.severity)
    if provenance.flagged_questionable:
        st.badge("Flagged questionable by the stored summary", color="orange")

    st.markdown("**Track**")
    if provenance.track is None:
        st.caption("No stored track row for this alert.")
    else:
        st.markdown(
            f"Track {provenance.track.track_id} · class {provenance.track.class_name} · "
            f"first {provenance.track.first_seen} · last {provenance.track.last_seen} · "
            f"{provenance.track.event_count} events"
        )

    st.markdown("**Event timeline**")
    st.caption("Zone changes are simulated drone locations, not object movement.")
    items = "".join(
        "<li><time>{ts}</time>{kind} · zone {zone} · dwell {dwell:.1f}s</li>".format(
            ts=escape(event.timestamp),
            kind=escape(event.event_type),
            zone=escape(event.zone or "-"),
            dwell=event.dwell_s,
        )
        for event in provenance.events
    )
    st.markdown(f'<ul class="ops-timeline">{items}</ul>', unsafe_allow_html=True)

    st.markdown("**VLM description**")
    description = provenance.description
    if description is None:
        st.caption("No VLM row is stored for this track.")
    else:
        styles.status_badge(description.status)
        st.write(description.description or "No description text stored.")
        latency = "n/a" if description.latency_ms is None else f"{description.latency_ms:.0f} ms"
        st.caption(f"Model: {description.model_name} · latency: {latency}")
        if description.has_crop:
            st.image(description.crop_path, caption=f"Track {selected.track_id} crop")
        else:
            st.info("Crop image missing (for example a too-small crop); no image to show.")

    with st.expander("Run tuning settings"):
        if provenance.tuning:
            st.table(
                [{"setting": key, "value": str(value)} for key, value in sorted(provenance.tuning.items())]
            )
        else:
            st.caption("No tuning settings stored for this run.")
