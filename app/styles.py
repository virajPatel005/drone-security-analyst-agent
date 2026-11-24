"""One place for the console's visual system.

The tokens below are the only source of colour, type, spacing and state rules.
Both themes are the same semantic roles with different values; the DOM carries
``data-theme`` so every Streamlit element, including tables, badges, the event
timeline and crop images, resolves the same variables.

Streamlit 1.67.0 supports native theming only at startup via ``config.toml``.
There is no supported runtime theme API, so the in-app switch injects this CSS
through a same-origin component script.  That limitation is documented in
docs/testing.md rather than hidden.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from streamlit.components.v1 import html as component_html

# Shared type and spacing primitives. Kept in CSS custom properties so the
# injected stylesheet, not Python, owns every rendered decision.
_CSS = r"""
html[data-theme="dark"] {
  color-scheme: dark;
  --page:#0F1418; --panel:#151C21; --raised:#1B242A; --border:#5A707A;
  --text:#E6EDF0; --text-secondary:#A9BBC2; --text-muted:#95A9B1;
  --accent:#5BA6B8; --accent-soft:#17323A; --on-accent:#071115;
  --warning:#F0B75E; --warning-bg:#332613;
  --info:#A9CBE9; --info-bg:#162734;
  --success:#9AD3B2; --success-bg:#143026;
  --focus:#8FD3E2;
}
html[data-theme="light"] {
  color-scheme: light;
  --page:#F4F7F8; --panel:#FBFCFC; --raised:#EEF3F5; --border:#71868E;
  --text:#162228; --text-secondary:#3E525B; --text-muted:#5B7079;
  --accent:#176C7B; --accent-soft:#E6F2F4; --on-accent:#F7FAFA;
  --warning:#7A4E08; --warning-bg:#FFF3DB;
  --info:#1E4E76; --info-bg:#E8F1F9;
  --success:#1B603E; --success-bg:#E6F4EB;
  --focus:#0F6676;
}
:root { --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
        --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;
        --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s6:24px; --s8:32px; --r-control:4px; --r-panel:2px; }
html, body, [data-testid="stAppViewContainer"] { background:var(--page); color:var(--text); font-family:var(--sans); font-size:14px; }
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color:var(--text-secondary); }
[data-testid="stAppViewContainer"] strong, [data-testid="stSidebar"] strong { color:var(--text); font-weight:600; }
[data-testid="stMainBlockContainer"] { padding:var(--s6) var(--s8) 80px; max-width:none; }
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--border); }
[data-testid="stSidebarContent"] { padding:var(--s4) var(--s3) var(--s6); }
/* Streamlit's auto-generated multipage list duplicates our own navigation. */
nav[data-testid="stSidebarNav"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], #MainMenu, footer, .stDeployButton { display:none !important; }
/* Keep stHeader and stToolbar: the sidebar expand control lives inside them. */
[data-testid="stHeader"], [data-testid="stToolbar"] { background:transparent; }
h1 { font-size:22px; font-weight:650; letter-spacing:-0.01em; margin:0 0 var(--s1); color:var(--text); }
h2, [data-testid="stAppViewContainer"] h3 { font-size:16px; font-weight:600; color:var(--text);
     border-bottom:1px solid var(--border); padding-bottom:var(--s2); margin:var(--s6) 0 var(--s3); }
[data-testid="stAppViewContainer"] h4 { font-size:14px; font-weight:600; color:var(--text); margin:var(--s4) 0 var(--s2); }
[data-testid="stCaptionContainer"] p, .stCaption { color:var(--text-muted); font-size:12px; }
.ops-brand { border-bottom:1px solid var(--border); padding-bottom:var(--s3); margin-bottom:var(--s4); }
.ops-brand .ops-brand-name { display:block; font-size:15px; font-weight:650; letter-spacing:-0.01em; color:var(--text); }
.ops-context { display:flex; flex-wrap:wrap; gap:var(--s2) var(--s4); align-items:baseline;
     background:var(--raised); border:1px solid var(--border); border-radius:var(--r-panel);
     padding:var(--s2) var(--s3); margin:0 0 var(--s4); font-size:12px; color:var(--text-secondary); }
.ops-context b { font-weight:600; color:var(--text); }
.ops-context code, .ops-mono, code { font-family:var(--mono); font-variant-numeric:tabular-nums;
     color:var(--text); background:transparent !important; padding:0; }
.ops-limits { color:var(--text-muted); font-size:12px; border-top:1px solid var(--border);
     padding-top:var(--s2); margin-top:var(--s6); }
.ops-timeline { list-style:none; margin:var(--s2) 0; padding:0 0 0 var(--s3); border-left:1px solid var(--border); }
.ops-timeline li { position:relative; padding:0 0 var(--s2) var(--s2); font-size:13px; color:var(--text-secondary); }
.ops-timeline li:before { content:""; position:absolute; left:calc(-1 * var(--s3) - 3px); top:7px;
     width:5px; height:5px; border-radius:50%; background:var(--accent); }
.ops-timeline time { font-family:var(--mono); font-variant-numeric:tabular-nums; color:var(--text); margin-right:var(--s2); }
button, [data-testid="stButton"] button, [data-testid="stToggle"] label,
[data-testid="stRadioOption"], [data-testid="stSelectbox"] [role="combobox"],
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
  min-height:40px; border-radius:var(--r-control); font-family:var(--sans); font-size:13px;
  transition:background-color 150ms ease, border-color 150ms ease, color 150ms ease;
}
[data-testid="stButton"] button { background:var(--raised); border:1px solid var(--border); color:var(--text); padding:var(--s2) var(--s3); }
[data-testid="stButton"] button[kind="primary"], [data-testid="stButton"] button[data-testid="baseButton-primary"] {
  background:var(--accent); border-color:var(--accent); color:var(--on-accent); font-weight:600;
}
[data-testid="stButton"] button:hover:not(:disabled) { background:var(--accent-soft); border-color:var(--accent); }
[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled) { background:var(--accent); filter:brightness(1.08); }
[data-testid="stButton"] button:active:not(:disabled) { transform:translateY(1px); }
[data-testid="stButton"] button:disabled { opacity:0.5; cursor:not-allowed; }
[data-testid="stRadioOption"] { border:1px solid transparent; padding:var(--s2) var(--s3); min-height:40px; }
[data-testid="stRadioOption"]:has(input:checked) { background:var(--accent-soft); border-color:var(--accent); }
[data-testid="stRadioOption"] p { color:var(--text-secondary) !important; }
[data-testid="stRadioOption"]:has(input:checked) p { color:var(--text) !important; font-weight:600; }
[data-testid="stSelectbox"] [role="combobox"], [data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea { background:var(--raised); border:1px solid var(--border); color:var(--text); padding:var(--s2) var(--s3); }
[data-testid="stSelectbox"] [role="combobox"]:hover, [data-testid="stTextInput"] input:hover,
[data-testid="stTextArea"] textarea:hover { border-color:var(--accent); }
[data-testid="stToggle"] label { border:1px solid var(--border); padding:var(--s2) var(--s3); background:var(--raised); }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="combobox"]:focus-visible,
[data-testid="stRadioOption"]:focus-within, summary:focus-visible {
  outline:2px solid var(--focus); outline-offset:2px;
}
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:var(--r-panel); font-size:12px; }
/* Static HTML tables are used instead of the canvas-backed dataframe so the
   same tokens apply to cells, headers and text in both themes. */
[data-testid="stTable"] table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
[data-testid="stDataFrame"] table { font-variant-numeric:tabular-nums; }
/* Streamlit paints table text with its own dark-on-light colour; override it. */
[data-testid="stTable"] td, [data-testid="stTable"] th { color:var(--text) !important; background:var(--panel) !important;
     font-family:var(--sans) !important; }
[data-testid="stTable"] thead th { color:var(--text-secondary) !important; background:var(--raised) !important; }
[data-testid="stTable"] td { overflow-wrap:anywhere; }
[data-testid="stDataFrame"] thead, [data-testid="stTable"] thead { background:var(--raised); color:var(--text-secondary); }
[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td,
[data-testid="stTable"] th, [data-testid="stTable"] td { border-bottom:1px solid var(--border); padding:var(--s2) var(--s3); }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-radius:var(--r-panel); padding:var(--s3); }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family:var(--mono); font-variant-numeric:tabular-nums;
     font-size:22px; color:var(--text); }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
  min-width:40px; min-height:40px; border-radius:var(--r-control);
}
[data-testid="stExpander"] details { border:1px solid var(--border); border-radius:var(--r-panel); background:var(--panel); }
[data-testid="stExpander"] summary { padding:var(--s2) var(--s3); color:var(--text); }
[data-testid="stAlert"] { border-radius:var(--r-panel); border:1px solid var(--border); }
[data-testid="stAlertContainer"] p { font-size:12px; }
/* Streamlit 1.67 st.badge emits .stMarkdownBadge with inline light-mode colours
   and no data-testid, so the semantic warning colours need !important here. */
span.stMarkdownBadge, [data-testid="stBadge"] { background:var(--warning-bg) !important;
     color:var(--warning) !important; border:1px solid var(--warning) !important;
     border-radius:var(--r-control); padding:2px var(--s2); font-size:11px !important; }
.ops-chip { display:inline-block; border-radius:var(--r-control); padding:2px var(--s2); font-size:11px;
     border:1px solid currentColor; margin-right:var(--s2); font-family:var(--mono); }
.ops-chip-warning { color:var(--warning); background:var(--warning-bg); }
.ops-chip-info { color:var(--info); background:var(--info-bg); }
.ops-chip-status { color:var(--text-secondary); background:var(--raised); }
[data-testid="stButton"] button[title="Example question"] { background:transparent; border-color:transparent;
     color:var(--accent); padding:var(--s2); text-align:left; white-space:normal; }
[data-testid="stButton"] button[title="Example question"]:hover { background:var(--accent-soft); border-color:transparent; }
[data-testid="stMain"] img { border:1px solid var(--border); border-radius:var(--r-panel); background:var(--raised); }
::selection { background:var(--accent); color:var(--on-accent); }
input, textarea { caret-color:var(--accent); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:var(--r-control); }
@media (max-width:900px) {
  [data-testid="stMainBlockContainer"] { padding:var(--s4) var(--s4) 60px; }
  .ops-context { gap:var(--s1) var(--s3); }
}
@media (prefers-reduced-motion:reduce) {
  [data-testid="stAppViewContainer"] button, [data-testid="stRadioOption"], input, textarea { transition:none !important; }
}
"""


def configure_page() -> None:
    """Set the wide desktop layout before any other Streamlit command runs."""
    st.set_page_config(page_title="Drone Security Analyst", layout="wide")


def apply(light: bool) -> None:
    """Switch the whole document's tokens and persist the injected stylesheet."""
    theme = "light" if light else "dark"
    script = (
        "<script>(function(){var d=window.parent.document;"
        "d.documentElement.setAttribute('data-theme','" + theme + "');"
        "var s=d.getElementById('ops-theme-style');"
        "if(!s){s=d.createElement('style');s.id='ops-theme-style';d.head.appendChild(s);}"
        "s.textContent=" + json.dumps(_CSS) + ";})();</script>"
    )
    component_html(script, height=0, scrolling=False)


def brand(run: Any) -> None:
    """Sidebar brand header with the run context underneath."""
    # No kicker above the brand name: the heading carries its own weight.
    st.markdown(
        '<div class="ops-brand"><span class="ops-brand-name">Drone Security Analyst</span></div>',
        unsafe_allow_html=True,
    )


def context_strip(run: Any) -> None:
    """Keep the selected run visible even when the sidebar is collapsed."""
    st.markdown(
        '<div class="ops-context">'
        f'<span><b>Run</b> <code>{run.id}</code></span>'
        f'<span><b>Video</b> <code>{run.video_name}</code></span>'
        f'<span><b>Simulated start</b> <code>{run.start_time}</code></span>'
        "</div>",
        unsafe_allow_html=True,
    )


def limits() -> None:
    """State the two honest limits on every page without relying on narration."""
    st.markdown(
        '<div class="ops-limits">Simulated clock and zones. Alerts count tracker '
        "tracks, not necessarily distinct physical objects.</div>",
        unsafe_allow_html=True,
    )


def primary_button(label: str, **kwargs: Any) -> bool:
    """The single primary action of a view."""
    return st.button(label, type="primary", **kwargs)


def severity_badge(severity: str) -> None:
    """Colour severity only: warning amber, info blue, anything else neutral."""
    kind = "warning" if severity == "warning" else "info" if severity == "info" else "status"
    st.markdown(
        f'<span class="ops-chip ops-chip-{kind}">{severity}</span>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> None:
    """VLM status is a state, not a severity; keep it neutral."""
    st.markdown(
        f'<span class="ops-chip ops-chip-status">{status}</span>',
        unsafe_allow_html=True,
    )
