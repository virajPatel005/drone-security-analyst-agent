"""AppTest smoke test: every page renders with the key missing and no network."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import agent
from src.config import settings

from app import agent_ui

PAGES = {
    "Overview": "Overview — run",
    "Alerts": "Alerts — run",
    "Ask": "Ask — run",
    "Traces": "Agent traces — run",
}


def test_every_page_renders_without_key_or_agent_calls(
    app_db: Path, monkeypatch
) -> None:
    # Both modules captured DB_PATH at import time; point them at the temp DB.
    monkeypatch.setattr(settings, "DB_PATH", str(app_db))
    monkeypatch.setattr(agent, "DB_PATH", str(app_db))
    monkeypatch.delenv(settings.OPENAI_API_KEY_ENV, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("UI tried to call OpenAI")

    monkeypatch.setattr(agent_ui, "generate_summary", no_network)
    monkeypatch.setattr(agent_ui, "ask", no_network)

    entrypoint = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(entrypoint, default_timeout=30)
    # Dark is the default; light is the toggle state. Both must render cleanly.
    for light in (False, True):
        at.session_state["theme_light"] = light
        for page, marker in PAGES.items():
            at.session_state["page"] = page
            at.run()
            label = f"{'light' if light else 'dark'}/{page}"
            assert not at.exception, f"{label} failed: {[str(e.value) for e in at.exception]}"
            assert any(marker in element.value for element in at.subheader), (
                f"{label} did not render its own content"
            )
