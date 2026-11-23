"""Tests for configuration loading without contacting external services."""

from __future__ import annotations

import importlib

from src.config import settings


def test_default_base_url_is_windows_host(monkeypatch):
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    reloaded = importlib.reload(settings)
    assert reloaded.LM_STUDIO_BASE_URL == "http://172.19.144.1:1234/v1"
    assert "http://localhost:1234/v1" in reloaded.LM_STUDIO_FALLBACK_BASE_URLS
    importlib.reload(settings)


def test_env_overrides_base_url(monkeypatch):
    with monkeypatch.context() as patched:
        patched.setenv("LM_STUDIO_BASE_URL", "http://example.invalid:1234/v1")
        reloaded = importlib.reload(settings)
        assert reloaded.LM_STUDIO_BASE_URL == "http://example.invalid:1234/v1"
    importlib.reload(settings)
