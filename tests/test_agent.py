"""No-network tests for the OpenAI agent using fakes only."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from types import SimpleNamespace

import pytest

from src import agent
from src.config.settings import OPENAI_MAX_ROUNDS
from src.storage import (
    init_db,
    query_agent_cost_total,
    record_agent_call,
    store_description,
    store_run,
    store_run_summary,
)


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> dict:
    """Build the dictionary shape OpenAIClient returns for a tool call."""
    return {"id": call_id, "name": name, "arguments": arguments}


def _completion(
    content: str = "",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    """Build a fake SDK completion with a usage object."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


class ScriptedClient:
    """Minimal client interface used by run_conversation; never touches network."""

    model = "fake-mini"

    def __init__(self, responses: list[dict], default: dict | None = None) -> None:
        self.responses = list(responses)
        self.default = default
        self.calls: list[dict] = []

    def call(self, *, messages, max_tokens, tools=None) -> dict:
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)}
        )
        if self.responses:
            response = self.responses.pop(0)
        else:
            assert self.default is not None, "fake client ran out of scripted replies"
            response = self.default
        return {
            "content": response.get("content", ""),
            "tool_calls": response.get("tool_calls", []),
            "prompt_tokens": response.get("prompt_tokens", 10),
            "completion_tokens": response.get("completion_tokens", 5),
            "estimated_cost_usd": response.get("cost", 0.0001),
            "latency_s": 0.01,
            "running_total_usd": response.get("running_total", 0.0001),
            "model": self.model,
        }


def test_only_allowed_tools_are_defined_and_runnable(tmp_path) -> None:
    defined = {item["function"]["name"] for item in agent.TOOL_DEFINITIONS}
    assert defined == set(agent.ALLOWED_TOOLS)

    db = init_db(tmp_path / "tools.db")
    error = json.loads(agent._run_tool("drop_table", {}, db))
    assert error == {"error": "unknown tool: drop_table"}
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "drop_table" not in tables
    db.close()


def test_round_cap_stops_tool_loop(tmp_path) -> None:
    class LoopClient:
        model = "fake-mini"

        def __init__(self) -> None:
            self.calls = 0

        def call(self, *, messages, max_tokens, tools=None) -> dict:
            self.calls += 1
            return {
                "content": "still working",
                "tool_calls": [_tool_call("list_runs", "{}", f"loop_{self.calls}")],
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "estimated_cost_usd": 0.0,
                "latency_s": 0.0,
                "running_total_usd": 0.0,
                "model": self.model,
            }

    db = init_db(tmp_path / "loop.db")
    client = LoopClient()
    result = agent.run_conversation(
        client, system="system", user="user", db=db
    )
    assert client.calls == OPENAI_MAX_ROUNDS == 4
    assert "Stopped after 4 rounds" in result["content"]
    db.close()


def test_unknown_tool_returns_error_to_model(tmp_path) -> None:
    db = init_db(tmp_path / "unknown.db")
    final = {"content": "No such tool is available.", "tool_calls": []}
    client = ScriptedClient(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("not_a_tool", "{}", "unknown_1")],
            },
            final,
        ],
        default=final,
    )
    result = agent.run_conversation(
        client, system="system", user="question", db=db
    )

    tool_messages = [
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "unknown tool: not_a_tool" in tool_messages[0]["content"]
    assert result["content"] == "No such tool is available."
    db.close()


def test_no_image_data_in_payload(tmp_path) -> None:
    db = init_db(tmp_path / "noimage.db")
    store_run(
        db,
        source_video="video.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )
    store_description(
        db,
        run_id=1,
        track_id=7,
        status="described",
        description="white car",
        attributes={"main_colour": "white"},
        crop_path="data/crops/run_001/track_000007.jpg",
        orig_width=40,
        orig_height=40,
        model_name="local-vlm",
        latency_ms=12.0,
    )
    final = {"content": "Text only.", "tool_calls": []}
    client = ScriptedClient(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call("get_descriptions", '{"run_id": 1}', "image_1")
                ],
            },
            final,
        ],
        default=final,
    )
    agent.run_conversation(client, system="system", user="question", db=db)

    payload = json.dumps(client.calls, default=str).lower()
    for forbidden in ("image_url", "data:image", "base64", '"image"', "crop_path"):
        assert forbidden not in payload
    db.close()


def test_stored_summary_returns_without_model_call(
    tmp_path, monkeypatch, capsys
) -> None:
    db_path = tmp_path / "cached.db"
    db = init_db(db_path)
    store_run(
        db,
        source_video="video.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )
    store_run_summary(
        db,
        run_id=1,
        text="CACHED SUMMARY TEXT",
        model_name="fake-mini",
        prompt_tokens=10,
        completion_tokens=5,
        estimated_cost_usd=0.001,
    )
    db.close()

    class ExplodingClient:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("OpenAIClient must not be built for a cached summary")

    monkeypatch.setattr(agent, "OpenAIClient", ExplodingClient)
    monkeypatch.setattr(agent, "DB_PATH", db_path)

    assert agent.cmd_summary(["1"]) == 0
    output = capsys.readouterr().out
    assert "Cached summary (no new model call)" in output
    assert "CACHED SUMMARY TEXT" in output
    assert "stats:" in output


def test_cost_guard_warns_and_refuses(tmp_path, monkeypatch, caplog) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            return _completion("ok", prompt_tokens=200_000, completion_tokens=0)

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(agent.openai, "OpenAI", FakeOpenAI)
    db = init_db(tmp_path / "cost.db")
    record_agent_call(
        db,
        run_id=1,
        command="ask",
        model_name="fake-mini",
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=0.40,
        latency_s=0.0,
    )
    client = agent.OpenAIClient(
        conn=db, run_id=1, command="ask", api_key="fake-key"
    )

    with caplog.at_level(logging.WARNING):
        result = client.call(
            messages=[{"role": "user", "content": "hello"}], max_tokens=10
        )
    assert result["estimated_cost_usd"] == pytest.approx(0.15)
    assert query_agent_cost_total(db) == pytest.approx(0.55)
    assert "spend warning" in caplog.text

    record_agent_call(
        db,
        run_id=1,
        command="ask",
        model_name="fake-mini",
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=0.50,
        latency_s=0.0,
    )
    with pytest.raises(agent.BudgetExceeded):
        client.call(messages=[{"role": "user", "content": "hello again"}], max_tokens=10)
    db.close()


def test_compact_rows_reports_total_and_truncation() -> None:
    long_payload = agent.compact_rows(list(range(60)), limit=50)
    assert long_payload["total"] == 60
    assert long_payload["showing"] == 50
    assert long_payload["truncated"] is True
    assert long_payload["notice"] == "showing 50 of 60"

    short_payload = agent.compact_rows([1, 2], limit=50)
    assert short_payload["total"] == 2
    assert short_payload["truncated"] is False
    assert "notice" not in short_payload


def test_long_tool_result_has_truncated_flag(tmp_path) -> None:
    db = init_db(tmp_path / "long.db")
    for track_id in range(1, 61):
        store_description(
            db,
            run_id=1,
            track_id=track_id,
            status="described",
            description="white car",
            attributes={"main_colour": "white"},
            crop_path=None,
            orig_width=50,
            orig_height=50,
            model_name="local-vlm",
            latency_ms=1.0,
        )
    payload = json.loads(agent._run_tool("get_descriptions", {"run_id": 1}, db))
    assert payload["total"] == 60
    assert payload["truncated"] is True
    assert payload["notice"] == "showing 50 of 60"
    assert len(payload["rows"]) == 50
    db.close()
