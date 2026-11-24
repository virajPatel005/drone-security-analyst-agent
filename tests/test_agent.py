"""No-network tests for the OpenAI agent using fakes only."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src import agent
from src.config.settings import OPENAI_MAX_ROUNDS
from src.storage import (
    _EventRow,
    init_db,
    query_agent_cost_total,
    query_agent_traces,
    query_alerts,
    query_events_in_time_window,
    query_events_for_track,
    record_agent_call,
    store_description,
    store_run,
    store_run_summary,
    store_tracks_and_events,
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


# ---------------------------------------------------------------------------
# Step 8 scenario tests: real SQLite rows, fake LLM, no network or GPU
# ---------------------------------------------------------------------------


def _scenario_event(
    event_type: str, track_id: int, class_name: str, when: datetime, zone: str = "gate"
) -> _EventRow:
    """Build a stored-row-shaped event; the database is real, only data is fake."""
    return _EventRow(
        id=0,
        run_id=1,
        track_id=track_id,
        event_type=event_type,
        timestamp=when.isoformat(timespec="seconds"),
        zone=zone,
        dwell_s=0.0,
        class_name=class_name,
        end_of_video=0,
    )


def test_scenario_midnight_vehicle_and_fake_agent(tmp_path) -> None:
    db = init_db(tmp_path / "midnight.db")
    run_id = store_run(
        db,
        source_video="fake.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )
    before = datetime(2026, 11, 23, 23, 59, 58)
    after = datetime(2026, 11, 24, 0, 0, 3)
    events: list[_EventRow] = []
    for track_id, when in ((1, before), (2, after)):
        events.append(_scenario_event("appeared", track_id, "car", when))
        events.append(
            _scenario_event("left", track_id, "car", when + timedelta(seconds=1))
        )
    store_tracks_and_events(db, run_id, events)

    from src.alerts import check_vehicle_at_night

    check_vehicle_at_night(db, run_id=run_id)
    alerts = query_alerts(db, run_id=run_id, rule_name="vehicle_at_night")
    assert [a["track_id"] for a in alerts] == [2]

    window = query_events_in_time_window(
        db,
        start=datetime(2026, 11, 23, 23, 59, 55),
        end=datetime(2026, 11, 24, 0, 0, 9),
        run_id=run_id,
    )
    assert {e.track_id for e in window} == {1, 2}
    assert any(e.timestamp.startswith("2026-11-23") for e in window)
    assert any(e.timestamp.startswith("2026-11-24") for e in window)

    final = {"content": "Only track 2 at 2026-11-24T00:00:03.", "tool_calls": []}
    client = ScriptedClient(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "get_alerts",
                        '{"run_id": 1, "rule_name": "vehicle_at_night"}',
                        "mid_1",
                    )
                ],
            },
            final,
        ],
        default=final,
    )
    result = agent.run_conversation(
        client, system="system", user="vehicles after midnight?", db=db
    )
    tool_message = [
        m for m in client.calls[1]["messages"] if m.get("role") == "tool"
    ][0]
    assert '"track_id": 2' in tool_message["content"]
    assert '"track_id": 1' not in tool_message["content"]
    assert "track 2" in result["content"]
    assert "2026-11-24T00:00:03" in result["content"]
    assert "track 1" not in result["content"]
    assert "23:59:58" not in result["content"]
    db.close()


def test_scenario_truck_logged_and_fake_agent_cites_track(tmp_path) -> None:
    db = init_db(tmp_path / "truck.db")
    run_id = store_run(
        db,
        source_video="fake.mp4",
        start_time=datetime(2026, 11, 24, 14, 0, 0),
        tuning_settings={},
    )
    when = datetime(2026, 11, 24, 14, 0, 0)
    store_tracks_and_events(
        db,
        run_id,
        [
            _scenario_event("appeared", 5, "truck", when),
            _scenario_event("left", 5, "truck", when + timedelta(seconds=2)),
        ],
    )

    from src.alerts import ALERT_RULES

    for _ in range(2):
        for rule_fn in ALERT_RULES.values():
            rule_fn(db, run_id=run_id)

    alerts = query_alerts(db, run_id=run_id, rule_name="heavy_vehicle_logged")
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 5
    assert alerts[0]["severity"] == "info"

    final = {"content": "Truck track 5 was logged.", "tool_calls": []}
    client = ScriptedClient(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "get_alerts",
                        '{"run_id": 1, "rule_name": "heavy_vehicle_logged"}',
                        "truck_1",
                    )
                ],
            },
            final,
        ],
        default=final,
    )
    result = agent.run_conversation(
        client, system="system", user="show truck events", db=db
    )
    assert [call["name"] for call in result["tool_calls"]] == ["get_alerts"]
    assert "5" in result["content"]
    tool_message = [
        m for m in client.calls[1]["messages"] if m.get("role") == "tool"
    ][0]
    assert "heavy_vehicle_logged" in tool_message["content"]
    db.close()


def test_scenario_questionable_alert_in_summary_digest(tmp_path) -> None:
    db = init_db(tmp_path / "questionable.db")
    run_id = store_run(
        db,
        source_video="fake.mp4",
        start_time=datetime(2026, 11, 24, 14, 0, 0),
        tuning_settings={},
    )
    when = datetime(2026, 11, 24, 14, 0, 0)
    store_tracks_and_events(
        db, run_id, [_scenario_event("appeared", 9, "truck", when)]
    )

    from src.alerts import check_heavy_vehicle_logged

    check_heavy_vehicle_logged(db, run_id=run_id)
    store_description(
        db,
        run_id=run_id,
        track_id=9,
        status="described",
        description="A blurry view of a brick building facade.",
        attributes={"main_colour": "grey"},
        crop_path=None,
        orig_width=40,
        orig_height=40,
        model_name="local-vlm",
        latency_ms=1.0,
    )

    digest = agent._build_tool_results(db, run_id)
    assert "heavy_vehicle_logged" in digest
    assert "brick building facade" in digest
    assert '"track_id": 9' in digest
    db.close()


def test_scenario_traces_for_ask_and_cached_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    db_path = tmp_path / "traces.db"
    db = init_db(db_path)
    store_run(
        db,
        source_video="fake.mp4",
        start_time=datetime(2026, 11, 23, 23, 59, 55),
        tuning_settings={},
    )
    store_run_summary(
        db,
        run_id=1,
        text="CACHED SUMMARY TEXT",
        model_name="fake-mini",
        prompt_tokens=99,
        completion_tokens=88,
        estimated_cost_usd=0.01,
    )
    db.close()

    final = {"content": "TRACED ANSWER", "tool_calls": []}

    def fake_client_factory(**kwargs):
        return ScriptedClient(
            [
                {
                    "content": "",
                    "tool_calls": [_tool_call("get_tracks", '{"run_id": 1}', "tr_1")],
                },
                final,
            ],
            default=final,
        )

    monkeypatch.setattr(agent, "OpenAIClient", fake_client_factory)
    monkeypatch.setattr(agent, "DB_PATH", db_path)

    assert agent.cmd_ask(["1", "show truck events"]) == 0
    capsys.readouterr()
    db = init_db(db_path)
    rows = query_agent_traces(db, run_id=1)
    assert len(rows) == 1
    assert rows[0]["mode"] == "ask"
    assert rows[0]["question"] == "show truck events"
    assert rows[0]["answer"] == "TRACED ANSWER"
    assert [call["name"] for call in rows[0]["tool_calls"]] == ["get_tracks"]
    assert rows[0]["cached"] is False
    db.close()

    assert agent.cmd_summary(["1"]) == 0
    capsys.readouterr()
    db = init_db(db_path)
    rows = query_agent_traces(db, run_id=1)
    assert len(rows) == 2
    cached_rows = [r for r in rows if r["mode"] == "summary"]
    assert len(cached_rows) == 1
    assert cached_rows[0]["cached"] is True
    assert cached_rows[0]["answer"] == "CACHED SUMMARY TEXT"
    assert cached_rows[0]["prompt_tokens"] == 0
    assert cached_rows[0]["completion_tokens"] == 0
    assert cached_rows[0]["estimated_cost_usd"] == 0.0
    assert cached_rows[0]["tool_calls"] == []

    # Read the key only to assert it never reached the database; never print it.
    key = agent.openai_api_key()
    dump = json.dumps(rows, default=str)
    assert key not in dump
    assert "Authorization" not in dump
    assert "Bearer" not in dump
    db.close()


def test_ask_system_prompt_states_partial_digest_and_must_call_tool() -> None:
    system = agent._ASK_SYSTEM.lower()
    for phrase in (
        "partial",
        "alerts, truck-class tracks and their events, descriptions and event counts only",
        "must call the matching read-only tool",
        "never answer \"the data cannot tell\" without first trying the relevant tool",
        "alerts are not events",
    ):
        assert phrase in system, f"missing instruction: {phrase}"


def test_ask_user_template_contains_run_id() -> None:
    prompt = agent._ASK_USER_TEMPLATE.format(
        run_id=7, question="why?", tool_results="DIGEST"
    )
    assert "Run 7" in prompt
    assert "Name run 7 in your answer" in prompt
    assert "Question: why?" in prompt
    assert "DIGEST" in prompt


def test_scenario_ask_tool_call_returns_real_events_and_trace_records_it(
    tmp_path, monkeypatch, capsys
) -> None:
    db_path = tmp_path / "tool_used.db"
    db = init_db(db_path)
    run_id = store_run(
        db,
        source_video="fake.mp4",
        start_time=datetime(2026, 11, 24, 14, 0, 0),
        tuning_settings={},
    )
    base = datetime(2026, 11, 24, 14, 0, 0)
    event_specs = [
        ("appeared", 0),
        ("entered_zone", 0),
        ("entered_zone", 3),
        ("entered_zone", 6),
        ("entered_zone", 9),
        ("left", 10),
        ("appeared", 12),
        ("entered_zone", 12),
        ("entered_zone", 15),
        ("left", 15),
    ]
    store_tracks_and_events(
        db,
        run_id,
        [
            _scenario_event(kind, 47, "car", base + timedelta(seconds=offset))
            for kind, offset in event_specs
        ],
    )
    stored = query_events_for_track(db, run_id=run_id, track_id=47)
    assert len(stored) == 10
    db.close()

    final = {"content": "Run 1: track 47 has 10 stored events.", "tool_calls": []}
    created: list[ScriptedClient] = []

    def fake_client_factory(**kwargs):
        client = ScriptedClient(
            [
                {
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            "get_events_for_track",
                            '{"run_id": 1, "track_id": 47}',
                            "ev_47",
                        )
                    ],
                },
                final,
            ],
            default=final,
        )
        created.append(client)
        return client

    monkeypatch.setattr(agent, "OpenAIClient", fake_client_factory)
    monkeypatch.setattr(agent, "DB_PATH", db_path)

    assert (
        agent.cmd_ask(
            ["1", "List every event for track 47, with timestamps and types."]
        )
        == 0
    )
    capsys.readouterr()

    tool_message = [
        m for m in created[0].calls[1]["messages"] if m.get("role") == "tool"
    ][0]
    for event in stored:
        assert event.event_type in tool_message["content"]
        assert event.timestamp in tool_message["content"]

    db = init_db(db_path)
    traces = query_agent_traces(db, run_id=1)
    assert len(traces) == 1
    assert [call["name"] for call in traces[0]["tool_calls"]] == [
        "get_events_for_track"
    ]
    assert traces[0]["question"] == (
        "List every event for track 47, with timestamps and types."
    )
    db.close()
