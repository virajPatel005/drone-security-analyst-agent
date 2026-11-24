"""OpenAI agent for reasoning over stored drone data.

The reasoning layer has one client, built from the OpenAI-specific settings.
It never uses LM Studio settings, never sees an image, and never calls a model
to re-describe a crop.  Tools read stored text from SQLite only.

Rules (non-negotiable):
  * No images, no crop files, no base64 in any request.
  * Answers come from stored text only, never from re-running VLM/video.
  * Model calls sit behind OpenAIClient so tests can use a fake.
  * Every chat call is recorded in agent_costs; the running total lives in
    SQLite, not in process memory.
  * Round cap (default 4) stops a runaway tool loop.
  * One summary per run (INSERT OR REPLACE); second call returns cached row.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import openai

from src.config.settings import (
    DB_PATH,
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL,
    OPENAI_COST_CAP_USD,
    OPENAI_INPUT_COST_PER_MILLION,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_MAX_ROUNDS,
    OPENAI_MODEL_NAME,
    OPENAI_OUTPUT_COST_PER_MILLION,
    OPENAI_TOOL_ROW_LIMIT,
    OPENAI_WARNING_USD,
)
from src.storage import (
    init_db,
    query_agent_cost_total,
    query_alerts,
    query_counts_by_event_type,
    query_descriptions,
    query_events_for_track,
    query_run_summary,
    query_tracks_by_class,
    record_agent_call,
    store_run_summary,
)

LOG = logging.getLogger(__name__)

ALLOWED_TOOLS = frozenset(
    {
        "list_runs",
        "get_alerts",
        "get_tracks",
        "get_events_for_track",
        "get_descriptions",
        "counts_by_event_type",
    }
)


class BudgetExceeded(RuntimeError):
    """Raised before a call that would start above the configured USD cap."""


def openai_api_key() -> str:
    """Read the key from the environment; never log it."""
    key = os.getenv(OPENAI_API_KEY_ENV, "")
    if not key:
        raise RuntimeError(f"Environment variable {OPENAI_API_KEY_ENV} not set")
    return key


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD for a call using the configured per-million-token prices."""
    return (prompt_tokens / 1_000_000.0) * OPENAI_INPUT_COST_PER_MILLION + (
        completion_tokens / 1_000_000.0
    ) * OPENAI_OUTPUT_COST_PER_MILLION


class OpenAIClient:
    """One OpenAI client for both summary and ask, with a SQLite-backed guard.

    ``conn`` is optional only for non-chat calls such as model listing.  Chat
    calls without a connection are still made, but cannot be charged to the
    persistent total, so command paths must always pass one.
    """

    def __init__(
        self,
        *,
        conn: Any = None,
        run_id: int | None = None,
        command: str = "",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or OPENAI_MODEL_NAME
        self.base_url = base_url or OPENAI_BASE_URL
        self.conn = conn
        self.run_id = run_id
        self.command = command
        self._client = openai.OpenAI(
            api_key=api_key if api_key is not None else openai_api_key(),
            base_url=self.base_url,
        )

    def list_model_names(self) -> list[str]:
        """Return model IDs available to this key, with no key material printed."""
        return [model.id for model in self._client.models.list()]

    def running_total_usd(self) -> float:
        """Read the accumulated spend from SQLite."""
        if self.conn is None:
            return 0.0
        return query_agent_cost_total(self.conn)

    def _assert_budget(self) -> None:
        """Refuse a new chat call once earlier calls have reached the cap."""
        if self.conn is None:
            return
        total = self.running_total_usd()
        if total >= OPENAI_COST_CAP_USD:
            raise BudgetExceeded(
                f"OpenAI budget cap reached: ${total:.4f} >= ${OPENAI_COST_CAP_USD:.2f}"
            )

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        tools: List[Dict[str, Any]] | None,
    ) -> Any:
        """One chat-completions call using the current tools/tool_choice API."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self._client.chat.completions.create(**kwargs)

    def call(
        self,
        *,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        tools: List[Dict[str, Any]] | None = None,
    ) -> dict:
        """Execute one chat turn, record its usage, and return its response."""
        self._assert_budget()
        started = time.perf_counter()
        completion = self._chat(messages, max_tokens, tools)
        latency = time.perf_counter() - started

        choice = completion.choices[0]
        message = choice.message
        tool_calls = []
        for call in getattr(message, "tool_calls", None) or []:
            raw_args = call.function.arguments
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)
            tool_calls.append(
                {"id": call.id, "name": call.function.name, "arguments": raw_args}
            )

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = estimate_cost(prompt_tokens, completion_tokens)

        total = 0.0
        if self.conn is not None:
            record_agent_call(
                self.conn,
                run_id=self.run_id,
                command=self.command,
                model_name=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=cost,
                latency_s=latency,
            )
            total = self.running_total_usd()
            LOG.info(
                "OpenAI call model=%s prompt_tokens=%d completion_tokens=%d "
                "estimated_cost_usd=%.6f latency_s=%.3f running_total_usd=%.6f",
                self.model,
                prompt_tokens,
                completion_tokens,
                cost,
                latency,
                total,
            )
            # A warning is informational; only the cap refuses further work.
            if total >= OPENAI_WARNING_USD:
                LOG.warning(
                    "OpenAI spend warning: running total $%.4f >= $%.2f",
                    total,
                    OPENAI_WARNING_USD,
                )

        return {
            "content": message.content or "",
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost_usd": cost,
            "latency_s": latency,
            "running_total_usd": total,
            "model": self.model,
        }


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_runs",
            "description": "Return a compact JSON list of runs, or one run if run_id is given.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alerts",
            "description": "Return alerts for an optional run, rule, and/or time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "rule_name": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tracks",
            "description": "Return tracks for an optional run and/or class name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "class_name": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_for_track",
            "description": "Return every stored event for one track, ordered by time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "track_id": {"type": "integer"},
                },
                "required": ["run_id", "track_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_descriptions",
            "description": "Return stored VLM text descriptions for an optional run and track.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "track_id": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "counts_by_event_type",
            "description": "Return event_type -> count for all events, or for one run.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "integer"}},
                "required": [],
            },
        },
    },
]


def compact_rows(rows: List[Any], limit: int | None = None) -> Dict[str, Any]:
    """Wrap rows with a total and a visible truncation notice.

    The old code silently dropped rows to fit a local 8K context.  Here every
    list reports its real size, and truncation is explicit when it happens.
    """
    row_limit = OPENAI_TOOL_ROW_LIMIT if limit is None else limit
    total = len(rows)
    shown = list(rows[:row_limit])
    payload: Dict[str, Any] = {
        "total": total,
        "showing": len(shown),
        "truncated": total > len(shown),
        "rows": shown,
    }
    if payload["truncated"]:
        payload["notice"] = f"showing {len(shown)} of {total}"
    return payload


def _json_row(row: Any, keys: List[str]) -> Dict[str, Any]:
    """Map a sqlite row or dict to the named fields, tolerating both."""
    if isinstance(row, dict):
        return {key: row.get(key) for key in keys}
    return {key: row[index] for index, key in enumerate(keys)}


def _run_tool(tool_name: str, args: dict, db: Any) -> str:
    """Execute one allowlisted tool and return compact JSON text."""
    if tool_name not in ALLOWED_TOOLS:
        return json.dumps({"error": f"unknown tool: {tool_name}"})

    if tool_name == "list_runs":
        run_id = args.get("run_id")
        if run_id is None:
            rows = db.execute(
                "SELECT id, source_video, start_time FROM runs ORDER BY id"
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, source_video, start_time FROM runs WHERE id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        result = [
            {"id": r[0], "source_video": r[1], "start_time": r[2]} for r in rows
        ]
        return json.dumps(compact_rows(result))

    if tool_name == "get_alerts":
        rows = query_alerts(
            db,
            run_id=args.get("run_id"),
            rule_name=args.get("rule_name"),
            start=args.get("start"),
            end=args.get("end"),
        )
        fields = ["id", "run_id", "track_id", "rule_name", "severity", "timestamp", "reason", "zone"]
        return json.dumps(compact_rows([_json_row(r, fields) for r in rows]))

    if tool_name == "get_tracks":
        rows = query_tracks_by_class(
            db, class_name=args.get("class_name"), run_id=args.get("run_id")
        )
        fields = ["track_id", "class_name", "first_seen", "last_seen", "frames_seen", "max_confidence"]
        result = [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "first_seen": t.first_seen,
                "last_seen": t.last_seen,
                "frames_seen": t.frames_seen,
                "max_confidence": t.max_confidence,
            }
            for t in rows
        ]
        return json.dumps(compact_rows(result))

    if tool_name == "get_events_for_track":
        rows = query_events_for_track(
            db, run_id=args["run_id"], track_id=args["track_id"]
        )
        result = [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "class_name": e.class_name,
                "zone": e.zone,
                "dwell_s": e.dwell_s,
            }
            for e in rows
        ]
        return json.dumps(compact_rows(result))

    if tool_name == "get_descriptions":
        rows = query_descriptions(
            db, run_id=args.get("run_id"), track_id=args.get("track_id")
        )
        result = [
            {
                "track_id": d["track_id"],
                "status": d["status"],
                "description": d["description"],
                "main_colour": (d.get("attributes") or {}).get("main_colour", ""),
            }
            for d in rows
        ]
        return json.dumps(compact_rows(result))

    # counts_by_event_type
    counts = query_counts_by_event_type(db, run_id=args.get("run_id"))
    return json.dumps({"counts": dict(counts)})


def _build_tool_results(db: Any, run_id: int) -> str:
    """Preload compact, text-only context for the one-shot summary prompt."""
    alerts = query_alerts(db, run_id=run_id)
    alert_rows = [
        {
            "rule_name": a["rule_name"],
            "severity": a["severity"],
            "timestamp": a["timestamp"],
            "reason": a["reason"],
            "track_id": a["track_id"],
        }
        for a in alerts
    ]

    truck_tracks = query_tracks_by_class(db, class_name="truck", run_id=run_id)
    truck_rows = [
        {
            "track_id": t.track_id,
            "class_name": t.class_name,
            "first_seen": t.first_seen,
            "last_seen": t.last_seen,
        }
        for t in truck_tracks
    ]
    truck_events = []
    for track in truck_tracks:
        for event in query_events_for_track(db, run_id=run_id, track_id=track.track_id):
            truck_events.append(
                {
                    "track_id": track.track_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                }
            )

    descriptions = query_descriptions(db, run_id=run_id)
    description_rows = [
        {
            "track_id": d["track_id"],
            "status": d["status"],
            "description": d["description"],
            "main_colour": (d.get("attributes") or {}).get("main_colour", ""),
        }
        for d in descriptions
    ]
    counts = query_counts_by_event_type(db, run_id=run_id)

    parts = [
        "alerts: " + json.dumps(compact_rows(alert_rows)),
        "truck_tracks: " + json.dumps(compact_rows(truck_rows)),
        "truck_events: " + json.dumps(compact_rows(truck_events)),
        "descriptions: " + json.dumps(compact_rows(description_rows)),
        "event_counts: " + json.dumps(dict(counts)),
    ]
    return "; ".join(parts)


def _parse_arguments(raw: str) -> dict:
    """Turn tool-call arguments into a dict, returning an error for bad JSON."""
    try:
        parsed = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {"_error": "tool arguments were not valid JSON"}
    return parsed if isinstance(parsed, dict) else {"_error": "tool arguments were not an object"}


def run_conversation(
    client: Any,
    *,
    system: str,
    user: str,
    db: Any,
    max_rounds: int | None = None,
) -> dict:
    """Run one bounded tools/auto conversation and return text plus usage stats."""
    rounds = OPENAI_MAX_ROUNDS if max_rounds is None else max_rounds
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    stats = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
        "cost_usd": 0.0,
        "running_total_usd": 0.0,
        "model": getattr(client, "model", OPENAI_MODEL_NAME),
    }
    final_text = ""
    for _ in range(rounds):
        result = client.call(
            messages=messages,
            max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            tools=TOOL_DEFINITIONS,
        )
        stats["calls"] += 1
        stats["prompt_tokens"] += result["prompt_tokens"]
        stats["completion_tokens"] += result["completion_tokens"]
        stats["latency_s"] += result["latency_s"]
        stats["cost_usd"] += result["estimated_cost_usd"]
        stats["running_total_usd"] = result["running_total_usd"]
        stats["model"] = result["model"]

        tool_calls = result["tool_calls"]
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": result["content"],
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in tool_calls
            ]
            messages.append(assistant_message)
            for call in tool_calls:
                tool_result = _run_tool(call["name"], _parse_arguments(call["arguments"]), db)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    }
                )
            final_text = result["content"]
            continue

        messages.append(assistant_message)
        final_text = result["content"]
        break
    else:
        suffix = f"Stopped after {rounds} rounds without a final answer."
        final_text = f"{final_text}\n{suffix}" if final_text else suffix

    return {"content": final_text, "stats": stats}


def _print_stats(stats: dict) -> None:
    """Print one compact usage line after each command."""
    print(
        "stats: "
        f"model={stats['model']} "
        f"prompt_tokens={stats['prompt_tokens']} "
        f"completion_tokens={stats['completion_tokens']} "
        f"latency_s={stats['latency_s']:.3f} "
        f"cost_usd={stats['cost_usd']:.6f} "
        f"running_total_usd={stats['running_total_usd']:.6f}"
    )


_SUMMARY_SYSTEM = (
    "You are an analyst for a drone security system.  Use only the stored text "
    "in the tool results.  Reply with ONE JSON object and no other text, with "
    "exactly these keys: \"summary\" (one sentence), \"notable_alerts\" (array "
    "of objects with track_id, rule_name, severity, timestamp, reason), "
    "\"questionable\" (array of track IDs whose alerts look questionable given "
    "the VLM descriptions), and \"recommendations\" (3-5 short strings).  Do "
    "not invent IDs, timestamps, colours or events.  Alerts count tracks, not "
    "physical objects; the clock is simulated; low-resolution aerial VLM "
    "descriptions can be vague.  If a truck alert's description says a building "
    "or a container, flag it as questionable.  If a detail is not in the data, "
    "say so plainly."
)

_ASK_SYSTEM = (
    "You are an analyst for a drone security system.  Use only the stored text "
    "returned by the tools.  Cite run and track IDs and timestamps.  Alerts "
    "count tracks, not physical objects; ID switching can split one object into "
    "several IDs; the clock is simulated; low-resolution aerial VLM descriptions "
    "can be vague, so flag alerts whose description contradicts the class.  If "
    "the stored data does not contain the requested detail, answer exactly "
    "\"the data cannot tell\" rather than guessing."
)

_SUMMARY_USER_TEMPLATE = (
    "Build a compact digest for run {run_id}.\n\nStored data:\n{tool_results}"
)
_ASK_USER_TEMPLATE = "Question: {question}\n\nStored data:\n{tool_results}"


def cmd_summary(args: List[str]) -> int:
    """CLI: summary <run-id> [--force]."""
    if len(args) < 1:
        print("Usage: ask_agent.py summary <run-id> [--force]", file=sys.stderr)
        return 1

    try:
        run_id = int(args[0])
    except ValueError:
        print("Run id must be an integer", file=sys.stderr)
        return 1
    force = "--force" in args

    db = init_db(DB_PATH)
    cached = query_run_summary(db, run_id=run_id)
    if cached and not force:
        print("Cached summary (no new model call):")
        print(cached["text"])
        _print_stats(
            {
                "model": cached["model_name"],
                "prompt_tokens": cached["prompt_tokens"],
                "completion_tokens": cached["completion_tokens"],
                "latency_s": 0.0,
                "cost_usd": 0.0,
                "running_total_usd": query_agent_cost_total(db),
            }
        )
        db.close()
        return 0

    tools_text = _build_tool_results(db, run_id)
    client = OpenAIClient(conn=db, run_id=run_id, command="summary")
    conversation = run_conversation(
        client,
        system=_SUMMARY_SYSTEM,
        user=_SUMMARY_USER_TEMPLATE.format(run_id=run_id, tool_results=tools_text),
        db=db,
    )
    stats = conversation["stats"]
    store_run_summary(
        db,
        run_id=run_id,
        text=conversation["content"],
        model_name=stats["model"],
        prompt_tokens=stats["prompt_tokens"],
        completion_tokens=stats["completion_tokens"],
        estimated_cost_usd=stats["cost_usd"],
    )
    print("Summary (OpenAI, tools auto):")
    print(conversation["content"])
    _print_stats(stats)
    db.close()
    return 0


def cmd_ask(args: List[str]) -> int:
    """CLI: ask <run-id> <question> [--rounds N]."""
    if len(args) < 2:
        print("Usage: ask_agent.py ask <run-id> <question> [--rounds N]", file=sys.stderr)
        return 1

    try:
        run_id = int(args[0])
    except ValueError:
        print("Run id must be an integer", file=sys.stderr)
        return 1

    max_rounds = OPENAI_MAX_ROUNDS
    question_parts: List[str] = []
    index = 1
    while index < len(args):
        if args[index] == "--rounds" and index + 1 < len(args):
            max_rounds = int(args[index + 1])
            index += 2
            continue
        question_parts.append(args[index])
        index += 1
    question = " ".join(question_parts)

    db = init_db(DB_PATH)
    tools_text = _build_tool_results(db, run_id)
    client = OpenAIClient(conn=db, run_id=run_id, command="ask")
    conversation = run_conversation(
        client,
        system=_ASK_SYSTEM,
        user=_ASK_USER_TEMPLATE.format(question=question, tool_results=tools_text),
        db=db,
        max_rounds=max_rounds,
    )
    print(conversation["content"])
    _print_stats(conversation["stats"])
    db.close()
    return 0
