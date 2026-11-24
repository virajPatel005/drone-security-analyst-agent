"""Ingest a JSONL text-simulation scenario and run the existing alert rules.

Example:
    .venv/bin/python scripts/run_text_demo.py \
        --scenario data/scenarios/midnight_gate.jsonl --db data/text_demo.db

The default DB is a dedicated file. Writing to data/demo.db is refused so the
OpenAI cost ledger, summaries and traces stored there are never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts import ALERT_RULES
from src.config.settings import DB_PATH, TEXT_DEMO_DB_PATH, TEXT_DEMO_VISIT_GAP_S
from src.storage import init_db
from src.text_demo import ingest_scenario, load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True, help="JSONL scenario file")
    parser.add_argument("--db", type=Path, default=TEXT_DEMO_DB_PATH)
    args = parser.parse_args(argv)

    if args.db.resolve() == Path(DB_PATH).resolve():
        parser.error(
            "refusing to write to the protected demo.db; use the default "
            f"{TEXT_DEMO_DB_PATH} or another separate file"
        )

    scenario = load_scenario(args.scenario)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(args.db)
    try:
        result = ingest_scenario(conn, scenario, visit_gap_s=TEXT_DEMO_VISIT_GAP_S)
        print(f"scenario: {args.scenario}")
        print(f"db: {args.db.resolve()}")
        print(f"run_id: {result.run_id}")
        print(f"source: {result.source}")
        print(
            f"frames={result.frames} objects={result.objects} "
            f"tracks={result.tracks} events={result.events}"
        )
        print("tracks:")
        for row in conn.execute(
            "SELECT track_id, class_name, first_seen, last_seen "
            "FROM tracks WHERE run_id = ? ORDER BY track_id",
            (result.run_id,),
        ):
            print(f"  track={row[0]} class={row[1]} first_seen={row[2]} last_seen={row[3]}")
        print("alerts:")
        for rule_name in ALERT_RULES:
            print(f"  {rule_name}: {result.alerts.get(rule_name, 0)}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
