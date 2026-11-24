"""Model-free text-simulation demo: parse JSONL scenarios and ingest them.

No import here touches YOLO, OpenCV, LM Studio or OpenAI. Everything in this
package is deterministic so the alert logic can be demonstrated without a GPU
or a network call.
"""

from __future__ import annotations

from src.text_demo.ingest import IngestResult, ingest_scenario, source_name
from src.text_demo.scenario import Scenario, ScenarioFrame, ScenarioObject, load_scenario

__all__ = [
    "IngestResult",
    "Scenario",
    "ScenarioFrame",
    "ScenarioObject",
    "ingest_scenario",
    "load_scenario",
    "source_name",
]
