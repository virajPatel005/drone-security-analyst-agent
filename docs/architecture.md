# Architecture and Data Flow

This page embeds the diagrams whose canonical text sources live in `docs/diagrams/*.mmd`.
The Mermaid blocks below are the same source, inlined so GitHub renders them directly.

## Three layers and storage

```mermaid
flowchart TB
    subgraph PERCEPTION["1. Perception (local, GPU)"]
        V["Sampled video frames"] --> S["Frame sampler<br/>seek + decode"]
        S --> Y["YOLO11s + ByteTrack"]
        Y --> O["Observations<br/>class, box, track ID"]
    end

    subgraph DESCRIPTION["2. Description (local VLM)"]
        O --> C["One crop per NEW track"]
        C --> L["LM Studio Gemma<br/>stateless, 1 call/track"]
        L --> T["Text description"]
    end

    subgraph STORAGE["Storage (SQLite, text only)"]
        DB[("runs, tracks, events,<br/>frames, frame_objects,<br/>alerts, descriptions,<br/>agent tables")]
    end

    subgraph REASONING["3. Reasoning (cloud, text only)"]
        DB --> A["OpenAI mini model<br/>summary + follow-up Q&A"]
        A --> ANS["Answer from stored text"]
    end

    O --> DB
    T --> DB
    Y --> DB
```

Perception runs locally on sampled frames and is the only layer that sees pixels. The
description layer sees one crop per newly confirmed track, never every frame, and each
call is stateless. The reasoning layer reads **stored text only** from SQLite, so no
image is ever sent to OpenAI.

## Data flow: video to alerts to agent

```mermaid
flowchart LR
    V["Video + simulated<br/>clock / telemetry"] --> F["Sampled frames"]
    F --> Y["YOLO + ByteTrack"]
    Y --> E["Events: appeared,<br/>entered_zone, left"]
    Y --> IX["Frame index:<br/>frames + frame_objects"]
    E --> DB[("SQLite")]
    IX --> DB
    DB --> RULES["Rule engine"]
    RULES --> AL["Alerts"]
    AL --> DB
    DB --> AG["Agent: summary / ask<br/>stored text only"]
    AG --> OUT["Answer + cost row"]
    OUT --> DB
    DB --> UI["Streamlit UI (read-only)"]
```

Everything is derived from stored rows: alerts are computed from `tracks`/`events` after
storage, and agent answers come from the same database. The UI opens SQLite read-only.

## Database schema

```mermaid
erDiagram
    RUNS ||--o{ TRACKS : "has"
    RUNS ||--o{ EVENTS : "has"
    RUNS ||--o{ FRAMES : "indexes"
    FRAMES ||--o{ FRAME_OBJECTS : "contains"
    RUNS ||--o{ ALERTS : "raises"
    RUNS ||--o{ DESCRIPTIONS : "describes"
    RUNS ||--o| RUN_SUMMARIES : "has"
    RUNS ||--o{ AGENT_COSTS : "bills"
    RUNS ||--o{ AGENT_TRACES : "audits"

    RUNS {
        int id PK
        text source_video
        text start_time
        text tuning_settings
    }
    TRACKS {
        int run_id PK
        int track_id PK
        text class_name
        text first_seen
        text last_seen
    }
    EVENTS {
        int id PK
        int run_id
        int track_id
        text event_type
        text timestamp
        text zone
    }
    FRAMES {
        int run_id PK
        int frame_index PK
        text timestamp
        text zone
        int detection_count
    }
    FRAME_OBJECTS {
        int id PK
        int run_id
        int frame_index
        int track_id
        text class_name
        real confidence
    }
    ALERTS {
        int id PK
        int run_id
        int track_id
        text rule_name
        text severity
    }
    DESCRIPTIONS {
        int id PK
        int run_id
        int track_id
        text status
        text description
        real latency_ms
    }
    RUN_SUMMARIES {
        int run_id PK
        text summary_json
        int prompt_tokens
        int completion_tokens
    }
    AGENT_COSTS {
        int id PK
        int run_id
        text command
        real estimated_cost_usd
        real latency_s
    }
    AGENT_TRACES {
        int id PK
        int run_id
        text mode
        text answer
        real estimated_cost_usd
    }
```

`frames` holds one row per sampled frame; `frame_objects` holds one row per detection.
`agent_costs` is the billing ledger and `agent_traces` is the per-answer audit; both are
written by the agent layer, never by perception or description.

## Alert rule flow

```mermaid
flowchart TD
    A["Stored tracks + events"] --> B["vehicle_at_night:<br/>first-seen hour in night window?"]
    A --> C["person_at_night:<br/>person first-seen hour in night window?"]
    A --> D["loitering:<br/>last_seen - first_seen > 5 s?"]
    A --> E["heavy_vehicle_logged:<br/>class is truck or bus?"]

    B -->|yes| W["warning alert"]
    C -->|yes| W
    D -->|yes| W
    E -->|yes| I["info alert"]

    W --> U["UNIQUE(run_id, track_id, rule_name)<br/>= one alert per track per rule"]
    I --> U
    U --> DB[("alerts table")]
```

The `UNIQUE` constraint plus `INSERT OR IGNORE` makes alert generation idempotent.
Alerts count tracker tracks, not verified physical objects.

## Text-simulation path

```mermaid
flowchart LR
    J["JSONL scenario<br/>hand-written frames"] --> P["Parser:<br/>validate time, class, order"]
    P --> R["FrameRecord<br/>description filled, image = None"]
    R --> B["Frame index rows"]
    R --> T["Tracks + events<br/>from stable labels"]
    B --> DB[("data/text_demo.db")]
    T --> DB
    DB --> RULES["SAME 4 alert rules<br/>read stored rows"]
    RULES --> DB
    DB --> Q["query_frames:<br/>time / class / track"]

    N["No YOLO, VLM or OpenAI call"]:::note
    classDef note fill:#f6f6f6,stroke:#888,stroke-dasharray: 4 3;
```

Text mode reuses the storage, frame-index and alert code unchanged, and writes to a
separate database so it cannot disturb the OpenAI cost ledger in `data/demo.db`.

## Syntax check and PNG export

Each block was checked by inspection against Mermaid's flowchart/ER grammar (balanced
quotes and brackets, valid edge and attribute syntax). No Mermaid renderer (`mmdc` /
`mermaid-cli`) is installed in this environment, so **no PNG export was produced and
nothing was installed**. The `.mmd` files render on GitHub when opened as Mermaid, and
the inlined blocks above render in this Markdown file.
