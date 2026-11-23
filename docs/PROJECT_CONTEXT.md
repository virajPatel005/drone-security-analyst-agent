# Project Context

## Goal
A docked drone patrols a fixed property daily. The system takes drone telemetry (time, location, altitude) and video, detects and tracks objects, describes them, logs events with context, raises rule-based security alerts, indexes every frame for search, and has an agent that summarizes the video and answers follow-up questions.

## Required deliverables
- Feature spec (value to property owners, 2 to 3 key requirements)
- Architecture and flow diagrams
- Prototype: simulated telemetry plus video frames, object/event logging, rule-based alerts, at least one AI-generated component
- Frame-by-frame index in a database, queryable by time or object (for example "show all truck events")
- Test cases (for example "truck logged correctly", "alert fires at midnight")
- README with setup and design decisions, PDF report, demo videos with voiceover
- Bonus: one-sentence video summary, agent that answers follow-up questions

## Sample expected outputs
- Log: "Blue Ford F150 spotted at garage, 12:00."
- Alert: "Person loitering at main gate, 00:01."
- Queryable frames by time or object.

## Grading weights
Correctness and model performance 25 percent, reasoning and scalability 25 percent (agent uses context across the whole video), experimentation and innovation 30 percent, documentation and code quality 20 percent. The report carries equal weight to the code.

## Architecture
1. Perception: ultralytics YOLO with ByteTrack on sampled frames. Gives class, count, box and persistent track ID. Track IDs enable "same vehicle entered twice" and dwell-time loitering.
2. Description: local VLM Gemma 4 E4B served by LM Studio. Called on the CROP of each newly tracked object (and optionally one scene description per keyframe). Cropping fixes the resolution problem seen in testing.
3. Reasoning: OpenAI mini model over stored text. Summary, recommendations, follow-up Q&A.
4. Storage: SQLite with tables for frames (timestamp plus telemetry), tracks, events, alerts, plus full-text search. A vector index is optional and only if time allows.
5. UI: Streamlit, for demo recordings.

## Telemetry and time
- Telemetry is simulated: timestamp, location label (gate, garage, driveway, perimeter), altitude, drone position.
- Video time is mapped to a configurable simulated clock (for example starting at 23:55) so night and midnight rules can be demonstrated.
- A text-only simulation mode ("Frame 1: blue truck at gate") must also exist so tests are deterministic and need no models.

## Alert rules (rule-based, configurable)
- Person detected in a restricted zone during night hours
- Loitering: same person track stays in one zone longer than a threshold
- Same vehicle track or repeated vehicle entries above a threshold in a day
- Unknown vehicle at night

## VLM findings from our own testing (use these)
- Gemma 4 E4B was clearly better than E2B. Qwen 3.5 9B was also tested. E4B is the chosen model.
- Small or crowded images made the VLM undercount badly. Forcing enumeration, adding count fields and describing region by region improved results a lot.
- Same prompt in one long chat contaminated results. Every call must be stateless with fresh context. Use temperature 0.
- The VLM missed an obvious large camera and gave a confident false negative. Treat VLM output as fallible, include a confidence field, and never let it be the only source for counts. YOLO is the source for counts.
- Occlusion in packed scenes remains a known limit. State it in the report.

## VLM output contract (per object crop)
Fields: category (vehicle, person, other), type, color, activity, confidence (low, medium, high), optional make_model that must be null when unsure.
- Vehicle type vocabulary: car, suv, sedan, pickup, truck, van, bus, motorbike, auto-rickshaw, other.
- Color vocabulary: white, black, silver, grey, red, blue, green, yellow, brown, other, unknown.
- People appear only in the people category, never duplicated as objects.
- Say "unknown" instead of guessing. Never guess plates.

## Agent capabilities
- Answer from the database: "show all truck events", "what happened between 00:00 and 01:00", "how many times did vehicle X enter".
- One-sentence summary of the whole video, plus a longer recommendation list for the property owner.
- Follow-up Q&A over stored events with context carried across the whole video.

## Scalability story
Detection runs locally on every sampled frame, the VLM runs only on new tracks, and only compact text goes to the cloud. Measure and record frames per second, VLM calls per video and OpenAI token cost.