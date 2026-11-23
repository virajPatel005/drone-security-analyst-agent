# Build Plan

Status key: [ ] todo, [~] in progress, [x] done. Update this file at the end of every step.

- [x] Step 0: Scaffolding (folders, .gitignore, .env.example, README skeleton, docs headings, first commit)
- [x] Step 1: Environment and checks (Python venv in WSL, ultralytics, confirm CUDA is used, confirm WSL can reach the LM Studio server)
- [x] Step 2: Sample video, frame sampling, simulated telemetry and simulated clock, plus text-only simulation mode
- [x] Step 3: YOLO + ByteTrack, convert tracks to events (appeared, entered zone, left, dwell time)
- [ ] Step 4: SQLite schema and indexing, queries by time and by object
- [ ] Step 5: Rule-based alerts
- [ ] Step 6: VLM on new-track crops via LM Studio, structured output, stored in the database
- [ ] Step 7: OpenAI agent: summary, recommendations, follow-up Q&A over stored data
- [ ] Step 8: Tests (fakes for models), scenario tests for midnight alert and truck logged
- [ ] Step 9: Streamlit UI for the demo
- [ ] Step 10: Scalability measurements, VLM comparison table, diagrams
- [ ] Step 11: README, PDF report, demo videos with voiceover, invite assignments@flytbase.com as collaborator, submit

## Rules
- One step at a time. Do not start a step until the human asks.
- Keep the simplest end-to-end path working before adding extras.
- Ask before Providing code and only when user asks.
- At the end of each step, propose the commit message and the file list, and wait for my approval before committing locally.
- Never push. I push myself.
- Never change code outside the requested step, and always tell me what was changed and why.

