# Chalksy: Autonomous AI Classroom Builder
### Stop Prompting, Solo Agents Hackathon (Lyzr × Qdrant × Omi)

> **Build Track**: **Collaborative Multi-Agent Workflows** *(also cross-qualifies for Meeting & Lecture Intelligence & Privacy-First Knowledge Systems)*  
> **Mandatory Loop**: **Omi Voice Capture ➔ Qdrant Vector Memory ➔ Lyzr Orchestrator Reasoning ➔ Memory Write-Back**  
> **Author**: Solo Participant  

---

## 🌟 Architecture & Mandatory Closed Loop

```
[Omi Wearable Necklace / Ambient Mic] ──► [FastAPI Gateway: /omi/conversation, /omi/realtime]
                                                      │
                                                      ▼
                                       [Salted PII Sanitization Engine]
                                     (Devanagari, Kannada, regex, obfuscation)
                                                      │
                                                      ▼
    ┌─────────────────────────────── [Lyzr Orchestrator Agent] ◄══════════════════════┐
    │                                (Builds typed execution DAG)                      │
    ▼                                                 │                                │
[Planner Agent] ◄─── Root-Gap Prerequisite KG         │                                │
[Worksheet Agent] ◄─ Spaced Repetition Scheduler      │                                │
[Critic Agent] ◄──── Contradiction & Syllabi Traps   │                                │
[Grader Agent] ───────────────────────────────────────┼───► Writes back to Qdrant      │
[Parent-Report Agent] (EN / HI / KN / SSML TTS)       │     student_events collection  │
[Diff / Assessment / Intervention / Substitute / Quiz]│                                │
    │                                                 │                                │
    ▼                                                 ▼                                │
[Human-in-the-Loop Teacher Approval Queue] ◄──────────┘                                │
    │                                                                                  │
    ▼                                                                                  │
[Export Engine: PDF / DOCX / WhatsApp API]                                             │
    │                                                                                  │
    ▼                                                                                  │
[Memory Consolidation Job: Raw Notes + Events ➔ Summarized Insights] ══════════════════╝
```

---

## 🚀 Comprehensive Capabilities (Phase 1 to Phase 7)

### 🧠 Phase 1: Autonomous Memory Loop
- **Memory Consolidation Job (`POST /api/memory/consolidate`)**: Clusters raw `teacher_notes` and `student_events` into high-level summaries written to the `insights` collection, down-weighting raw items without deleting provenance.
- **Contradiction Detection**: Flags teacher spoken claims that conflict with recent student error trends in vector memory, displaying side-by-side evidence in the approval queue.
- **Misconception Knowledge Graph (`GET /api/memory/knowledge-graph/trace`)**: Traces failed student concepts to root foundational gaps and automates remediation planning.
- **Spaced-Repetition Scheduler (`GET /api/memory/spaced-repetition/due`)**: Calculates memory decay ($R = e^{-\Delta t / S}$) and auto-seeds due review items into generated worksheets.
- **Hybrid vs. Dense Benchmark (`GET /api/memory/benchmark/retrieval`)**: Dense + sparse hybrid search with Reciprocal Rank Fusion.

### 👩‍🏫 Phase 2: Learning from Teachers & Measurable Evals
- **Teacher-Correction Learning (`POST /api/corrections/record`)**: Stores human teacher overrides as few-shot memory items in `corrections`, improving future grading and worksheet generations.
- **Rubric-Explained Confidence Grading**: Emits per-criterion scores with confidence metrics; auto-routes low-confidence evaluations ($< 0.70$) to human review.
- **Ablation Study (`GET /api/evals/ablation`)**: Quantifies accuracy, syllabus alignment, and personalization with vector memory ON vs. OFF.
- **Adversarial & Privacy Red-Team Suite**: Validates 100% catch rate against off-syllabus calculus, age hazards, and Devanagari/Kannada/obfuscated PII attacks.

### 🎙️ Phase 3: Voice, Language & Ambient Workflow
- **Code-Switching Detection**: Classifies Hinglish, Kanglish, and English markers directly from teacher speech.
- **Voice Approval Commands Router**: Recognizes spoken commands (*"approve the worksheet"*, *"sab approve kardo"*, *"make it easier"*) and updates tasks autonomously.
- **PTM Conference Intelligence (`POST /api/agents/ptm-conference`)**: Extracts parent concerns, teacher promises, and action items into a checklist.
- **End-of-Day 60-Second Reflection (`GET /api/agents/reflection/eod`)**: Generates rapid summaries of topics taught, confusion points, and action items for tomorrow.

### 🤝 Phase 4: Classroom Support Agents
- **Differentiation Agent (`POST /api/agents/differentiate`)**: ESL scaffolding, dyslexia-friendly spacing, and gifted extensions.
- **Assessment Designer (`GET /api/agents/assessment/blueprint`)**: Bloom-taxonomy blueprint tables with answers.
- **Intervention Agent (`GET /api/agents/intervention/plan`)**: Pinpoints at-risk cohorts and generates small-group remediation.
- **Substitute Handover Pack (`GET /api/agents/substitute/handover`)**: Generates handover plans with objectives and fallback games.
- **Quiz/Game Agent (`GET /api/agents/quiz-game/generate`)**: Generates printable Bingo cards and Kahoot questions.
- **Peer-Review Consensus**: Dual-agent proposal synthesis before final guardrail submission.

### 📸 Phase 5: Grading Inputs & Teacher Analytics
- **Handwritten Photo OCR Grading (`POST /api/agents/grade/photo`)**: Base64 photo ingestion and grading against lesson rubrics.
- **Curriculum Coverage Tracker (`GET /api/curriculum/coverage`)**: Syllabus coverage tracker for CBSE, ICSE, and Karnataka State boards.
- **WhatsApp Business API Formatter (`POST /api/agents/parent-report/whatsapp`)**: Generates 160-char template-ready WhatsApp payload.
- **Voice TTS Parent Report (`POST /api/agents/parent-report/tts`)**: SSML markup ready for Google Cloud TTS, AWS Polly, or Azure Speech.
- **Teacher Analytics Dashboard Heatmap (`GET /api/analytics/teacher`)**: Misconception frequency heatmap (Concept × Week).

### 🔒 Phase 6: Privacy Hardening
- **Consent Registry (`POST /api/privacy/consent`)**: Granular, per-student, per-collection opt-in/opt-out with expiration support.
- **Hash-Chained Audit Log (`GET /api/privacy/audit-log/verify`)**: Tamper-evident HMAC SHA-256 chain verifying log integrity.
- **Laplace Differential Privacy (`POST /api/privacy/dp-aggregate`)**: Adds differential privacy noise to aggregate scores to prevent re-identification.
- **Extended Right-to-be-Forgotten (`POST /api/privacy/rtbf`)**: Cascade purge across registry, consent records, submission fingerprints, and Qdrant vectors.
- **Plagiarism Advisory (`POST /api/privacy/plagiarism-check`)**: Jaccard similarity detection across student submissions.

### 🎨 Phase 7: Demo Polish & Warm Notebook GUI
- **"Prepare My Monday" (`POST /api/agents/prepare-monday`)**: Chained one-click command running EOD Reflection ➔ Spaced Rep ➔ Consolidation ➔ Plan ➔ Worksheet ➔ Critic.
- **Memory Timeline View (`GET /api/memory/timeline`)**: Color-coded timeline of memories across all Qdrant collections.
- **Live PII Scrub Diff View (`POST /api/privacy/pii-diff`)**: Side-by-side token diff showing original vs. scrubbed input.
- **Memory On/Off Live Toggle (`POST /api/settings/memory-toggle`)**: Toggle Qdrant vector memory on/off in real-time to demonstrate ablation.
- **Warm Notebook GUI**: Pastel school report card style (lavender, gold, soft blue, pink) with bold underlined headers and Kalam font accents.

---

## 🛠️ Quickstart

### 1. Configure Environment
```bash
cp .env.example .env
```
Fill in your API credentials (or run in offline fallback mode with built-in cognitive engine):
```ini
LYZR_API_KEY=your_lyzr_api_key
LYZR_AGENT_ID=your_lyzr_agent_id
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=  # Leave blank for local embedded Qdrant
```

### 2. Seed Synthetic Classroom & Curriculum
```bash
python seed_data.py
```

### 3. Run the Application
```bash
# Windows / PowerShell:
run.bat
# Or manually:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

### 4. Run the Full Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```
*(30 tests covering all 7 phases pass with 100% success).*

---

## 📜 License
MIT License
