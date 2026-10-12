<div align="center">

# Chalksy
### Autonomous AI Classroom Agent

**Voice to Lesson Plans to Worksheets to Parent Reports**
with full Human-in-the-Loop Teacher Approval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Lyzr](https://img.shields.io/badge/Powered%20by-Lyzr-purple.svg)](https://lyzr.ai)
[![Qdrant](https://img.shields.io/badge/Memory-Qdrant%20Cloud-red.svg)](https://qdrant.tech)

> **Hackathon Track**: Collaborative Multi-Agent Workflows
> **Stack**: Lyzr - Qdrant - OpenRouter - FastAPI - Python

</div>

---

## What is Chalksy?

Chalksy is an autonomous multi-agent AI system that acts as a co-teacher. A teacher speaks naturally about their classroom. Chalksy listens, understands, and automatically generates lesson plans, differentiated worksheets, and multilingual parent reports. Everything goes through a human-in-the-loop approval queue before reaching students or parents.

---

## Architecture

Full interactive diagram: [gitdiagram.com/Obedientononline/Chalksy](https://gitdiagram.com/obedientononline/chalksy)

```
TEACHER (User)
  Speaks / Types input
        |
        v
CAPTURE AND PRIVACY LAYER
  FastAPI Gateway  (/api/omi/webhook)
        |
        |---> PII Sanitizer (anonymizer.py)
        |     Scrubs names, phones, emails
        |
        v
AGENT WORKFLOW  (crew.py)
        |
        v
  Planner / Orchestrator (crew.py)
        |
        |-- Grading and Assessment      -- Worksheets and Quizzes
        |-- LLM Provider Engine         -- Parent Reports
        |-- Classroom Support Agents    -- Curriculum Alignment
        |-- Lesson Planning
        |
        v
  Runtime Config (config.py)
        |
        |
        |----> TEACHER APPROVAL QUEUE      LLM PROVIDERS        CLASSROOM MEMORY
               Artifact Exporter           OpenRouter            Memory Ingestion
               (exporter.py)               Lyzr Studio           (qdrant_store.py)
               Teacher Dashboard           Gemini Free
               (dag_state.db)                                     Qdrant Cloud Store
                                                                  (qdrant_store.py)
```

---

## Features

### Voice to Action Pipeline
- Teacher speaks naturally and transcript is auto-processed
- PII scrubbing (names, phones, emails) before any LLM call
- Hinglish and multilingual support

### Multi-Agent DAG

| Agent | Role |
|-------|------|
| Planner | Classifies intent, identifies learning gaps, routes tasks |
| Worksheet Agent | Generates 3-tier differentiated questions (Basic / Medium / Challenge) |
| Critic / Guardrail | Validates curriculum alignment, flags off-syllabus content |
| Grader Agent | Rubric-based grading with per-criterion confidence scores |
| Parent Report Agent | Multilingual reports (EN / HI / KN) + WhatsApp + TTS-ready |

### Human-in-the-Loop
- All AI outputs go to Teacher Approval Queue first
- Approve / Reject / Restore with full history tabs
- Export approved content as PDF or DOCX

### Qdrant Cloud Memory (6 Collections)

| Collection | What is stored |
|------------|----------------|
| teacher_notes | Raw classroom observations |
| curriculum | Syllabus alignment knowledge |
| student_events | Grading outcomes and misconceptions |
| artifacts | Generated plans and worksheets |
| insights | Consolidated memory summaries |
| corrections | Teacher overrides for few-shot learning |

### BYOK - Bring Your Own API Keys
- Visitors can enter their own OpenRouter / Lyzr keys via the API Keys modal
- Keys stored in browser localStorage only
- Server .env keys are never exposed to visitors

### Privacy First
- Salted pseudonymization with Right-to-be-Forgotten (RTBF)
- Laplace differential privacy on aggregate scores
- Hash-chained tamper-evident audit log

---

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/Obedientononline/Chalksy.git
cd Chalksy
cp .env.example .env
```

Edit `.env` with your keys (all free tiers available):

```ini
# Free from https://openrouter.ai/keys
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Free from https://cloud.qdrant.io
QDRANT_URL=https://xxxx.qdrant.io:6333
QDRANT_API_KEY=your_key_here

# Free from https://studio.lyzr.ai
LYZR_API_KEY=your_key_here
LYZR_AGENT_ID=your_agent_id
```

### 2. Install and Seed

```bash
pip install -r requirements.txt
python seed_data.py
```

### 3. Run

```bash
# Windows
run.bat

# Mac / Linux
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

---

## Demo Script (5 Minutes)

| Step | Action | What to show |
|------|--------|-------------|
| 1 | Click Sample 1 preset then Ingest Voice | PII scrub diff, live DAG |
| 2 | Open Teacher Approval Queue | Pending worksheet with 3-tier questions |
| 3 | Click Approve then Export PDF | Generated student handout |
| 4 | Submit a student answer with a mistake | Misconception detection + Qdrant write-back |
| 5 | Click Prepare My Monday | Full chained 6-agent autonomous run |
| 6 | Click API Keys in header | BYOK modal - visitor enters own key |

---

## Project Structure

```
Chalksy/
├── app/
│   ├── main.py              # FastAPI routes and endpoints
│   ├── agents/
│   │   ├── crew.py          # Multi-agent orchestration DAG
│   │   └── engine.py        # LLM provider (OpenRouter/Lyzr/Gemini)
│   ├── core/
│   │   ├── config.py        # Pydantic settings
│   │   ├── dag_state.py     # SQLite task state machine
│   │   └── exporter.py      # PDF / DOCX export
│   ├── memory/
│   │   └── qdrant_store.py  # Qdrant Cloud vector memory
│   ├── privacy/
│   │   └── anonymizer.py    # PII scrubbing and RTBF
│   └── templates/
│       └── dashboard.html   # Full-stack UI (single HTML file)
├── data/
│   └── demo_students.csv    # 200 synthetic students, 40 classes
├── tests/
│   └── test_pipeline.py     # Full pipeline test suite
├── .env.example             # Environment template
├── netlify.toml             # Netlify Functions deploy
├── render.yaml              # One-click Render deploy
├── Dockerfile               # Docker container
└── seed_data.py             # DB and Qdrant seeder
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11 |
| Multi-Agent | Lyzr Studio |
| Vector Memory | Qdrant Cloud |
| Free LLMs | OpenRouter (Llama 3.3 70B, Gemini 2.0 Flash) |
| Frontend | Vanilla JS + HTML (single file, no build step) |
| Export | ReportLab PDF + python-docx |
| Privacy | Custom PII anonymizer + Laplace DP |

---

## License

MIT 2026 Chalksy
