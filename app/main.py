import os
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import base64

from app.core.config import settings
from app.privacy.anonymizer import privacy_manager
from app.memory.qdrant_store import memory_manager
from app.core.dag_state import task_db
from app.agents.crew import crew_orchestrator
from app.agents.engine import AgentEngine
from app.core.exporter import export_manager
from app.evals.harness import eval_harness


app = FastAPI(
    title="Chalksy - Autonomous AI Classroom Builder",
    description="Voice-first, Qdrant memory-backed multi-agent orchestration for classroom teachers",
    version="1.0.0"
)

# Startup: Auto-load demo student list from CSV if available
@app.on_event("startup")
def startup_event():
    try:
        privacy_manager.load_students_from_csv("./data/demo_students.csv")
    except Exception as e:
        print(f"[STARTUP] Notice loading students CSV: {e}")

@app.get("/api/students")
async def get_students(class_id: Optional[str] = None):
    """Returns the list of students loaded from the demo CSV for dropdown selection."""
    if class_id and class_id.lower() in ["all", "any", "*", ""]:
        class_id = None
    students = privacy_manager.get_all_students(class_id=class_id)
    if not students:
        privacy_manager.load_students_from_csv("./data/demo_students.csv")
        students = privacy_manager.get_all_students(class_id=class_id)
    
    # Also fetch all distinct classes available
    all_students = privacy_manager.get_all_students(class_id=None)
    classes = sorted(list({s["class_id"] for s in all_students}))
    return {"status": "SUCCESS", "students": students, "available_classes": classes, "total": len(students)}



# Request schemas
class OmiWebhookPayload(BaseModel):
    transcript: str
    session_id: Optional[str] = "omi_session_01"
    teacher_id: Optional[str] = "TCHR_101"
    class_id: Optional[str] = "5B"
    is_final: Optional[bool] = True
    secret: Optional[str] = None

class GradeSubmissionPayload(BaseModel):
    student_name: str
    submission_text: str
    class_id: Optional[str] = "5B"

class ParentReportPayload(BaseModel):
    student_name: str
    language: Optional[str] = "English"
    class_id: Optional[str] = "5B"

class ExportPayload(BaseModel):
    task_id: str
    format: str # 'pdf' or 'docx' or 'whatsapp'

class ForgetStudentPayload(BaseModel):
    student_name_or_pid: str

def extract_custom_keys(request: Request) -> Dict[str, str]:
    raw = request.headers.get("x-user-keys")
    if not raw:
        return {}
    try:
        import urllib.parse
        import json as _json
        decoded = urllib.parse.unquote(raw)
        return _json.loads(decoded)
    except Exception:
        try:
            import json as _json
            return _json.loads(raw)
        except Exception:
            return {}

@app.get("/api/user-keys/status")
async def get_user_keys_status(request: Request):
    """Returns info about whether server has defaults and whether visitor has passed custom keys."""
    keys = extract_custom_keys(request)
    return {
        "has_custom_keys": bool(keys),
        "active_provider": "User Custom Keys" if keys else ("Lyzr Studio / Qdrant Cloud" if settings.LYZR_API_KEY else "Local Offline Engine"),
        "server_managed": bool(settings.LYZR_API_KEY or settings.OPENROUTER_API_KEY)
    }

class VerifyKeyPayload(BaseModel):
    openrouter_key: Optional[str] = None

@app.post("/api/user-keys/verify")
async def verify_user_keys(payload: VerifyKeyPayload):
    """Verifies if the visitor's key works."""
    if payload.openrouter_key:
        try:
            import httpx as _httpx
            headers = {"Authorization": f"Bearer {payload.openrouter_key}"}
            res = _httpx.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10.0)
            if res.status_code == 200:
                data = res.json().get("data", {})
                return {"status": "VALID", "provider": "OpenRouter", "label": data.get("label", "Active Key")}
            else:
                return {"status": "INVALID", "message": "OpenRouter authentication failed."}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    return {"status": "SKIPPED", "message": "Key stored locally in your browser."}

# Ingestion Gateway for Omi Wearable Webhook
@app.post("/api/omi/webhook")
async def omi_transcript_webhook(payload: OmiWebhookPayload, request: Request):
    """
    Real-time webhook receiving live transcript from Omi wearable device
    or browser/phone microphone fallback.
    """
    if not payload.transcript.strip():
        return {"status": "EMPTY_TRANSCRIPT", "message": "No voice transcript detected."}

    custom_keys = extract_custom_keys(request)
    pipeline_result = crew_orchestrator.run_voice_pipeline(
        raw_transcript=payload.transcript,
        teacher_id=payload.teacher_id,
        class_id=payload.class_id,
        custom_keys=custom_keys
    )
    return {"status": "SUCCESS", "data": pipeline_result}

# Official Omi Developer Webhook: Memory creation trigger (Page 6 of Guide)
@app.post("/omi/conversation")
async def on_omi_conversation(uid: str = "TCHR_101", payload: Dict[str, Any] = None):
    """Fires when an Omi conversation finishes."""
    payload = payload or {}
    segments = payload.get("transcript_segments", [])
    transcript = " ".join(s.get("text", "") for s in segments)
    overview = (payload.get("structured") or {}).get("overview", "")
    text = (overview + " " + transcript).strip()
    if not text:
        text = payload.get("text", "")
    
    if text:
        pipeline_res = crew_orchestrator.run_voice_pipeline(
            raw_transcript=text,
            teacher_id=uid,
            class_id="5B"
        )
        return {"status": "ok", "dag_id": pipeline_res.get("dag_id")}
    return {"status": "ok", "message": "empty transcript"}

# Official Omi Developer Webhook: Real-time transcript streaming (Page 6 of Guide)
@app.post("/omi/realtime")
async def on_omi_realtime(uid: str = "TCHR_101", session_id: str = "", payload: Any = None):
    """Fires with small batches in real-time while you speak."""
    if isinstance(payload, list):
        segments = payload
    elif isinstance(payload, dict):
        segments = payload.get("segments", [])
    else:
        segments = []
    
    text = " ".join(s.get("text", "") for s in segments if isinstance(s, dict)).strip()
    if text:
        memory_manager.add_memory(
            collection="teacher_notes",
            text=text,
            payload={"uid": uid, "source": "omi_realtime", "session_id": session_id},
            provenance="omi_realtime_stream"
        )
    return {"status": "ok"}

# Official /ask endpoint matching Hackathon Guide Step 5
@app.post("/ask")
async def ask_endpoint(body: Dict[str, Any]):
    """Standard grounded retrieval & reasoning endpoint from Guide."""
    uid = body.get("uid", "TCHR_101")
    question = body.get("question", "")
    
    # Retrieve from Qdrant
    hits = memory_manager.search_memories(
        collection="teacher_notes",
        query=question,
        role="teacher",
        limit=5
    )
    context = [h["text"] for h in hits]
    
    # Lyzr agent grounded answer
    ctx_str = "\n".join("- " + c for c in context) or "none"
    answer = AgentEngine.call_llm(
        system_prompt=f"You are the Teacher Assistant Agent. Answer based on the retrieved memories:\n{ctx_str}",
        user_prompt=question
    )
    return {"answer": answer, "context": context}

# Audio mic fallback endpoint for browser/phone
@app.post("/api/voice/browser-ingest")
async def browser_voice_ingest(payload: OmiWebhookPayload):
    return await omi_transcript_webhook(payload)

# Task DAG & Approval Queue Endpoints
@app.get("/api/dag/tasks")
async def get_dag_tasks():
    tasks = task_db.get_all_tasks(limit=50)
    return {"tasks": tasks}

@app.get("/api/dag/approvals")
async def get_pending_approvals():
    approvals = task_db.get_pending_approvals()
    return {"pending_approvals": approvals}

@app.post("/api/dag/approve/{task_id}")
async def approve_task(task_id: str):
    success = task_db.approve_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already approved.")
    return {"status": "SUCCESS", "message": f"Task {task_id} approved by teacher."}

@app.post("/api/dag/approve-all")
async def approve_all_tasks():
    pending = task_db.get_pending_approvals()
    approved = []
    for t in pending:
        task_db.approve_task(t["task_id"])
        approved.append(t["task_id"])
    return {"status": "SUCCESS", "approved_count": len(approved), "approved_tasks": approved}

class BulkApprovePayload(BaseModel):
    task_ids: List[str]

@app.post("/api/dag/approve-selected")
async def approve_selected_tasks(payload: BulkApprovePayload):
    approved = []
    for tid in payload.task_ids:
        if task_db.approve_task(tid):
            approved.append(tid)
    return {"status": "SUCCESS", "approved_count": len(approved), "approved_tasks": approved}

@app.post("/api/dag/reject/{task_id}")
async def reject_task(task_id: str):
    """Dismiss / reject a task from the approval queue (marks as REJECTED)."""
    import time as _time
    from app.core.dag_state import task_db as _task_db
    now = _time.time()
    import sqlite3 as _sqlite3
    with _sqlite3.connect(_task_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE task_dag SET status = 'REJECTED', updated_at = ? WHERE task_id = ?",
            (now, task_id)
        )
        conn.commit()
        changed = cursor.rowcount > 0
    if not changed:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "SUCCESS", "message": f"Task {task_id} rejected and removed from queue."}

@app.get("/api/dag/approved")
async def get_approved_tasks():
    import sqlite3 as _sqlite3
    with _sqlite3.connect(task_db.db_path) as conn:
        conn.row_factory = _sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM task_dag WHERE status = 'APPROVED' OR approved = 1 ORDER BY updated_at DESC LIMIT 30")
        rows = cursor.fetchall()
        return {"approved_tasks": [task_db._format_row(r) for r in rows]}

@app.get("/api/dag/rejected")
async def get_rejected_tasks():
    import sqlite3 as _sqlite3
    with _sqlite3.connect(task_db.db_path) as conn:
        conn.row_factory = _sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM task_dag WHERE status = 'REJECTED' ORDER BY updated_at DESC LIMIT 30")
        rows = cursor.fetchall()
        return {"rejected_tasks": [task_db._format_row(r) for r in rows]}

@app.post("/api/dag/restore/{task_id}")
async def restore_task_to_pending(task_id: str):
    """Restores an approved or rejected task back to AWAITING_APPROVAL."""
    import time as _time
    now = _time.time()
    import sqlite3 as _sqlite3
    with _sqlite3.connect(task_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE task_dag SET status = 'AWAITING_APPROVAL', approved = 0, updated_at = ? WHERE task_id = ?",
            (now, task_id)
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "SUCCESS", "message": f"Task {task_id} restored to pending approval queue."}

@app.post("/api/dag/seed-sample")
async def seed_sample_tasks(class_id: str = "5B"):
    import time as _time
    now = _time.time()
    dag_id = f"DAG_DEMO_{int(now)}"
    p_id = f"task_demo_plan_{int(now)}"
    ws_id = f"task_demo_ws_{int(now)}"
    
    plan_out = {
        "lesson_title": f"Fractions & Sensor Thresholds (Class {class_id})",
        "duration": "45 mins",
        "focus_areas": ["Denominator equivalence", "Sensor reading boundaries"],
        "bloom_level": "Analysis & Application"
    }
    ws_out = {
        "title": f"Differentiated Exit Ticket — Class {class_id}",
        "tier_1_foundational": [{"q": "Find common denominator for 1/3 + 1/6", "a": "6"}],
        "tier_2_proficient": [{"q": "Calculate robot threshold when distance < 12cm", "a": "Slow motor to 20%"}],
        "tier_3_advanced": [{"q": "Optimize closed-loop obstacle avoidance condition", "a": "PID feedback on ultrasonic echo"}],
        "rubric": {"accuracy": "50%", "logic": "50%"}
    }
    
    task_db.create_task(
        task_id=p_id, dag_id=dag_id, agent_name="Planner Agent",
        title=f"Sample Lesson Plan: Fractions & Sensors ({class_id})",
        parent_id=None, inputs={"class_id": class_id}
    )
    task_db.update_task_status(p_id, "AWAITING_APPROVAL", plan_out, "Critic verified: zero PII, CBSE/ICSE aligned.")
    
    task_db.create_task(
        task_id=ws_id, dag_id=dag_id, agent_name="Worksheet Agent",
        title=f"Sample 3-Tier Worksheet & Rubric ({class_id})",
        parent_id=p_id, inputs={"class_id": class_id}
    )
    task_db.update_task_status(ws_id, "AWAITING_APPROVAL", ws_out, "Critic verified: 3 tiers with solution key.")
    
    return {
        "status": "SUCCESS",
        "tasks_seeded": [p_id, ws_id],
        "message": f"Seeded 2 sample tasks for Class {class_id} into the approval queue."
    }


# Grader Agent Endpoint
@app.post("/api/agents/grade")
async def grade_student_work(payload: GradeSubmissionPayload):
    result = crew_orchestrator.grade_submission(
        student_name_or_pid=payload.student_name,
        student_answer=payload.submission_text,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "grade_report": result}

# Parent Report Agent Endpoint
@app.post("/api/agents/parent-report")
async def create_parent_report(payload: ParentReportPayload):
    report = crew_orchestrator.generate_parent_report(
        student_name_or_pid=payload.student_name,
        language=payload.language,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "report": report}

class DirectNotePayload(BaseModel):
    text: str
    class_id: str = "5B"
    teacher_id: str = "TCHR_101"

@app.post("/api/memory/note")
async def save_direct_note(payload: DirectNotePayload):
    sanitized_text, redacts = privacy_manager.sanitize_text(payload.text, class_id=payload.class_id)
    note_id = memory_manager.add_memory(
        collection="teacher_notes",
        text=sanitized_text,
        payload={
            "teacher_id": payload.teacher_id,
            "class_id": payload.class_id,
            "source": "teacher_direct_save",
            "visibility": "class",
            "sanitized": True,
            "pii_redacted": redacts
        },
        provenance="teacher_direct_input"
    )
    return {
        "status": "SUCCESS",
        "note_id": note_id,
        "class_id": payload.class_id,
        "sanitized_text": sanitized_text,
        "message": f"Saved directly to teacher_notes for Class {payload.class_id}."
    }

# Memory stats and search
@app.get("/api/memory/stats")
async def get_memory_stats():
    stats = memory_manager.get_collection_stats()
    return {"collections": stats}

@app.get("/api/memory/search")
async def search_memory(collection: str, query: Optional[str] = "", class_id: Optional[str] = None):
    effective_class = None if (not class_id or class_id.lower() in ["all", "any", "none", "*"]) else class_id
    effective_query = (query or "").strip()
    res = memory_manager.search_memories(collection=collection, query=effective_query, class_id=effective_class, limit=10)
    return {"results": res}


# Phase 1: Feature 1 - Memory Consolidation Job
@app.post("/api/memory/consolidate")
async def consolidate_memory_endpoint(class_id: str = "5B"):
    """Clusters raw notes & student events into the insights collection and down-weights raw items."""
    result = memory_manager.consolidate_memories(class_id=class_id)
    return {"status": "SUCCESS", "consolidation": result}

# Phase 1: Feature 3 - Misconception Knowledge Graph Root Gap
@app.get("/api/memory/knowledge-graph/trace")
async def trace_knowledge_graph(concept: str = "infinite_loops"):
    trace = memory_manager.trace_root_gap(concept)
    return {"status": "SUCCESS", "trace": trace}

# Phase 1: Feature 4 - Spaced Repetition Due Review
@app.get("/api/memory/spaced-repetition/due")
async def get_spaced_repetition_due(class_id: str = "5B"):
    due_items = memory_manager.get_due_spaced_repetition_items(class_id=class_id)
    return {"status": "SUCCESS", "class_id": class_id, "due_items": due_items}

# Phase 1: Feature 5 - Hybrid vs Dense Retrieval Benchmark
@app.get("/api/memory/benchmark/retrieval")
async def benchmark_retrieval():
    benchmark = memory_manager.benchmark_hybrid_vs_dense()
    return {"status": "SUCCESS", "benchmark": benchmark}

# Phase 2: Feature 6 - Teacher-Correction Learning Endpoint
class TeacherCorrectionPayload(BaseModel):
    item_type: str = "grade"
    original_output: Dict[str, Any]
    teacher_correction: Dict[str, Any]
    class_id: Optional[str] = "5B"

@app.post("/api/corrections/record")
async def record_correction(payload: TeacherCorrectionPayload):
    point_id = crew_orchestrator.record_teacher_correction(
        item_type=payload.item_type,
        original_output=payload.original_output,
        teacher_correction=payload.teacher_correction,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "correction_memory_id": point_id, "message": "Teacher correction saved to memory as few-shot example."}

# Phase 2: Feature 8 - Ablation Study Endpoint (Memory ON vs OFF)
@app.get("/api/evals/ablation")
async def get_ablation_study():
    ablation = eval_harness.evaluate_ablation_memory_on_vs_off()
    return {"status": "SUCCESS", "ablation": ablation}

# Phase 3: Feature 12 - PTM Conference Checklist Endpoint
class PTMPayload(BaseModel):
    transcript: str
    student_name: str
    parent_name: Optional[str] = "Parent"
    class_id: Optional[str] = "5B"

@app.post("/api/agents/ptm-conference")
async def process_ptm_endpoint(payload: PTMPayload):
    ptm_plan = crew_orchestrator.process_ptm_meeting(
        raw_transcript=payload.transcript,
        student_name_or_pid=payload.student_name,
        parent_name=payload.parent_name,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "ptm_record": ptm_plan}

# Phase 3: Feature 13 - End-of-Day 60-Second Reflection Endpoint
@app.get("/api/agents/reflection/eod")
async def get_eod_reflection(class_id: str = "5B"):
    digest = crew_orchestrator.generate_end_of_day_reflection(class_id=class_id)
    return {"status": "SUCCESS", "reflection": digest}

# Phase 4: Feature 15 - Differentiation Agent Endpoint
class DiffPayload(BaseModel):
    content: Dict[str, Any]
    class_id: Optional[str] = "5B"

@app.post("/api/agents/differentiate")
async def differentiate_endpoint(payload: DiffPayload):
    diff = crew_orchestrator.differentiate_lesson_content(payload.content, class_id=payload.class_id)
    return {"status": "SUCCESS", "differentiated_pack": diff}

# Phase 4: Feature 16 - Assessment Designer Agent Endpoint
@app.get("/api/agents/assessment/blueprint")
async def assessment_blueprint_endpoint(topic: str = "Robotics & Loops", class_id: str = "5B"):
    assessment = crew_orchestrator.design_assessment(topic=topic, class_id=class_id)
    return {"status": "SUCCESS", "assessment": assessment}

# Phase 4: Feature 17 - Intervention Agent Endpoint
@app.get("/api/agents/intervention/plan")
async def intervention_plan_endpoint(class_id: str = "5B"):
    plan = crew_orchestrator.detect_and_plan_interventions(class_id=class_id)
    return {"status": "SUCCESS", "intervention_plan": plan}

# Phase 4: Feature 18 - Substitute Handover Pack Endpoint
@app.get("/api/agents/substitute/handover")
async def substitute_handover_endpoint(class_id: str = "5B"):
    pack = crew_orchestrator.generate_substitute_pack(class_id=class_id)
    return {"status": "SUCCESS", "substitute_pack": pack}

# Phase 4: Feature 19 - Quiz/Game Bingo & Kahoot Endpoint
@app.get("/api/agents/quiz-game/generate")
async def quiz_game_endpoint(topic: str = "Loops & Sensors", class_id: str = "5B"):
    game = crew_orchestrator.generate_quiz_game(topic=topic, class_id=class_id)
    return {"status": "SUCCESS", "quiz_game": game}

# Exporter
@app.post("/api/export")
async def export_artifact(payload: ExportPayload):
    tasks = task_db.get_all_tasks(limit=100)
    target_task = next((t for t in tasks if t["task_id"] == payload.task_id), None)
    if not target_task:
        raise HTTPException(status_code=404, detail="Task not found.")

    output_data = target_task.get("output", {})
    title = target_task.get("title", "Classroom Resource")
    filename = f"{payload.task_id}.{payload.format}"
    filepath = os.path.join("exports", filename)

    if payload.format == "pdf":
        export_manager.export_to_pdf(title, output_data, filepath)
        return FileResponse(filepath, media_type="application/pdf", filename=filename)
    elif payload.format == "docx":
        export_manager.export_to_docx(title, output_data, filepath)
        return FileResponse(filepath, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=filename)
    elif payload.format == "whatsapp":
        msg = export_manager.format_whatsapp_message("Parent", "Student", output_data)
        return {"whatsapp_text": msg}
    else:
        raise HTTPException(status_code=400, detail="Unsupported format.")


# Statistical Evaluation Endpoint

@app.get("/api/evals/run")
async def run_evaluations():
    results = eval_harness.run_full_suite()
    return {"status": "SUCCESS", "metrics": results}


# Phase 5: Feature 21 - Handwritten Photo OCR Grading Endpoint
class PhotoGradePayload(BaseModel):
    image_base64: str  # base64-encoded image of handwritten work
    student_name: str
    class_id: Optional[str] = "5B"

@app.post("/api/agents/grade/photo")
async def grade_photo_submission(payload: PhotoGradePayload, request: Request):
    """Accepts base64 image of handwritten student work and grades via cognitive engine."""
    custom_keys = extract_custom_keys(request)
    result = crew_orchestrator.grade_from_photo(
        image_base64=payload.image_base64,
        student_name_or_pid=payload.student_name,
        class_id=payload.class_id,
        custom_keys=custom_keys
    )
    return {"status": "SUCCESS", "grade_report": result}

@app.post("/api/agents/grade/photo/upload")
async def grade_photo_upload(request: Request, student_name: str = "Student", class_id: str = "5B", file: UploadFile = File(...)):
    """Accepts multipart photo upload of handwritten work."""
    custom_keys = extract_custom_keys(request)
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    result = crew_orchestrator.grade_from_photo(
        image_base64=img_b64,
        student_name_or_pid=student_name,
        class_id=class_id,
        custom_keys=custom_keys
    )
    return {"status": "SUCCESS", "filename": file.filename, "grade_report": result}


# Phase 5: Feature 22 - Curriculum Coverage Tracker
@app.get("/api/curriculum/coverage")
async def get_curriculum_coverage(class_id: str = "5B", board: str = "CBSE"):
    """Returns CBSE/ICSE/Karnataka curriculum coverage with taught vs pending topics."""
    coverage = crew_orchestrator.get_curriculum_coverage(class_id=class_id, board=board)
    return {"status": "SUCCESS", "curriculum_coverage": coverage}


# Phase 5: Feature 23 - WhatsApp Business API Parent Report
class WhatsAppReportPayload(BaseModel):
    student_name: str
    language: Optional[str] = "English"
    class_id: Optional[str] = "5B"

@app.post("/api/agents/parent-report/whatsapp")
async def whatsapp_parent_report(payload: WhatsAppReportPayload):
    """Generates WhatsApp Business API-ready parent report message."""
    result = crew_orchestrator.format_whatsapp_report(
        student_name_or_pid=payload.student_name,
        language=payload.language,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "whatsapp_report": result}


# Phase 5: Feature 24 - Voice TTS Parent Report (SSML)
@app.post("/api/agents/parent-report/tts")
async def tts_parent_report(payload: WhatsAppReportPayload):
    """Generates SSML-marked-up voice parent report for Google TTS / AWS Polly / Azure."""
    result = crew_orchestrator.generate_tts_parent_report(
        student_name_or_pid=payload.student_name,
        language=payload.language,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "tts_report": result}


# Phase 5: Feature 25 - Teacher Analytics Heatmap Data
@app.get("/api/analytics/teacher")
async def get_teacher_analytics(class_id: str = "5B"):
    """Returns misconception heatmap, per-student score trends, and agent performance metrics."""
    analytics = crew_orchestrator.get_teacher_analytics(class_id=class_id)
    return {"status": "SUCCESS", "analytics": analytics}


# ── Phase 6: Privacy Hardening ──────────────────────────────────────────────

# Phase 6: Consent Registry
class ConsentPayload(BaseModel):
    student_name_or_pid: str
    collection: str
    consented: bool
    granted_by: Optional[str] = "parent"
    expires_in_days: Optional[int] = None

@app.post("/api/privacy/consent")
async def set_consent(payload: ConsentPayload):
    """Set or revoke per-student, per-collection data consent."""
    pid = payload.student_name_or_pid
    if not pid.startswith("STU_"):
        pid = privacy_manager.generate_pseudonym(pid, "5B")
    ok = privacy_manager.set_consent(
        student_pid=pid,
        collection=payload.collection,
        consented=payload.consented,
        granted_by=payload.granted_by,
        expires_in_days=payload.expires_in_days
    )
    return {"status": "SUCCESS", "student_pid": pid, "consent_updated": ok}

@app.get("/api/privacy/consent/{student_name_or_pid}")
async def get_consent_report(student_name_or_pid: str):
    """Returns full consent report for a student."""
    pid = student_name_or_pid
    if not pid.startswith("STU_"):
        pid = privacy_manager.generate_pseudonym(pid, "5B")
    return {"status": "SUCCESS", "consent_report": privacy_manager.get_consent_report(pid)}


# Phase 6: Hash-Chained Audit Log
@app.get("/api/privacy/audit-log")
async def get_audit_log(limit: int = 50, actor: Optional[str] = None, event_type: Optional[str] = None):
    """Returns recent hash-chained audit log entries."""
    entries = privacy_manager.get_audit_log(limit=limit, actor=actor, event_type=event_type)
    return {"status": "SUCCESS", "audit_entries": entries, "count": len(entries)}

@app.get("/api/privacy/audit-log/verify")
async def verify_audit_chain():
    """Verifies the tamper-evident hash chain integrity of the audit log."""
    result = privacy_manager.verify_audit_chain()
    return {"status": "SUCCESS", "chain_integrity": result}


# Phase 6: Differential Privacy Aggregate
class DPPayload(BaseModel):
    scores: List[float]
    epsilon: Optional[float] = 1.0

@app.post("/api/privacy/dp-aggregate")
async def dp_aggregate(payload: DPPayload):
    """Applies Laplace differential-privacy noise to class score aggregates."""
    result = privacy_manager.privatize_score_aggregate(payload.scores, epsilon=payload.epsilon)
    return {"status": "SUCCESS", "dp_stats": result}


# Phase 6: Extended Right-to-be-Forgotten
class RTBFPayload(BaseModel):
    student_name_or_pid: str
    requested_by: Optional[str] = "parent"

@app.post("/api/privacy/rtbf")
async def extended_rtbf(payload: RTBFPayload):
    """Extended Right-to-be-Forgotten: cascade purge from registry, consent, fingerprints + vector memory."""
    pid = payload.student_name_or_pid
    if not pid.startswith("STU_"):
        pid = privacy_manager.generate_pseudonym(pid, "5B")
    # SQLite cascade
    cert = privacy_manager.extended_rtbf(pid, requested_by=payload.requested_by)
    # Qdrant vector purge
    vec_purged = memory_manager.purge_by_filter("student_events", pid)
    cert["vector_memories_purged"] = vec_purged
    return {"status": "SUCCESS", "purge_certificate": cert}


# Phase 6: Plagiarism Advisory
class PlagiarismPayload(BaseModel):
    student_name_or_pid: str
    submission_text: str
    class_id: Optional[str] = "5B"
    threshold: Optional[float] = 0.75

@app.post("/api/privacy/plagiarism-check")
async def plagiarism_check(payload: PlagiarismPayload):
    """Checks submission similarity against stored class fingerprints. Advisory only."""
    pid = payload.student_name_or_pid
    if not pid.startswith("STU_"):
        pid = privacy_manager.register_or_get_student(pid, payload.class_id or "5B")
    result = privacy_manager.check_plagiarism(
        student_pid=pid,
        submission_text=payload.submission_text,
        class_id=payload.class_id,
        threshold=payload.threshold
    )
    return {"status": "SUCCESS", "plagiarism_advisory": result}


# Original Privacy Purge (Right to be Forgotten — single-endpoint legacy)
@app.post("/api/privacy/forget")
async def forget_student(payload: "ForgetStudentPayload"):
    pid = payload.student_name_or_pid
    if not pid.startswith("STU_"):
        pid = privacy_manager.generate_pseudonym(payload.student_name_or_pid, "5B")
    reg_deleted = privacy_manager.delete_student(pid)
    vec_deleted = memory_manager.purge_by_filter("student_events", pid)
    return {
        "status": "PURGED",
        "student_pid": pid,
        "registry_cleared": reg_deleted,
        "vector_memories_purged": vec_deleted
    }


# ── Phase 7: Demo Polish ──────────────────────────────────────────────────────

# Phase 7: Prepare My Monday chained command
@app.post("/api/agents/prepare-monday")
async def prepare_monday(class_id: str = "5B", background_tasks: BackgroundTasks = None):
    """
    'Prepare My Monday' — one voice trigger that chains 6 autonomous agents:
    EOD Reflection → Spaced Rep → Consolidation → Lesson Plan → Worksheet → Critic
    """
    pack = crew_orchestrator.prepare_my_monday(class_id=class_id)
    return {"status": "SUCCESS", "monday_pack": pack}


# Phase 7: Memory Timeline View
@app.get("/api/memory/timeline")
async def get_memory_timeline(class_id: str = "5B", limit: int = 20):
    """Returns chronological timeline of all memories with color-coded collection badges."""
    timeline = crew_orchestrator.get_memory_timeline(class_id=class_id, limit=limit)
    return {"status": "SUCCESS", "memory_timeline": timeline}


# Phase 7: PII Scrub Diff Preview
class PIIDiffPayload(BaseModel):
    raw_text: str
    class_id: Optional[str] = "5B"

@app.post("/api/privacy/pii-diff")
async def pii_diff_preview(payload: PIIDiffPayload):
    """Returns side-by-side token-level diff showing exactly what PII was scrubbed."""
    diff = crew_orchestrator.pii_diff_preview(
        raw_text=payload.raw_text,
        class_id=payload.class_id
    )
    return {"status": "SUCCESS", "pii_diff": diff}


# Phase 7: Memory On/Off Toggle (for ablation demo)
class MemoryTogglePayload(BaseModel):
    memory_enabled: bool

@app.post("/api/settings/memory-toggle")
async def toggle_memory(payload: MemoryTogglePayload):
    """Toggle Qdrant memory on or off for live ablation demonstration."""
    settings.OFFLINE_MODE = not payload.memory_enabled
    return {
        "status": "SUCCESS",
        "memory_enabled": payload.memory_enabled,
        "offline_mode": settings.OFFLINE_MODE,
        "message": f"Memory {'ENABLED' if payload.memory_enabled else 'DISABLED (offline mode)'}. Ablation study toggle active."
    }


# Observability Dashboard HTML Route
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("app/templates/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
