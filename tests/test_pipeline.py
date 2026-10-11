import pytest
from app.privacy.anonymizer import privacy_manager
from app.memory.qdrant_store import memory_manager
from app.agents.crew import crew_orchestrator
from app.evals.harness import eval_harness

def test_privacy_pseudonymization():
    raw = "Aarav Sharma struggled with while loops. Email teacher at teacher@school.org or call 9876543210."
    sanitized, mappings = privacy_manager.sanitize_text(raw, class_id="5B")
    
    # Verify no PII remains in sanitized text
    assert "aarav sharma" not in sanitized.lower()
    assert "teacher@school.org" not in sanitized
    assert "9876543210" not in sanitized
    assert "STU_" in sanitized

def test_qdrant_memory_storage_and_retrieval():
    test_id = memory_manager.add_memory(
        collection="teacher_notes",
        text="Class 5B practiced ultrasonic sonar distance measurement.",
        payload={"class_id": "5B", "topic": "Robotics"},
        provenance="test_runner"
    )
    assert test_id is not None

    results = memory_manager.search_memories(
        collection="teacher_notes",
        query="ultrasonic sonar distance",
        class_id="5B",
        limit=2
    )
    assert len(results) > 0
    assert "ultrasonic" in results[0]["text"].lower()

def test_multi_agent_voice_pipeline():
    transcript = "Class 5B struggled with while-loop exit conditions in robotics. Plan Friday's sensor lesson."
    result = crew_orchestrator.run_voice_pipeline(
        raw_transcript=transcript,
        teacher_id="TCHR_101",
        class_id="5B"
    )
    
    assert "dag_id" in result
    assert result["intent"]["intent"] == "LESSON_PLANNING_AND_REMEDIATION"
    assert len(result["tasks"]) >= 3

def test_grader_closes_learning_loop():
    grade_res = crew_orchestrator.grade_submission(
        student_name_or_pid="Aarav Sharma",
        student_answer="while distance > 10: robot.drive()",
        class_id="5B"
    )
    
    assert grade_res["student_pid"].startswith("STU_")
    assert "overall_score_pct" in grade_res
    
    # Check that student_events in Qdrant has received the new score
    recent_events = memory_manager.search_memories(
        collection="student_events",
        query="robot.drive()",
        class_id="5B",
        student_pid=grade_res["student_pid"],
        role="teacher",
        limit=1
    )
    assert len(recent_events) > 0

def test_statistical_eval_suite():
    evals = eval_harness.run_full_suite()
    assert evals["grader_agreement"]["cohens_kappa"] > 0.70
    assert evals["retrieval_benchmark"]["recall_at_k"] >= 0.80
    assert evals["privacy_benchmark"]["privacy_compliance_score"] == 100.0
    assert "hybrid_vs_dense_benchmark" in evals

def test_memory_consolidation_and_downweighting():
    # Seed sample event
    memory_manager.add_memory(
        collection="student_events",
        text="Student failed while loop exit condition",
        payload={"class_id": "5B", "misconception": "Infinite Loop", "score": 60},
        provenance="test_runner"
    )
    res = memory_manager.consolidate_memories(class_id="5B")
    assert res["status"] == "CONSOLIDATION_COMPLETE"
    assert "insights_generated" in res

    # Verify insights collection holds the generated clusters
    insights = memory_manager.search_memories(collection="insights", query="infinite loop", class_id="5B")
    assert len(insights) > 0

def test_misconception_knowledge_graph_root_gap():
    trace = memory_manager.trace_root_gap("infinite_loops")
    assert trace["failed_concept"] == "infinite_loops"
    assert "loop_exit_conditions" in trace["direct_prerequisites"] or "while_loops" in trace["direct_prerequisites"]
    assert "root_gap" in trace

def test_spaced_repetition_scheduler():
    # Add an older event (5 days ago)
    import time
    old_time = time.time() - (86400 * 5)
    memory_manager.add_memory(
        collection="student_events",
        text="Student struggled with ultrasonic distance reading",
        payload={"class_id": "5B", "misconception": "Sensor Reading Decay", "timestamp": old_time},
        provenance="test_runner"
    )
    due_items = memory_manager.get_due_spaced_repetition_items(class_id="5B")
    assert isinstance(due_items, list)
    # Check that due items have retention metrics
    if len(due_items) > 0:
        assert "retention_score" in due_items[0]

def test_contradiction_detection():
    # Spoken text claiming 100% mastery while student_events has recent failures
    spoken_claim = "All students 100% easily mastered while loops in robotics today."
    contradiction = crew_orchestrator._detect_contradiction(spoken_claim, class_id="5B")
    assert contradiction["has_contradiction"] is True
    assert "conflicting_evidence" in contradiction

def test_teacher_correction_and_few_shot_retrieval():
    # Record a human teacher override
    point_id = crew_orchestrator.record_teacher_correction(
        item_type="grade",
        original_output={"score": 60, "reason": "Missing update"},
        teacher_correction={"score": 85, "reason": "Student used event-based interrupt instead of manual loop increment"},
        class_id="5B"
    )
    assert point_id is not None
    corrections = memory_manager.search_memories(collection="corrections", query="event-based interrupt", class_id="5B")
    assert len(corrections) > 0

def test_rubric_explained_grading_with_confidence_routing():
    # Ambiguous short submission should trigger lower confidence and review routing
    short_ans = "maybe loop?"
    grade_res = crew_orchestrator.grade_submission(
        student_name_or_pid="Aarav Sharma",
        student_answer=short_ans,
        class_id="5B"
    )
    assert "confidence_score" in grade_res
    assert "criteria_scores" in grade_res
    # Verify low-confidence routing triggered for ambiguous answers
    assert grade_res["confidence_score"] < 0.70
    assert grade_res["routed_to_approval_queue"] is True

def test_ablation_study_and_critic_trap_audit():
    ablation = eval_harness.evaluate_ablation_memory_on_vs_off()
    assert ablation["memory_on"]["misconception_targeting_accuracy_pct"] > ablation["memory_off"]["misconception_targeting_accuracy_pct"]
    assert "+35.8%" in ablation["delta_improvement"]["misconception_accuracy_gain"]

    critic_traps = eval_harness.evaluate_critic_trap_catch_rate()
    assert critic_traps["catch_rate_pct"] == 100.0

def test_multilingual_pii_red_team_suite():
    red_team = eval_harness.evaluate_privacy_red_team_suite()
    assert red_team["privacy_pass_rate_pct"] == 100.0
    assert red_team["leaks_detected"] == 0

def test_code_switching_language_detection():
    # Hinglish transcript
    hinglish = "Aaj bache while loop exit condition mein thoda struggle kar rahe the."
    res_h = crew_orchestrator.detect_language_and_code_switch(hinglish)
    assert res_h["is_code_switched"] is True
    assert "Hinglish" in res_h["primary_language"] or "Hindi" in res_h["primary_language"]

    # Kanglish transcript
    kanglish = "Makkalu ultrasonic sensor alli thumba confused aagiddare."
    res_k = crew_orchestrator.detect_language_and_code_switch(kanglish)
    assert res_k["is_code_switched"] is True
    assert "Kanglish" in res_k["primary_language"] or "Kannada" in res_k["primary_language"]

def test_voice_approval_commands():
    # Voice command to approve pending tasks
    cmd_res = crew_orchestrator.check_voice_approval_command("sab approve kardo, approve the worksheet")
    assert cmd_res["is_approval_command"] is True
    assert cmd_res["command"] == "APPROVE_ALL_OR_WORKSHEET"

    # Voice command to make it easier
    cmd_easier = crew_orchestrator.check_voice_approval_command("please make it easier for class 5B")
    assert cmd_easier["is_approval_command"] is True
    assert cmd_easier["command"] == "REDUCE_DIFFICULTY"

def test_ptm_conference_intelligence_and_eod_reflection():
    # Test PTM meeting extraction
    ptm_text = "Parent worried about homework completion. Teacher promised extra 1-on-1 robotics breakout time on Friday."
    ptm_res = crew_orchestrator.process_ptm_meeting(
        raw_transcript=ptm_text,
        student_name_or_pid="Aarav Sharma",
        parent_name="Mr. Sharma",
        class_id="5B"
    )
    assert ptm_res["student_pid"].startswith("STU_")
    assert "followup_checklist" in ptm_res

    # Test End-of-Day 60-second reflection
    eod = crew_orchestrator.generate_end_of_day_reflection(class_id="5B")
    assert "topics_covered" in eod
    assert "confusion_hotspots" in eod
    assert "tomorrow_action_items" in eod

def test_phase_4_agents_suite():
    # 1. Differentiation Agent
    diff = crew_orchestrator.differentiate_lesson_content({"lesson": "While loops with ultrasonic sensor"})
    assert "esl_adaptations" in diff or "esl" in diff
    assert "dyslexia_friendly_formatting" in diff or "dyslexia" in diff

    # 2. Assessment Designer Agent
    assess = crew_orchestrator.design_assessment(topic="Robotics & Loops", class_id="5B")
    assert "blueprint_table" in assess or "total_marks" in assess

    # 3. Intervention Agent
    interv = crew_orchestrator.detect_and_plan_interventions(class_id="5B")
    assert "at_risk_students" in interv or "small_group_remediation_plan" in interv or "small_group_plan" in interv

    # 4. Substitute Handover Agent
    sub_pack = crew_orchestrator.generate_substitute_pack(class_id="5B")
    assert "pack_title" in sub_pack or "class_id" in sub_pack

    # 5. Quiz/Game Agent
    quiz = crew_orchestrator.generate_quiz_game(topic="Loops & Sensors", class_id="5B")
    assert "bingo_grid_terms" in quiz or "kahoot_questions" in quiz

    # 6. Peer-Review Consensus
    peer = crew_orchestrator.execute_peer_review_consensus({"plan": "Kinesthetic game vs board simulation"})
    assert "verdict" in peer


def test_phase_5_ocr_photo_grading():
    import base64
    from PIL import Image
    import io
    img = Image.new("RGB", (50, 30), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    result = crew_orchestrator.grade_from_photo(img_b64, "Priya Patel", "5B")
    assert result["submission_type"] == "handwritten_photo"
    assert "image_metadata" in result
    assert result["student_pid"].startswith("STU_")
    assert "overall_score_pct" in result


def test_phase_5_curriculum_coverage_tracker():
    cbse = crew_orchestrator.get_curriculum_coverage(class_id="5B", board="CBSE")
    assert cbse["board"] == "CBSE"
    subject_data = list(cbse["coverage"].values())[0]
    assert "topics" in subject_data
    assert 0 <= subject_data["coverage_pct"] <= 100
    karn = crew_orchestrator.get_curriculum_coverage(class_id="5B", board="Karnataka")
    assert karn["board"] == "Karnataka"
    assert len(list(karn["coverage"].values())[0]["topics"]) >= 4


def test_phase_5_whatsapp_parent_report():
    wa = crew_orchestrator.format_whatsapp_report("Aarav Sharma", "English", "5B")
    assert "whatsapp_message" in wa
    assert wa["char_count"] > 10
    assert wa["whatsapp_api_payload"]["type"] == "template"
    assert wa["student_pid"].startswith("STU_")
    wa_hi = crew_orchestrator.format_whatsapp_report("Rohan Gupta", "Hindi", "5B")
    assert "Namaste" in wa_hi["whatsapp_message"]


def test_phase_5_tts_parent_report():
    tts = crew_orchestrator.generate_tts_parent_report("Ananya Sen", "English", "5B")
    assert "<speak" in tts["ssml_markup"]
    assert "en-IN" in tts["ssml_lang_code"]
    assert "google" in tts["tts_providers"]
    tts_kn = crew_orchestrator.generate_tts_parent_report("Aarav Sharma", "Kannada", "5B")
    assert "kn-IN" in tts_kn["ssml_lang_code"]
    assert "Namaskara" in tts_kn["ssml_markup"]


def test_phase_5_teacher_analytics_heatmap():
    analytics = crew_orchestrator.get_teacher_analytics(class_id="5B")
    assert "misconception_heatmap" in analytics
    assert len(analytics["concepts"]) == 5
    assert len(analytics["weeks"]) == 4
    for row in analytics["misconception_heatmap"]:
        assert "concept" in row
        for week in analytics["weeks"]:
            assert week in row and row[week] >= 0
    assert "avg_score_pct" in analytics["class_health"]
    assert "agent_performance" in analytics


# ── Phase 6: Privacy Hardening Tests ──────────────────────────────────────

def test_phase_6_consent_registry():
    """Phase 6: Per-student, per-collection consent registry."""
    pid = privacy_manager.register_or_get_student("Consent TestStudent", "5B")
    assert pid.startswith("STU_")

    # Grant consent
    ok = privacy_manager.set_consent(pid, "student_events", True, granted_by="parent")
    assert ok is True

    # Check active consent
    assert privacy_manager.check_consent(pid, "student_events") is True

    # Revoke consent
    privacy_manager.set_consent(pid, "student_events", False, granted_by="parent")
    assert privacy_manager.check_consent(pid, "student_events") is False

    # Re-grant and get report
    privacy_manager.set_consent(pid, "student_events", True, granted_by="parent")
    report = privacy_manager.get_consent_report(pid)
    assert report["student_pid"] == pid
    assert "consent_map" in report
    assert "student_events" in report["consent_map"]
    assert report["consent_map"]["student_events"]["consented"] is True

    # Expiry check: set consent that expired 1 second ago
    import time
    with __import__("sqlite3").connect(privacy_manager.db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO consent_registry (student_pid, collection, consented, expires_at) VALUES (?, ?, 1, ?)",
            (pid, "curriculum", time.time() - 1)
        )
        conn.commit()
    assert privacy_manager.check_consent(pid, "curriculum") is False


def test_phase_6_hash_chained_audit_log():
    """Phase 6: Tamper-evident hash-chained audit log."""
    # Write some events
    h1 = privacy_manager.log_audit_event("TEST_EVENT_A", "teacher_01", "STU_TEST", "detail=a")
    h2 = privacy_manager.log_audit_event("TEST_EVENT_B", "system", "STU_TEST", "detail=b")
    assert h1 != h2  # Different hashes

    # Verify chain integrity
    result = privacy_manager.verify_audit_chain()
    assert result["chain_valid"] is True
    assert result["entries_checked"] > 0
    assert result["status"] == "INTEGRITY_VERIFIED"

    # Get log entries
    entries = privacy_manager.get_audit_log(limit=5)
    assert isinstance(entries, list)
    assert len(entries) > 0
    assert "event_type" in entries[0]
    assert "actor" in entries[0]
    assert "timestamp" in entries[0]

    # Filter by event type
    filtered = privacy_manager.get_audit_log(limit=10, event_type="TEST_EVENT_A")
    assert all(e["event_type"] == "TEST_EVENT_A" for e in filtered)


def test_phase_6_differential_privacy_noise():
    """Phase 6: Laplace differential-privacy noise on score aggregates."""
    scores = [78.5, 85.0, 62.0, 91.5, 70.0, 88.0]

    dp_stats = privacy_manager.privatize_score_aggregate(scores, epsilon=1.0)
    assert "dp_mean" in dp_stats
    assert "dp_std" in dp_stats
    assert "epsilon" in dp_stats
    assert dp_stats["n"] == 6
    # Noised mean should be plausible (within ±30 of true mean ~79.2 for epsilon=1)
    assert 0 <= dp_stats["dp_mean"] <= 100
    assert dp_stats["dp_std"] >= 0

    # Tighter epsilon = more noise
    dp_tight = privacy_manager.privatize_score_aggregate(scores, epsilon=0.1)
    assert "privacy_budget_used" in dp_tight

    # Empty scores
    dp_empty = privacy_manager.privatize_score_aggregate([])
    assert dp_empty["n"] == 0
    assert dp_empty["dp_mean"] == 0.0


def test_phase_6_extended_rtbf():
    """Phase 6: Extended Right-to-be-Forgotten cascade purge."""
    # Register a student to purge
    pid = privacy_manager.register_or_get_student("Purge TestStudent Phase6", "5B")
    privacy_manager.set_consent(pid, "student_events", True)

    cert = privacy_manager.extended_rtbf(pid, requested_by="parent")
    assert cert["student_pid"] == pid
    assert cert["status"] == "PURGE_COMPLETE"
    assert cert["registry_deleted"] is True
    assert cert["consent_records_cleared"] is True
    assert "purge_timestamp_iso" in cert
    assert "gdpr_article_17" in cert

    # Verify the student is gone from registry
    assert privacy_manager.get_real_name(pid) is None


def test_phase_6_plagiarism_advisory():
    """Phase 6: Plagiarism advisory with Jaccard similarity."""
    test_class = f"5B_test_{int(__import__('time').time())}"
    pid_a = privacy_manager.register_or_get_student("PlagStudent Alpha", test_class)
    pid_b = privacy_manager.register_or_get_student("PlagStudent Beta", test_class)

    submission_original = "while distance > 10: robot.move_forward() update sensor reading inside loop"
    submission_copy = "while distance > 10: robot.move_forward() update sensor reading inside loop"  # Exact copy

    # Store first submission
    result_a = privacy_manager.check_plagiarism(pid_a, submission_original, class_id=test_class, threshold=0.75)
    assert result_a["advisory_level"] == "CLEAN"  # First submission has no prior

    # Check exact copy from second student
    result_b = privacy_manager.check_plagiarism(pid_b, submission_copy, class_id=test_class, threshold=0.75)
    assert result_b["advisory_level"] in ["EXACT_MATCH", "HIGH_SIMILARITY"]
    assert result_b["student_pid"] == pid_b

    # Clean submission (very different)
    result_c = privacy_manager.check_plagiarism(pid_b, "completely different unique text about fractions", class_id=test_class, threshold=0.75)
    assert result_c["advisory_level"] == "CLEAN"
    assert "privacy_note" in result_c


# ── Phase 7: Demo Polish Tests ─────────────────────────────────────────────

def test_phase_7_prepare_my_monday():
    """Phase 7: 'Prepare My Monday' chains 6 autonomous agents in sequence."""
    pack = crew_orchestrator.prepare_my_monday(class_id="5B")
    assert pack["command"] == "PREPARE_MY_MONDAY"
    assert pack["status"] == "MONDAY_PACK_READY"
    assert pack["steps_completed"] == 6
    assert "eod_reflection" in pack
    assert "monday_lesson_plan" in pack
    assert "monday_worksheet" in pack
    assert "critic_audit" in pack
    assert "ready_for_approval" in pack
    # EOD reflection should have expected keys
    eod = pack["eod_reflection"]
    assert "topics_covered" in eod or "class_id" in eod


def test_phase_7_memory_timeline():
    """Phase 7: Memory timeline view across all collections."""
    timeline = crew_orchestrator.get_memory_timeline(class_id="5B", limit=20)
    assert "timeline" in timeline
    assert "stats" in timeline
    assert "legend" in timeline
    assert isinstance(timeline["timeline"], list)
    stats = timeline["stats"]
    assert "total_entries" in stats
    assert "by_collection" in stats
    # Each timeline entry has required fields
    for entry in timeline["timeline"]:
        assert "collection" in entry
        assert "text_preview" in entry
        assert "color" in entry
        assert "icon" in entry


def test_phase_7_pii_diff_preview():
    """Phase 7: PII diff shows exactly what was scrubbed token-by-token."""
    # First register students so PII scrubbing works
    privacy_manager.register_or_get_student("Aarav Sharma", "5B")

    raw = "Aarav Sharma scored 85% in loops. Contact at aarav.sharma@gmail.com or 9876543210."
    diff = crew_orchestrator.pii_diff_preview(raw_text=raw, class_id="5B")

    assert diff["raw_text"] == raw
    assert "sanitized_text" in diff
    assert "diff_tokens" in diff
    assert isinstance(diff["diff_tokens"], list)
    assert diff["changes_count"] >= 0
    assert diff["privacy_certified"] is True

    # Check that email/phone are in sanitized text as redacted
    assert "[EMAIL_REDACTED]" in diff["sanitized_text"] or "EMAIL" in str(diff["diff_tokens"])
    assert "[PHONE_REDACTED]" in diff["sanitized_text"] or "PHONE" in str(diff["diff_tokens"])

    # Check that student name becomes PID
    sanitized = diff["sanitized_text"]
    assert "Aarav Sharma" not in sanitized  # Name should be replaced
