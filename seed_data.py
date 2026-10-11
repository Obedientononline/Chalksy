import time
from app.memory.qdrant_store import memory_manager
from app.privacy.anonymizer import privacy_manager

def seed_demo_environment():
    """
    Seeds initial synthetic classroom data:
    1. CBSE/ICSE Grade 5 Robotics & Computational Thinking Curriculum
    2. Synthetic Student Cohort
    3. Initial Classroom Observations
    """
    print("[SEED] Seeding Synthetic Classroom & Curriculum Memories...")

    # 1. Register synthetic student cohort
    students = [
        ("Aarav Sharma", "5B", "+91 98765 43210"),
        ("Priya Patel", "5B", "+91 98765 43211"),
        ("Rohan Gupta", "5B", "+91 98765 43212"),
        ("Ananya Sen", "5B", "+91 98765 43213"),
        ("Devansh Mehta", "5B", "+91 98765 43214"),
        ("Kavya Nair", "5B", "+91 98765 43215")
    ]

    pids = {}
    for name, cid, contact in students:
        pids[name] = privacy_manager.register_or_get_student(name, cid, contact)

    # 2. Seed Curriculum Units (CBSE / ICSE Grade 5 Computer Science)
    curriculum_units = [
        {
            "text": "CBSE Grade 5 CS Unit 4: Iterative Statements. Students understand the need for loops (while, for), conditional stops, and sensor input handling.",
            "payload": {"board": "CBSE", "grade": "5", "topic": "Loops & Iterations", "bloom_level": "Understanding", "source": "official_curriculum"}
        },
        {
            "text": "ICSE Grade 5 Science & Robotics Unit 2: Sonar & Ultrasonic distance measurement. Speed of sound calculation, distance thresholding for collision avoidance.",
            "payload": {"board": "ICSE", "grade": "5", "topic": "Ultrasonic Sensors", "bloom_level": "Application", "source": "official_curriculum"}
        },
        {
            "text": "Bloom Taxonomy Benchmark: Creating autonomous obstacle avoidance algorithms by nesting conditional branches inside continuous sensor poll loops.",
            "payload": {"board": "General", "grade": "5", "topic": "Autonomous Navigation", "bloom_level": "Synthesis", "source": "pedagogy_guide"}
        }
    ]

    for item in curriculum_units:
        memory_manager.add_memory(
            collection="curriculum",
            text=item["text"],
            payload=item["payload"],
            provenance="curriculum_syllabus"
        )

    # 3. Seed Prior Spoken Teacher Notes
    past_notes = [
        {
            "text": "Class 5B completed basic boolean variables last Tuesday. All students passed the initialization quiz.",
            "payload": {"class_id": "5B", "topic": "Variables", "timestamp": time.time() - 86400 * 5, "source": "teacher_voice"}
        },
        {
            "text": "Three students in group 2 hesitated when explaining why an exit condition is needed in a while loop.",
            "payload": {"class_id": "5B", "topic": "Loops", "timestamp": time.time() - 86400 * 2, "source": "teacher_voice"}
        }
    ]

    for item in past_notes:
        memory_manager.add_memory(
            collection="teacher_notes",
            text=item["text"],
            payload=item["payload"],
            provenance="ambient_voice"
        )

    # 4. Seed Past Student Events (Pseudonymized Misconceptions)
    events = [
        {
            "text": f"Student {pids['Aarav Sharma']} forgot to update the loop counter variable, creating an infinite loop in the simulator.",
            "payload": {"student_pid": pids["Aarav Sharma"], "class_id": "5B", "misconception": "Infinite Loop / Missing Counter Update", "score": 70, "visibility": "private"}
        },
        {
            "text": f"Student {pids['Priya Patel']} mastered nested sensor checks with zero syntax errors.",
            "payload": {"student_pid": pids["Priya Patel"], "class_id": "5B", "mastery": "Nested Conditions", "score": 96, "visibility": "private"}
        }
    ]

    for item in events:
        memory_manager.add_memory(
            collection="student_events",
            text=item["text"],
            payload=item["payload"],
            provenance="grader_agent"
        )

    print("[SUCCESS] Seed complete! All memories indexed in Qdrant with zero PII leakage.")

if __name__ == "__main__":
    seed_demo_environment()
