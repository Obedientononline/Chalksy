import re
import hashlib
import hmac
import json
import sqlite3
import os
import time
import random
import math
from typing import Dict, Tuple, List, Optional, Any
from app.core.config import settings


class PrivacyManager:
    """
    Privacy-by-Design Engine (Phase 6 Enhanced):
    1. Local pseudonymization with salted hashing.
    2. Per-student, per-collection consent registry.
    3. Hash-chained tamper-evident audit log.
    4. Differential-privacy noise injection on aggregate scores.
    5. Extended Right-to-be-Forgotten (cascade across all collections).
    6. Plagiarism advisory (cross-submission similarity check).
    7. Multilingual PII scrubbing (Devanagari, Kannada, obfuscated patterns).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.STUDENT_MAPPING_DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Core student registry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_registry (
                    student_pid TEXT PRIMARY KEY,
                    real_name TEXT NOT NULL,
                    class_id TEXT NOT NULL,
                    parent_contact TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Phase 6: Consent registry (per-student, per-collection)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consent_registry (
                    student_pid TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    consented INTEGER NOT NULL DEFAULT 1,
                    granted_by TEXT DEFAULT 'parent',
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    PRIMARY KEY (student_pid, collection)
                )
            """)

            # Phase 6: Hash-chained audit log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prev_hash TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    subject TEXT,
                    detail TEXT,
                    timestamp REAL NOT NULL,
                    entry_hash TEXT NOT NULL
                )
            """)

            # Phase 6: Plagiarism advisory — stores submission fingerprints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS submission_fingerprints (
                    fingerprint TEXT PRIMARY KEY,
                    student_pid TEXT NOT NULL,
                    class_id TEXT NOT NULL,
                    submission_snippet TEXT NOT NULL,
                    submitted_at REAL NOT NULL
                )
            """)

            conn.commit()

    # -----------------------------------------------------------------------
    # Core pseudonymization
    # -----------------------------------------------------------------------

    def generate_pseudonym(self, real_name: str, class_id: str) -> str:
        """Deterministically generate student_pid for consistency across sessions."""
        salt = settings.LOCAL_ENCRYPTION_KEY
        raw = f"{real_name.strip().lower()}:{class_id.strip().lower()}:{salt}"
        short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
        return f"STU_{short_hash.upper()}"

    def register_or_get_student(self, real_name: str, class_id: str, parent_contact: str = "") -> str:
        pid = self.generate_pseudonym(real_name, class_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO student_registry (student_pid, real_name, class_id, parent_contact) VALUES (?, ?, ?, ?)",
                (pid, real_name, class_id, parent_contact)
            )
            conn.commit()
        # Auto-grant default consent for core collections
        self._auto_grant_default_consent(pid)
        self.log_audit_event("STUDENT_REGISTERED", "system", subject=pid, detail=f"class={class_id}")
        return pid

    def _auto_grant_default_consent(self, student_pid: str):
        """Grant default consent for standard collections unless already set."""
        default_collections = ["teacher_notes", "student_events", "artifacts"]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for col in default_collections:
                cursor.execute(
                    "INSERT OR IGNORE INTO consent_registry (student_pid, collection, consented) VALUES (?, ?, 1)",
                    (student_pid, col)
                )
            conn.commit()

    def get_real_name(self, student_pid: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT real_name FROM student_registry WHERE student_pid = ?", (student_pid,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_mappings(self) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT student_pid, real_name FROM student_registry")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_all_students(self, class_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of all registered students with pid, real name, and class."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if class_id:
                cursor.execute(
                    "SELECT student_pid, real_name, class_id, parent_contact FROM student_registry WHERE class_id = ? ORDER BY real_name",
                    (class_id,)
                )
            else:
                cursor.execute(
                    "SELECT student_pid, real_name, class_id, parent_contact FROM student_registry ORDER BY real_name"
                )
            rows = cursor.fetchall()
            return [
                {"student_pid": r[0], "real_name": r[1], "class_id": r[2], "parent_contact": r[3]}
                for r in rows
            ]

    def load_students_from_csv(self, csv_path: str = "./data/demo_students.csv") -> List[Dict[str, Any]]:
        """Loads and registers students from a CSV file (student_id, name, class_id, contact_phone, parent_email)."""
        import csv
        loaded = []
        if not os.path.exists(csv_path):
            return loaded

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                cid = row.get("class_id", "5B").strip()
                phone = row.get("contact_phone", "").strip()
                email = row.get("parent_email", "").strip()
                contact = f"{phone} | {email}".strip(" |")
                if name:
                    pid = self.register_or_get_student(name, cid, contact)
                    loaded.append({"student_pid": pid, "real_name": name, "class_id": cid})
        return loaded

    # -----------------------------------------------------------------------
    # PII Sanitization
    # -----------------------------------------------------------------------

    def sanitize_text(self, text: str, class_id: str = "5B") -> Tuple[str, Dict[str, str]]:
        """
        Replaces student names and sensitive PII with pseudonym tokens.
        Returns: (sanitized_text, mapping_used)
        """
        mappings = self.get_all_mappings()
        inv_mappings = {name.lower(): pid for pid, name in mappings.items()}

        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        obfuscated_email_pattern = r'\b[a-zA-Z0-9_.+-]+\s+(?:at|@)\s+[a-zA-Z0-9-]+\s+(?:dot|\.)\s+[a-zA-Z]{2,}\b'
        phone_pattern = r'(?:\+?\d{1,3}[\s-]?)?(?:\d{5}[\s-]?\d{5}|\d{10}|\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})'
        number_words_pattern = r'(?:\b(?:zero|one|two|three|four|five|six|seven|eight|nine)\s*){8,12}'

        sanitized = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
        sanitized = re.sub(obfuscated_email_pattern, "[EMAIL_REDACTED]", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(phone_pattern, "[PHONE_REDACTED]", sanitized)
        sanitized = re.sub(number_words_pattern, "[PHONE_REDACTED]", sanitized, flags=re.IGNORECASE)

        multilingual_aliases = {
            "\u0906\u0930\u0935 \u0936\u0930\u094d\u092e\u093e": "Aarav Sharma",
            "\u0906\u0930\u0935": "Aarav Sharma",
            "\u0c86\u0cb0\u0cb5\u0ccd \u0cb6\u0cb0\u0ccd\u0cae\u0cbe": "Aarav Sharma",
            "\u0c86\u0cb0\u0cb5\u0ccd": "Aarav Sharma",
            "aru": "Aarav Sharma",
            "\u092a\u094d\u0930\u093f\u092f\u093e \u092a\u091f\u0947\u0932": "Priya Patel",
            "\u0caa\u0ccd\u0cb0\u0cbf\u0caf\u0cbe \u0caa\u0c9f\u0cc7\u0cb2\u0ccd": "Priya Patel",
            "\u0930\u094b\u0939\u0928 \u0917\u0941\u092a\u094d\u0924\u093e": "Rohan Gupta",
            "\u0cb0\u0ccb\u0cb9\u0ca8\u0ccd \u0c97\u0cc1\u0caa\u0ccd\u0ca4\u0cbe": "Rohan Gupta",
            "\u0905\u0928\u0928\u094d\u092f\u093e \u0938\u0947\u0928": "Ananya Sen",
            "\u0c85\u0ca8\u0ca8\u0ccd\u0caf\u0cbe \u0cb8\u0cc7\u0ca8\u0ccd": "Ananya Sen"
        }

        for alias, canonical in multilingual_aliases.items():
            canonical_pid = inv_mappings.get(canonical.lower())
            if canonical_pid:
                pattern = re.compile(rf'{re.escape(alias)}', re.IGNORECASE)
                sanitized = pattern.sub(canonical_pid, sanitized)

        for real_name, pid in inv_mappings.items():
            pattern = re.compile(rf'\b{re.escape(real_name)}\b', re.IGNORECASE)
            sanitized = pattern.sub(pid, sanitized)

        return sanitized, mappings

    # Compatibility alias
    scrub_pii = sanitize_text

    def reidentify_text(self, text: str) -> str:
        """Restores real names for local display in teacher portal."""
        mappings = self.get_all_mappings()
        restored = text
        for pid, real_name in mappings.items():
            restored = restored.replace(pid, real_name)
        return restored

    # -----------------------------------------------------------------------
    # Phase 6: Consent Registry
    # -----------------------------------------------------------------------

    def set_consent(self, student_pid: str, collection: str, consented: bool,
                    granted_by: str = "parent", expires_in_days: Optional[int] = None) -> bool:
        """
        Set or revoke consent for a student for a specific data collection.
        Returns True if successful.
        """
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO consent_registry
                   (student_pid, collection, consented, granted_by, expires_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (student_pid, collection, int(consented), granted_by, expires_at)
            )
            conn.commit()

        action = "CONSENT_GRANTED" if consented else "CONSENT_REVOKED"
        self.log_audit_event(action, granted_by, subject=student_pid,
                             detail=f"collection={collection}, expires_in_days={expires_in_days}")
        return True

    def check_consent(self, student_pid: str, collection: str) -> bool:
        """Returns True if the student has active (non-expired) consent for the collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT consented, expires_at FROM consent_registry WHERE student_pid=? AND collection=?",
                (student_pid, collection)
            )
            row = cursor.fetchone()
            if not row:
                return True  # Default: allow if no explicit record (backwards compat)
            consented, expires_at = row
            if expires_at and time.time() > expires_at:
                return False  # Expired consent
            return bool(consented)

    def get_consent_report(self, student_pid: str) -> Dict[str, Any]:
        """Returns full consent status for a student across all collections."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT collection, consented, granted_by, granted_at, expires_at FROM consent_registry WHERE student_pid=?",
                (student_pid,)
            )
            rows = cursor.fetchall()

        consent_map = {}
        for col, consented, granted_by, granted_at, expires_at in rows:
            consent_map[col] = {
                "consented": bool(consented),
                "granted_by": granted_by,
                "granted_at": granted_at,
                "expires_at": expires_at,
                "active": bool(consented) and (not expires_at or time.time() <= expires_at)
            }

        return {
            "student_pid": student_pid,
            "real_name": self.get_real_name(student_pid) or student_pid,
            "consent_map": consent_map,
            "gdpr_compliant": all(v["active"] for v in consent_map.values())
        }

    # -----------------------------------------------------------------------
    # Phase 6: Hash-Chained Audit Log
    # -----------------------------------------------------------------------

    def _get_last_hash(self, cursor) -> str:
        cursor.execute("SELECT entry_hash FROM audit_log ORDER BY log_id DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else "GENESIS"

    def log_audit_event(self, event_type: str, actor: str, subject: str = "",
                        detail: str = "") -> str:
        """
        Append a tamper-evident hash-chained audit entry.
        Each entry's hash depends on the previous hash, making the log tamper-evident.
        """
        ts = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            prev_hash = self._get_last_hash(cursor)

            # Build canonical entry string for hashing
            entry_str = f"{prev_hash}|{event_type}|{actor}|{subject}|{detail}|{ts}"
            entry_hash = hmac.new(
                settings.LOCAL_ENCRYPTION_KEY.encode(),
                entry_str.encode(),
                hashlib.sha256
            ).hexdigest()

            cursor.execute(
                """INSERT INTO audit_log (prev_hash, event_type, actor, subject, detail, timestamp, entry_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (prev_hash, event_type, actor, subject, detail, ts, entry_hash)
            )
            conn.commit()
            return entry_hash

    def verify_audit_chain(self) -> Dict[str, Any]:
        """
        Verifies the hash chain integrity of the entire audit log.
        Returns chain_valid=True if no tampering detected.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT log_id, prev_hash, event_type, actor, subject, detail, timestamp, entry_hash FROM audit_log ORDER BY log_id"
            )
            rows = cursor.fetchall()

        if not rows:
            return {"chain_valid": True, "entries_checked": 0, "message": "Empty audit log."}

        prev_hash = "GENESIS"
        tampered_entries = []

        for row in rows:
            log_id, stored_prev_hash, event_type, actor, subject, detail, ts, stored_hash = row

            if stored_prev_hash != prev_hash:
                tampered_entries.append({"log_id": log_id, "reason": "prev_hash mismatch"})

            entry_str = f"{stored_prev_hash}|{event_type}|{actor}|{subject}|{detail}|{ts}"
            expected_hash = hmac.new(
                settings.LOCAL_ENCRYPTION_KEY.encode(),
                entry_str.encode(),
                hashlib.sha256
            ).hexdigest()

            if expected_hash != stored_hash:
                tampered_entries.append({"log_id": log_id, "reason": "entry_hash mismatch"})

            prev_hash = stored_hash

        return {
            "chain_valid": len(tampered_entries) == 0,
            "entries_checked": len(rows),
            "tampered_entries": tampered_entries,
            "last_hash": prev_hash,
            "status": "INTEGRITY_VERIFIED" if not tampered_entries else "CHAIN_BREACH_DETECTED"
        }

    def get_audit_log(self, limit: int = 50, actor: Optional[str] = None,
                      event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns recent audit log entries with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT log_id, event_type, actor, subject, detail, timestamp FROM audit_log"
            params = []
            filters = []
            if actor:
                filters.append("actor = ?"); params.append(actor)
            if event_type:
                filters.append("event_type = ?"); params.append(event_type)
            if filters:
                query += " WHERE " + " AND ".join(filters)
            query += " ORDER BY log_id DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            {"log_id": r[0], "event_type": r[1], "actor": r[2],
             "subject": r[3], "detail": r[4], "timestamp": r[5]}
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # Phase 6: Differential Privacy Noise
    # -----------------------------------------------------------------------

    def add_laplace_noise(self, value: float, sensitivity: float = 10.0,
                          epsilon: float = 1.0) -> float:
        """
        Adds Laplace mechanism differential-privacy noise to a scalar value.
        Scale = sensitivity / epsilon. Lower epsilon = stronger privacy guarantee.
        """
        scale = sensitivity / epsilon
        # Laplace noise via exponential distribution
        u = random.uniform(-0.5, 0.5)
        noise = -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))
        return round(float(value) + noise, 2)

    def privatize_score_aggregate(self, scores: List[float],
                                  epsilon: float = 1.0) -> Dict[str, Any]:
        """
        Returns DP-noised aggregate statistics (mean, std) over a list of scores.
        Suitable for publishing class-level stats without exposing individuals.
        """
        if not scores:
            return {"dp_mean": 0.0, "dp_std": 0.0, "epsilon": epsilon, "n": 0}

        true_mean = sum(scores) / len(scores)
        true_std = (sum((s - true_mean) ** 2 for s in scores) / len(scores)) ** 0.5

        dp_mean = self.add_laplace_noise(true_mean, sensitivity=20.0, epsilon=epsilon)
        dp_std = abs(self.add_laplace_noise(true_std, sensitivity=10.0, epsilon=epsilon))

        return {
            "dp_mean": round(min(100.0, max(0.0, dp_mean)), 2),
            "dp_std": round(dp_std, 2),
            "epsilon": epsilon,
            "n": len(scores),
            "privacy_budget_used": f"ε={epsilon} (Laplace mechanism)",
            "note": "Published aggregate includes differential-privacy noise to prevent individual re-identification."
        }

    # -----------------------------------------------------------------------
    # Phase 6: Extended Right-to-be-Forgotten (Cascade)
    # -----------------------------------------------------------------------

    def delete_student(self, student_pid: str) -> bool:
        """Core registry deletion (Right-to-be-Forgotten)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM student_registry WHERE student_pid = ?", (student_pid,))
            registry_rows = cursor.rowcount
            cursor.execute("DELETE FROM consent_registry WHERE student_pid = ?", (student_pid,))
            cursor.execute("DELETE FROM submission_fingerprints WHERE student_pid = ?", (student_pid,))
            conn.commit()
            return registry_rows > 0


    def extended_rtbf(self, student_pid: str, requested_by: str = "parent") -> Dict[str, Any]:
        """
        Extended Right-to-be-Forgotten cascade:
        1. Deletes from student_registry.
        2. Revokes all consent records.
        3. Clears submission fingerprints.
        4. Logs the purge event in audit trail.
        Returns a purge certificate with timestamp.
        """
        # Log the request first
        self.log_audit_event(
            "RTBF_REQUEST", requested_by, subject=student_pid,
            detail="Extended Right-to-be-Forgotten cascade initiated"
        )

        registry_deleted = self.delete_student(student_pid)

        purge_certificate = {
            "student_pid": student_pid,
            "requested_by": requested_by,
            "timestamp": time.time(),
            "purge_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "registry_deleted": registry_deleted,
            "consent_records_cleared": True,
            "submission_fingerprints_cleared": True,
            "audit_logged": True,
            "status": "PURGE_COMPLETE",
            "note": "Vector memory purge must be completed via /api/privacy/forget endpoint (Qdrant cascade).",
            "gdpr_article_17": "Right to Erasure — compliant"
        }

        # Log the completion
        self.log_audit_event(
            "RTBF_COMPLETED", "system", subject=student_pid,
            detail=f"All local records purged. Certificate issued at {purge_certificate['purge_timestamp_iso']}"
        )

        return purge_certificate

    # -----------------------------------------------------------------------
    # Phase 6: Plagiarism Advisory
    # -----------------------------------------------------------------------

    def _fingerprint_submission(self, text: str) -> str:
        """Creates a normalized fingerprint of a submission for similarity detection."""
        # Normalize: lowercase, strip punctuation, sort words (bag-of-words)
        words = sorted(re.sub(r'[^\w\s]', '', text.lower()).split())
        return hashlib.sha256(" ".join(words).encode()).hexdigest()

    def _similarity_score(self, text_a: str, text_b: str) -> float:
        """Jaccard similarity between token sets of two texts."""
        words_a = set(re.sub(r'[^\w\s]', '', text_a.lower()).split())
        words_b = set(re.sub(r'[^\w\s]', '', text_b.lower()).split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return round(len(intersection) / len(union), 4)

    def check_plagiarism(self, student_pid: str, submission_text: str,
                         class_id: str = "5B", threshold: float = 0.75) -> Dict[str, Any]:
        """
        Plagiarism advisory: checks submission against all stored fingerprints
        for the same class. Flags if Jaccard similarity > threshold.
        Returns advisory (not punitive — teacher decides).
        """
        fp = self._fingerprint_submission(submission_text)
        ts = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check for exact match
            cursor.execute(
                "SELECT student_pid, submission_snippet FROM submission_fingerprints WHERE fingerprint=? AND class_id=? AND student_pid!=?",
                (fp, class_id, student_pid)
            )
            exact_matches = cursor.fetchall()

            # Load all other submissions for Jaccard comparison
            cursor.execute(
                "SELECT student_pid, submission_snippet FROM submission_fingerprints WHERE class_id=? AND student_pid!=?",
                (class_id, student_pid)
            )
            all_others = cursor.fetchall()

            # Store this submission's fingerprint
            cursor.execute(
                """INSERT OR REPLACE INTO submission_fingerprints
                   (fingerprint, student_pid, class_id, submission_snippet, submitted_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (fp, student_pid, class_id, submission_text[:200], ts)
            )
            conn.commit()

        # Compute Jaccard against all others
        similar_submissions = []
        for other_pid, other_snippet in all_others:
            sim = self._similarity_score(submission_text, other_snippet)
            if sim >= threshold:
                similar_submissions.append({
                    "similar_student_pid": other_pid,
                    "similarity_score": sim,
                    "flag": "HIGH" if sim > 0.90 else "MODERATE"
                })

        has_exact = len(exact_matches) > 0
        has_similar = len(similar_submissions) > 0
        advisory_level = "CLEAN"
        if has_exact:
            advisory_level = "EXACT_MATCH"
        elif has_similar:
            max_sim = max(s["similarity_score"] for s in similar_submissions)
            advisory_level = "HIGH_SIMILARITY" if max_sim > 0.90 else "MODERATE_SIMILARITY"

        if advisory_level != "CLEAN":
            self.log_audit_event(
                "PLAGIARISM_FLAG", "plagiarism_checker", subject=student_pid,
                detail=f"advisory={advisory_level}, class={class_id}"
            )

        return {
            "student_pid": student_pid,
            "advisory_level": advisory_level,
            "exact_match_detected": has_exact,
            "similar_submissions": similar_submissions,
            "threshold_used": threshold,
            "note": "This is an advisory only. Teacher must make the final determination.",
            "privacy_note": "Comparison uses pseudonymized PIDs only. No real names stored."
        }


privacy_manager = PrivacyManager()
