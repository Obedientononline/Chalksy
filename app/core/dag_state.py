import sqlite3
import json
import time
import os
from typing import List, Dict, Any, Optional
from app.core.config import settings

class TaskStateDB:
    """
    Typed DAG Task State Database (SQLite)
    Manages:
      - task_id, parent_id, dag_id
      - agent_name, status ('PENDING', 'RUNNING', 'COMPLETED', 'AWAITING_APPROVAL', 'FAILED')
      - inputs, memory_refs, outputs
      - approval flags, timestamps
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.TASK_STATE_DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_dag (
                    task_id TEXT PRIMARY KEY,
                    dag_id TEXT NOT NULL,
                    parent_id TEXT,
                    agent_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    inputs TEXT,
                    memory_refs TEXT,
                    output TEXT,
                    status TEXT NOT NULL,
                    approved BOOLEAN DEFAULT 0,
                    critic_feedback TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()

    def create_task(
        self,
        task_id: str,
        dag_id: str,
        agent_name: str,
        title: str,
        parent_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        memory_refs: Optional[List[Dict[str, Any]]] = None
    ):
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO task_dag (
                    task_id, dag_id, parent_id, agent_name, title, inputs, memory_refs, output, status, approved, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                dag_id,
                parent_id,
                agent_name,
                title,
                json.dumps(inputs or {}),
                json.dumps(memory_refs or []),
                json.dumps({}),
                "PENDING",
                0,
                now,
                now
            ))
            conn.commit()

    def update_task_status(
        self,
        task_id: str,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        critic_feedback: Optional[str] = None
    ):
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if output is not None and critic_feedback is not None:
                cursor.execute("""
                    UPDATE task_dag 
                    SET status = ?, output = ?, critic_feedback = ?, updated_at = ?
                    WHERE task_id = ?
                """, (status, json.dumps(output), critic_feedback, now, task_id))
            elif output is not None:
                cursor.execute("""
                    UPDATE task_dag 
                    SET status = ?, output = ?, updated_at = ?
                    WHERE task_id = ?
                """, (status, json.dumps(output), now, task_id))
            else:
                cursor.execute("""
                    UPDATE task_dag 
                    SET status = ?, updated_at = ?
                    WHERE task_id = ?
                """, (status, now, task_id))
            conn.commit()

    def approve_task(self, task_id: str) -> bool:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE task_dag 
                SET approved = 1, status = 'APPROVED', updated_at = ?
                WHERE task_id = ?
            """, (now, task_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_dag_tasks(self, dag_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM task_dag WHERE dag_id = ? ORDER BY created_at ASC", (dag_id,))
            rows = cursor.fetchall()
            return [self._format_row(r) for r in rows]

    def get_all_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM task_dag ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [self._format_row(r) for r in rows]

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM task_dag WHERE status = 'AWAITING_APPROVAL' ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._format_row(r) for r in rows]

    def _format_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["inputs"] = json.loads(d["inputs"] or "{}")
        except Exception:
            pass
        try:
            d["memory_refs"] = json.loads(d["memory_refs"] or "[]")
        except Exception:
            pass
        try:
            d["output"] = json.loads(d["output"] or "{}")
        except Exception:
            pass
        return d

task_db = TaskStateDB()
