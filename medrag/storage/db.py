"""Relational storage (SQLite) with an append-only, hash-chained audit log.

The audit table is protected twice: SQL triggers abort any UPDATE or DELETE,
and every row stores the SHA-256 of its content plus the previous row's hash,
so tampering with the file directly is detectable with ``verify_audit_chain``.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('clinician', 'admin', 'viewer')),
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS patients (
    patient_ref TEXT PRIMARY KEY,          -- irreversible HMAC pseudonym
    age_band TEXT NOT NULL,
    sex TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY,
    patient_ref TEXT NOT NULL REFERENCES patients(patient_ref),
    modality TEXT NOT NULL,
    age_band TEXT NOT NULL,
    clinical_note TEXT NOT NULL DEFAULT '',  -- PHI-scrubbed
    image_key TEXT NOT NULL,
    width INTEGER, height INTEGER,
    deid_summary TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    study_id TEXT REFERENCES studies(id),
    status TEXT NOT NULL,
    content TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    signed_by TEXT,
    signed_at TEXT,
    comment TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    action TEXT NOT NULL,
    entity TEXT NOT NULL,
    details TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, _ = stored.split("$")
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(password, salt), stored)


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def _q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _x(self, sql: str, args=()) -> None:
        with self.lock:
            self.conn.execute(sql, args)

    # ------------------------------------------------------------------ users
    def create_user(self, username: str, password: str, role: str, full_name: str) -> None:
        self._x("INSERT INTO users VALUES (?,?,?,?,?)", (username, full_name, role, hash_password(password), now()))

    def get_user(self, username: str) -> dict | None:
        rows = self._q("SELECT * FROM users WHERE username=?", (username,))
        return rows[0] if rows else None

    def authenticate(self, username: str, password: str) -> dict | None:
        u = self.get_user(username)
        return u if u and check_password(password, u["password_hash"]) else None

    def count_users(self) -> int:
        return self._q("SELECT COUNT(*) n FROM users")[0]["n"]

    # --------------------------------------------------------------- patients
    def upsert_patient(self, patient_ref: str, age_band: str, sex: str) -> None:
        self._x("INSERT INTO patients VALUES (?,?,?,?) "
                "ON CONFLICT(patient_ref) DO UPDATE SET age_band=excluded.age_band",
                (patient_ref, age_band, sex, now()))

    def list_patients(self) -> list[dict]:
        return self._q("SELECT p.*, (SELECT COUNT(*) FROM studies s WHERE s.patient_ref=p.patient_ref) AS studies "
                       "FROM patients p ORDER BY created_at DESC")

    def get_patient(self, ref: str) -> dict | None:
        r = self._q("SELECT * FROM patients WHERE patient_ref=?", (ref,))
        return r[0] if r else None

    # ---------------------------------------------------------------- studies
    def create_study(self, **f) -> str:
        sid = "st_" + uuid.uuid4().hex[:12]
        self._x("INSERT INTO studies (id, patient_ref, modality, age_band, clinical_note, image_key, width, height, "
                "deid_summary, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (sid, f["patient_ref"], f["modality"], f["age_band"], f.get("clinical_note", ""), f["image_key"],
                 f.get("width"), f.get("height"), json.dumps(f["deid_summary"]), f["created_by"], now()))
        return sid

    def get_study(self, sid: str) -> dict | None:
        r = self._q("SELECT * FROM studies WHERE id=?", (sid,))
        if not r:
            return None
        r[0]["deid_summary"] = json.loads(r[0]["deid_summary"])
        return r[0]

    def list_studies(self, patient_ref: str | None = None) -> list[dict]:
        sql = ("SELECT s.id, s.patient_ref, s.modality, s.age_band, s.clinical_note, s.created_at, s.created_by, "
               "(SELECT status FROM reports r WHERE r.study_id=s.id ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1)"
               " AS latest_report_status FROM studies s")
        if patient_ref:
            return self._q(sql + " WHERE s.patient_ref=? ORDER BY s.created_at DESC", (patient_ref,))
        return self._q(sql + " ORDER BY s.created_at DESC")

    # ---------------------------------------------------------------- reports
    def create_report(self, study_id: str | None, status: str, content: dict, created_by: str) -> str:
        rid = "rp_" + uuid.uuid4().hex[:12]
        self._x("INSERT INTO reports (id, study_id, status, content, created_by, created_at) VALUES (?,?,?,?,?,?)",
                (rid, study_id, status, json.dumps(content), created_by, now()))
        return rid

    def get_report(self, rid: str) -> dict | None:
        r = self._q("SELECT * FROM reports WHERE id=?", (rid,))
        if not r:
            return None
        r[0]["content"] = json.loads(r[0]["content"])
        return r[0]

    def list_reports(self, study_id: str | None = None) -> list[dict]:
        sql = "SELECT id, study_id, status, created_by, created_at, signed_by, signed_at FROM reports"
        if study_id:
            return self._q(sql + " WHERE study_id=? ORDER BY created_at DESC, rowid DESC", (study_id,))
        return self._q(sql + " ORDER BY created_at DESC, rowid DESC")

    def set_report_status(self, rid: str, status: str, signed_by: str | None = None, comment: str | None = None):
        """Low-level write. Callers MUST go through ``reports.lifecycle.transition`` first."""
        self._x("UPDATE reports SET status=?, signed_by=COALESCE(?, signed_by), "
                "signed_at=CASE WHEN ? IS NOT NULL THEN ? ELSE signed_at END, comment=COALESCE(?, comment) WHERE id=?",
                (status, signed_by, signed_by, now(), comment, rid))

    # ------------------------------------------------------------------ audit
    @staticmethod
    def _row_hash(ts, actor, kind, action, entity, details, prev) -> str:
        return hashlib.sha256("|".join([ts, actor, kind, action, entity, details, prev]).encode()).hexdigest()

    def audit(self, actor: str, actor_kind: str, action: str, entity: str, details: dict | None = None) -> None:
        with self.lock:
            last = self.conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last["hash"] if last else "0" * 64
            ts, d = now(), json.dumps(details or {}, sort_keys=True)
            h = self._row_hash(ts, actor, actor_kind, action, entity, d, prev)
            self.conn.execute("INSERT INTO audit_log (ts, actor, actor_kind, action, entity, details, prev_hash, hash) "
                              "VALUES (?,?,?,?,?,?,?,?)", (ts, actor, actor_kind, action, entity, d, prev, h))

    def audit_entries(self, limit: int = 200, entity: str | None = None) -> list[dict]:
        if entity:
            return self._q("SELECT * FROM audit_log WHERE entity=? ORDER BY seq DESC LIMIT ?", (entity, limit))
        return self._q("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))

    def verify_audit_chain(self) -> dict:
        prev = "0" * 64
        rows = self._q("SELECT * FROM audit_log ORDER BY seq")
        for r in rows:
            expect = self._row_hash(r["ts"], r["actor"], r["actor_kind"], r["action"], r["entity"], r["details"], prev)
            if r["prev_hash"] != prev or r["hash"] != expect:
                return {"valid": False, "entries": len(rows), "broken_at": r["seq"]}
            prev = r["hash"]
        return {"valid": True, "entries": len(rows), "head": prev}
