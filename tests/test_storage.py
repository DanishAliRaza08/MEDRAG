import sqlite3

import pytest

from medrag.storage import Database, ObjectStore
from medrag.storage.db import check_password, hash_password


def test_password_hashing():
    h = hash_password("secret")
    assert check_password("secret", h) and not check_password("wrong", h) and "secret" not in h


def test_audit_log_is_append_only():
    db = Database(":memory:")
    db.audit("u", "human", "login", "auth")
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("UPDATE audit_log SET actor='x'")
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("DELETE FROM audit_log")


def test_audit_chain_detects_tampering(tmp_path):
    db = Database(tmp_path / "a.db")
    for i in range(5):
        db.audit("u", "human", f"a{i}", "e")
    assert db.verify_audit_chain() == {"valid": True, "entries": 5, "head": db.audit_entries(1)[0]["hash"]}
    db.conn.execute("DROP TRIGGER audit_no_update")  # an attacker with file access
    db.conn.execute("UPDATE audit_log SET action='forged' WHERE seq=3")
    v = db.verify_audit_chain()
    assert v["valid"] is False and v["broken_at"] == 3


def test_object_store_roundtrip_and_key_validation(tmp_path):
    s = ObjectStore(tmp_path)
    k = s.put(b"abc", "studies", "png")
    assert s.get(k) == b"abc" and s.exists(k)
    with pytest.raises(ValueError):
        s.get("../../etc/passwd")


def test_users_and_reports():
    db = Database(":memory:")
    db.create_user("d", "pw", "clinician", "Dr D")
    assert db.authenticate("d", "pw")["role"] == "clinician" and db.authenticate("d", "x") is None
    with pytest.raises(sqlite3.IntegrityError):
        db.create_user("e", "pw", "superuser", "E")
    rid = db.create_report(None, "draft", {"a": 1}, "agent")
    db.set_report_status(rid, "signed_off", signed_by="d", comment="ok")
    r = db.get_report(rid)
    assert r["status"] == "signed_off" and r["signed_by"] == "d" and r["content"] == {"a": 1}
