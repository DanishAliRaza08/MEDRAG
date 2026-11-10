import json
import sqlite3

import pytest

from medrag.imaging import synthetic
from medrag.reports.lifecycle import SignOffError
from medrag.service import NotFound


def _case(service, finding="effusion"):
    clin = service.db.get_user("clinician")
    pt = service.register_patient(clin, "MRN-777", 72, "f")
    png = synthetic.to_png_bytes(synthetic.chest_xray(finding, seed=2, burn_text=True), PatientName="Jane Roe")
    st = service.ingest_study(clin, pt["patient_ref"], "xray", png,
                              "Patient: Jane Roe, MRN 7771234, seen 01/02/2025, phone 0300-1234567.")
    return clin, pt, st


def test_no_identifiable_data_is_stored(service, settings):
    clin, pt, st = _case(service)
    assert st["deid_summary"]["masked_regions"] > 0 and "PatientName" in st["deid_summary"]["metadata_removed"]
    assert "Jane" not in st["clinical_note"] and "7771234" not in st["clinical_note"]
    service.analyze(clin, st["id"], "Patient Jane Roe breathless, review film")
    # Scan every byte the system persisted for the raw identifiers.
    blobs = [settings.db_path.read_bytes()] + [p.read_bytes() for p in settings.data_dir.rglob("*") if p.is_file()]
    for needle in (b"Jane", b"MRN-777", b"7771234", b"0300-1234567", b"JOHN DOE"):
        assert not any(needle in b for b in blobs), needle


def test_patient_identifier_stored_only_as_hash(service):
    _case(service)
    rows = service.db.list_patients()
    assert len(rows) == 1 and rows[0]["patient_ref"].startswith("pt_") and rows[0]["age_band"] == "65+"


def test_analyze_creates_verified_draft_never_signed(service):
    clin, _, st = _case(service)
    rep = service.analyze(clin, st["id"], "review chest radiograph")
    assert rep["status"] == "verified" and rep["signed_by"] is None
    assert rep["content"]["draft"]["claims"]


def test_agent_cannot_sign_off_and_attempt_is_audited(service):
    clin, _, st = _case(service)
    rep = service.analyze(clin, st["id"], "review")
    with pytest.raises(SignOffError):
        service.agent_attempt_signoff(rep["id"])
    assert service.db.get_report(rep["id"])["status"] == "verified"
    assert any(e["action"] == "signed_off_denied" for e in service.db.audit_entries(entity=rep["id"]))


def test_clinician_sign_off_flow_and_audit_chain(service):
    clin, _, st = _case(service)
    rep = service.analyze(clin, st["id"], "review")
    rep = service.sign_off(clin, rep["id"], "agree")
    assert rep["status"] == "signed_off" and rep["signed_by"] == "clinician"
    with pytest.raises(SignOffError):
        service.reject(clin, rep["id"], "changed my mind")
    actions = [e["action"] for e in service.db.audit_entries()]
    assert {"ingest_study", "request_analysis", "draft_verified", "signed_off"} <= set(actions)
    assert service.db.verify_audit_chain()["valid"]


def test_admin_cannot_sign_off(service):
    clin, _, st = _case(service)
    rep = service.analyze(clin, st["id"], "review")
    with pytest.raises(SignOffError):
        service.sign_off(service.db.get_user("admin"), rep["id"])


def test_unknown_patient_and_modality(service):
    clin = service.db.get_user("clinician")
    png = synthetic.to_png_bytes(synthetic.chest_xray())
    with pytest.raises(NotFound):
        service.ingest_study(clin, "pt_missing", "xray", png)
    pt = service.register_patient(clin, "X1", 30, "M")
    with pytest.raises(ValueError):
        service.ingest_study(clin, pt["patient_ref"], "mri", png)


def test_audit_table_append_only_via_service(service):
    _case(service)
    with pytest.raises(sqlite3.IntegrityError):
        service.db.conn.execute("DELETE FROM audit_log")


def test_kb_document_ingestion(service):
    admin = service.db.get_user("admin")
    out = service.add_kb_document(admin, "Sarcoid", "Sarcoidosis causes bilateral hilar lymphadenopathy. "
                                  "Contact Dr. Ahmed Khan on 0300-1234567.", "test", ["xray"])
    assert out["chunks"] == 1 and "0300" not in out["text"]
    assert service.pipeline.search("hilar lymphadenopathy sarcoidosis").hits[0].doc_id == out["doc_id"]


def test_similar_cases_after_signoff(service):
    clin, pt, st = _case(service, "cardiomegaly")
    service.sign_off(clin, service.analyze(clin, st["id"], "review")["id"])
    png = synthetic.to_png_bytes(synthetic.chest_xray("cardiomegaly", seed=8))
    st2 = service.ingest_study(clin, pt["patient_ref"], "xray", png)
    rep = service.analyze(clin, st2["id"], "review")
    sims = rep["content"]["similar_cases"]
    assert sims and sims[0]["study_id"] == st["id"] and "cardiomegaly" in sims[0]["finding_summary"]
    assert json.dumps(sims).count("pt_") == 0  # no patient refs in case payloads


def test_index_rebuilds_when_kb_changes_and_keeps_uploads(settings, tmp_path):
    import shutil

    from medrag.config import ROOT_DIR
    from medrag.service import MedRAGService

    kb = tmp_path / "kb"
    shutil.copytree(ROOT_DIR / "data" / "knowledge_base", kb)
    settings.kb_dir = kb
    svc = MedRAGService(settings)
    admin = svc.db.get_user("admin")
    up = svc.add_kb_document(admin, "Sarcoid", "Sarcoidosis causes bilateral hilar lymphadenopathy.", "t", ["xray"])
    (kb / "new.md").write_text("---\nid: kb-new\ntitle: New\nmodality: xray\n---\nAsbestosis causes pleural plaques.")
    svc2 = MedRAGService(settings)  # restart: sources changed -> rebuild
    assert svc2.pipeline.search("asbestosis pleural plaques").hits[0].doc_id == "kb-new"
    assert svc2.pipeline.search("sarcoidosis hilar").hits[0].doc_id == up["doc_id"]
    svc3 = MedRAGService(settings)  # unchanged -> loaded from disk
    assert svc3.pipeline.stats()["chunks"] == svc2.pipeline.stats()["chunks"]
