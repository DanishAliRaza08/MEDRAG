import pytest

from medrag.reports.document import WATERMARK, render_report_pdf
from medrag.reports.lifecycle import Actor, ReportStatus, SignOffError, transition

AGENT = Actor("agent", "agent")
SYSTEM = Actor("sys", "system")
DOC = Actor("dr", "human", "clinician")
ADMIN = Actor("adm", "human", "admin")
VIEWER = Actor("v", "human", "viewer")


def test_agent_can_record_verification():
    assert transition("draft", "verified", AGENT) == ReportStatus.VERIFIED
    assert transition("draft", "needs_review", AGENT) == ReportStatus.NEEDS_REVIEW


@pytest.mark.parametrize("actor", [AGENT, SYSTEM])
@pytest.mark.parametrize("start", ["draft", "verified", "needs_review"])
def test_non_human_can_never_sign_off(actor, start):
    with pytest.raises(SignOffError):
        transition(start, "signed_off", actor)


@pytest.mark.parametrize("actor", [ADMIN, VIEWER])
def test_non_clinician_humans_cannot_sign_off(actor):
    with pytest.raises(SignOffError, match="not a credentialed clinician"):
        transition("verified", "signed_off", actor)


def test_clinician_signs_off_and_terminal_is_final():
    assert transition("verified", "signed_off", DOC) == ReportStatus.SIGNED_OFF
    for target in ("rejected", "verified", "signed_off"):
        with pytest.raises(SignOffError):
            transition("signed_off", target, DOC)


def test_reject_requires_reason():
    with pytest.raises(SignOffError):
        transition("needs_review", "rejected", DOC, "")
    assert transition("needs_review", "rejected", DOC, "wrong side") == ReportStatus.REJECTED


def test_humans_cannot_forge_verification():
    with pytest.raises(SignOffError):
        transition("draft", "verified", DOC)


def test_illegal_transition():
    with pytest.raises(SignOffError):
        transition("verified", "needs_review", AGENT)


def _report(status, **kw):
    return {"id": "rp_1", "status": status, "signed_by": kw.get("by"), "signed_at": "2026-01-01", "comment": None,
            "content": {"question": "q <script>", "status": "verified", "iterations": 1,
                        "draft": {"summary": "s", "claims": [
                            {"text": "Consolidation & opacity", "section": "findings", "citations": [],
                             "source": "image"},
                            {"text": "Pneumonia appears as consolidation.", "section": "impression", "citations": [1],
                             "source": "evidence", "supported": True}]},
                        "evidence": [{"title": "CAP", "source": "x", "chunk_id": "kb-cap#0"}],
                        "verification": {"supported": 2, "unsupported": 0, "hallucination_rate": 0.0}}}


def test_unsigned_pdf_has_watermark():
    pdf = render_report_pdf(_report("verified"), {"id": "st_1", "modality": "xray", "patient_ref": "pt_abc",
                                                  "age_band": "40-64"})
    assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    assert WATERMARK.encode() in pdf and b"NOT SIGNED" in pdf


def test_signed_pdf_has_no_watermark():
    pdf = render_report_pdf(_report("signed_off", by="dr"), {"id": "st_1"})
    assert WATERMARK.encode() not in pdf and b"Signed off by" in pdf
