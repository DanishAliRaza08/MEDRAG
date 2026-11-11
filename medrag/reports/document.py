"""PDF report generation (ReportLab).

Unsigned reports carry a diagonal "DRAFT - NOT FOR CLINICAL USE" watermark on
every page; only a signed-off report is rendered without it.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

WATERMARK = "DRAFT - NOT FOR CLINICAL USE"
SECTIONS = [("findings", "Findings"), ("impression", "Impression"),
            ("differential", "Differential considerations"), ("recommendations", "Recommendations")]


def _watermark(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 40)
    canvas.setFillColor(colors.Color(0.85, 0.1, 0.1, alpha=0.18))
    canvas.translate(A4[0] / 2, A4[1] / 2)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, WATERMARK)
    canvas.restoreState()
    _footer(canvas, doc)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    canvas.drawString(15 * mm, 10 * mm, "MedRAG clinical decision support - AI-assisted draft. "
                                        "Accountability rests with the signing clinician.")
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def render_report_pdf(report: dict, study: dict, image_png: bytes | None = None) -> bytes:
    """``report``: row dict with ``status``, ``content`` (agent result), sign-off fields."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
                            bottomMargin=18 * mm, title=f"MedRAG report {report['id']}", author="MedRAG",
                            pageCompression=0)
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.grey)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=12.5)
    content = report["content"]
    signed = report["status"] == "signed_off"
    story = [Paragraph("MedRAG - Evidence-Grounded Diagnostic Report", ss["Title"])]
    status_txt = {"signed_off": "SIGNED OFF", "rejected": "REJECTED"}.get(report["status"], "DRAFT (unsigned)")
    meta = [
        ["Report ID", report["id"], "Status", status_txt],
        ["Study ID", study["id"], "Modality", study.get("modality", "-")],
        ["Patient ref", study.get("patient_ref", "-")[:20] + "...", "Age band", study.get("age_band", "-")],
        ["Agent status", content.get("status", "-"), "Iterations", str(content.get("iterations", "-"))],
    ]
    t = Table(meta, colWidths=[28 * mm, 62 * mm, 28 * mm, 62 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
                           ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8.5),
                           ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 8.5),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke)]))
    story += [t, Spacer(1, 6)]
    story.append(Paragraph(f"<b>Clinical question:</b> {escape(content.get('question', ''))}", body))
    if image_png:
        story += [Spacer(1, 4), RLImage(io.BytesIO(image_png), width=55 * mm, height=55 * mm)]
    draft = content.get("draft") or {}
    if draft.get("summary"):
        story += [Spacer(1, 4), Paragraph(escape(draft["summary"]), body)]
    claims = draft.get("claims", [])
    for key, label in SECTIONS:
        items = [c for c in claims if c["section"] == key]
        if not items:
            continue
        story += [Spacer(1, 6), Paragraph(label, ss["Heading3"])]
        for c in items:
            cites = "".join(f"[{i}]" for i in c.get("citations", []))
            flag = "" if c.get("supported", True) else (
                ' <font color="red">(contradicts cited source)</font>' if (c.get("support_score") or 0) < 0
                else ' <font color="red">(unsupported)</font>')
            src = " <i>(image)</i>" if c.get("source") == "image" else ""
            story.append(Paragraph(f"&bull; {escape(c['text'])} {cites}{src}{flag}", body))
    evidence = content.get("evidence", [])
    if evidence:
        story += [Spacer(1, 6), Paragraph("Cited evidence", ss["Heading3"])]
        for i, e in enumerate(evidence, start=1):
            story.append(Paragraph(f"[{i}] <b>{escape(e['title'])}</b> - {escape(e['source'])} "
                                   f"({escape(e['chunk_id'])})", small))
    ver = content.get("verification") or {}
    if ver:
        story += [Spacer(1, 6), Paragraph(
            f"Self-verification: {ver.get('unsupported', 0)} unsupported of "
            f"{ver.get('supported', 0) + ver.get('unsupported', 0)} claims "
            f"(ungrounded rate {ver.get('hallucination_rate', 0):.2f}).", small)]
    story += [Spacer(1, 10), Paragraph("Clinician sign-off", ss["Heading3"])]
    if signed:
        story.append(Paragraph(f"Signed off by <b>{escape(report['signed_by'])}</b> at {escape(report['signed_at'])}."
                               + (f" Comment: {escape(report['comment'])}" if report.get("comment") else ""), body))
    elif report["status"] == "rejected":
        story.append(Paragraph(f"Rejected by <b>{escape(report.get('signed_by') or '')}</b>. Reason: "
                               f"{escape(report.get('comment') or '')}", body))
    else:
        story.append(Paragraph("NOT SIGNED. This draft has not been reviewed by a clinician and must not be used "
                               "for patient care.", body))
    story.append(Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", small))
    deco = _footer if signed else _watermark
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()
