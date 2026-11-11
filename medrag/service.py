"""Application service: the single place where storage, retrieval, the agent,
de-identification and the report lifecycle meet.

Privacy is enforced *here*, at ingestion, rather than trusted to callers:
raw identifiers are converted to an HMAC pseudonym and an age band, free text
is scrubbed, images are re-encoded and burned-in text is masked before any byte
reaches the database, the object store or the vector index.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict
from pathlib import Path

from .agent import AgentConfig, MedRAGAgent
from .config import Settings, get_settings
from .deid import age_band, deidentify_image, pseudonymize, scrub_text
from .imaging.encoder import BiomedCLIPEncoder, CaseIndex, ThumbnailEncoder
from .reports.document import render_report_pdf
from .reports.lifecycle import Actor, ReportStatus, SignOffError, transition
from .retrieval import RetrievalPipeline
from .retrieval.chunking import Document, doc_id_for
from .storage import Database, ObjectStore

log = logging.getLogger("medrag")

MODALITIES = ("xray", "fundus", "histopathology")
AGENT = Actor("medrag-agent", "agent")


class NotFound(LookupError):
    pass


class MedRAGService:
    def __init__(self, settings: Settings | None = None, db: Database | None = None):
        self.s = settings or get_settings()
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or Database(self.s.db_path)
        self.objects = ObjectStore(self.s.object_dir)
        self.uploads_dir = self.s.data_dir / "kb_uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = self._load_or_build_index()
        encoder = ThumbnailEncoder() if self.s.offline else BiomedCLIPEncoder(self.s.image_encoder)
        self.cases = CaseIndex(encoder)
        self.agent = MedRAGAgent(self.pipeline, self.s, case_index=self.cases)
        self._rebuild_case_index()
        if self.s.seed_demo and self.db.count_users() == 0:
            self.seed_users()

    # -------------------------------------------------------------- bootstrap
    def _kb_fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.s.mode}|{self.s.chunk_tokens}|{self.s.chunk_overlap}".encode())
        for d in (*self.s.kb_dirs, self.uploads_dir):
            for p in sorted(Path(d).glob("*.md")):
                h.update(p.name.encode() + p.read_bytes())
        return h.hexdigest()

    def _load_or_build_index(self) -> RetrievalPipeline:
        """Reuse the persisted index only if the knowledge-base sources are unchanged."""
        fp, fp_file = self._kb_fingerprint(), self.s.index_dir / "kb_fingerprint"
        pipeline = RetrievalPipeline(self.s)
        if fp_file.exists() and fp_file.read_text() == fp and pipeline.load():
            return pipeline
        pipeline = RetrievalPipeline(self.s)
        n = sum(pipeline.ingest_directory(d) for d in (*self.s.kb_dirs, self.uploads_dir) if Path(d).exists())
        pipeline.save()
        fp_file.write_text(fp)
        log.info("indexed %d chunks from %s", n, [str(d) for d in (*self.s.kb_dirs, self.uploads_dir)])
        return pipeline

    def seed_users(self) -> None:
        for u, p, r, n in [("clinician", "clinician123", "clinician", "Dr. Demo Clinician"),
                           ("admin", "admin123", "admin", "System Administrator"),
                           ("viewer", "viewer123", "viewer", "Medical Student (read-only)")]:
            self.db.create_user(u, p, r, n)
        self.db.audit("system", "system", "seed_users", "users", {"count": 3})

    def _rebuild_case_index(self) -> None:
        for st in self.db.list_studies():
            full = self.db.get_study(st["id"])
            reports = [r for r in self.db.list_reports(st["id"]) if r["status"] == "signed_off"]
            summary = ""
            if reports:
                rep = self.db.get_report(reports[0]["id"])
                summary = ", ".join(f["concept"] for f in rep["content"].get("findings", []))
            try:
                self.cases.add(st["id"], self.objects.get(full["image_key"]), st["modality"], st["age_band"],
                               summary or "unreported")
            except (OSError, ValueError):  # pragma: no cover
                continue

    # --------------------------------------------------------------- patients
    def register_patient(self, user: dict, patient_id: str, age: int | None, sex: str) -> dict:
        ref = pseudonymize(patient_id, self.s.patient_salt)
        band = age_band(age)
        sex = sex.upper()[:1] if sex and sex.upper()[:1] in ("M", "F") else "U"
        self.db.upsert_patient(ref, band, sex)
        self.db.audit(user["username"], "human", "register_patient", ref, {"age_band": band})
        return {"patient_ref": ref, "age_band": band, "sex": sex}

    # ---------------------------------------------------------------- studies
    def ingest_study(self, user: dict, patient_ref: str, modality: str, image_bytes: bytes,
                     clinical_note: str = "") -> dict:
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        patient = self.db.get_patient(patient_ref)
        if not patient:
            raise NotFound("unknown patient_ref; register the patient first")
        try:
            deid = deidentify_image(image_bytes, use_ocr=not self.s.offline)
        except Exception as e:
            raise ValueError(f"could not read image: {e}") from e
        note, phi_counts = scrub_text(clinical_note or "")
        key = self.objects.put(deid.png, "studies", "png")
        summary = {"metadata_removed": deid.metadata_removed, "masked_regions": len(deid.masked_regions),
                   "ocr_used": deid.ocr_used, "note_phi_redactions": phi_counts}
        sid = self.db.create_study(patient_ref=patient_ref, modality=modality, age_band=patient["age_band"],
                                   clinical_note=note, image_key=key, width=deid.width, height=deid.height,
                                   deid_summary=summary, created_by=user["username"])
        self.cases.add(sid, deid.png, modality, patient["age_band"], "unreported")
        self.db.audit(user["username"], "human", "ingest_study", sid, {"modality": modality, "deid": summary})
        return self.db.get_study(sid)

    def study_image(self, sid: str) -> bytes:
        st = self.db.get_study(sid)
        if not st:
            raise NotFound("study not found")
        return self.objects.get(st["image_key"])

    # ---------------------------------------------------------------- agent
    def analyze(self, user: dict, sid: str, question: str, variant: str = "full") -> dict:
        st = self.db.get_study(sid)
        if not st:
            raise NotFound("study not found")
        question, q_phi = scrub_text(question)
        if st["clinical_note"]:
            question = f"{question} Clinical context: {st['clinical_note']}"
        state = self.agent.run(question, modality=st["modality"], age_band=st["age_band"],
                               image_png=self.objects.get(st["image_key"]), study_id=sid,
                               config=AgentConfig.variant(variant, max_iterations=self.s.max_iterations))
        result = state.result()
        result["variant"] = variant
        status = {"verified": ReportStatus.VERIFIED}.get(state.status, ReportStatus.NEEDS_REVIEW)
        rid = self.db.create_report(sid, ReportStatus.DRAFT.value, result, AGENT.id)
        self.db.audit(user["username"], "human", "request_analysis", rid, {"study": sid, "variant": variant,
                                                                          "question_phi_redactions": q_phi})
        new = transition(ReportStatus.DRAFT, status, AGENT)
        self.db.set_report_status(rid, new.value)
        self.db.audit(AGENT.id, AGENT.kind, f"draft_{new.value}", rid,
                      {"iterations": state.iterations,
                       "hallucination_rate": state.verification.hallucination_rate if state.verification else None})
        return self.db.get_report(rid)

    def ask(self, user: dict, question: str, modality: str | None = None) -> dict:
        question, _ = scrub_text(question)
        out = self.agent.ask(question, modality=modality)
        self.db.audit(user["username"], "human", "ask", "assistant", {"modality": modality})
        return out

    # --------------------------------------------------------------- sign-off
    def _actor(self, user: dict) -> Actor:
        return Actor(user["username"], "human", user["role"])

    def sign_off(self, user: dict, rid: str, comment: str = "") -> dict:
        return self._decide(user, rid, ReportStatus.SIGNED_OFF, comment)

    def reject(self, user: dict, rid: str, reason: str) -> dict:
        return self._decide(user, rid, ReportStatus.REJECTED, reason)

    def _decide(self, user: dict, rid: str, target: ReportStatus, comment: str) -> dict:
        rep = self.db.get_report(rid)
        if not rep:
            raise NotFound("report not found")
        comment, _ = scrub_text(comment or "")
        actor = self._actor(user)
        try:
            transition(rep["status"], target, actor, comment)
        except SignOffError as e:
            self.db.audit(actor.id, actor.kind, f"{target.value}_denied", rid, {"error": str(e)})
            raise
        self.db.set_report_status(rid, target.value, signed_by=actor.id, comment=comment or None)
        self.db.audit(actor.id, actor.kind, target.value, rid, {"comment": comment})
        if target == ReportStatus.SIGNED_OFF and rep["study_id"]:
            st = self.db.get_study(rep["study_id"])
            summary = ", ".join(f["concept"] for f in rep["content"].get("findings", []))
            self.cases.add(st["id"], self.objects.get(st["image_key"]), st["modality"], st["age_band"],
                           summary or "signed off")
        return self.db.get_report(rid)

    def agent_attempt_signoff(self, rid: str) -> None:
        """Exists only to demonstrate (and test) that the agent cannot sign off."""
        rep = self.db.get_report(rid)
        try:
            transition(rep["status"], ReportStatus.SIGNED_OFF, AGENT)
        except SignOffError as e:
            self.db.audit(AGENT.id, AGENT.kind, "signed_off_denied", rid, {"error": str(e)})
            raise

    def report_pdf(self, user: dict, rid: str) -> bytes:
        rep = self.db.get_report(rid)
        if not rep:
            raise NotFound("report not found")
        st = self.db.get_study(rep["study_id"]) if rep["study_id"] else {"id": "-"}
        img = self.objects.get(st["image_key"]) if st.get("image_key") else None
        self.db.audit(user["username"], "human", "export_pdf", rid, {"status": rep["status"]})
        return render_report_pdf(rep, st, img)

    # --------------------------------------------------------- knowledge base
    def import_kb_file(self, user: dict, filename: str, data: bytes, title: str, source: str, modality: str) -> dict:
        from .kb.importer import import_bytes
        from .retrieval.chunking import parse_markdown

        path = import_bytes(filename, data, self.uploads_dir, title, source, modality)
        doc = parse_markdown(path.read_text(encoding="utf-8"), path.stem)
        n = self.pipeline.add_documents([doc])
        self.pipeline.save()
        (self.s.index_dir / "kb_fingerprint").write_text(self._kb_fingerprint())
        self.db.audit(user["username"], "human", "kb_import_file", doc.doc_id,
                      {"file": filename[:120], "chunks": n, "words": len(doc.text.split())})
        return {"doc_id": doc.doc_id, "title": doc.title, "chunks": n, "words": len(doc.text.split())}

    def add_kb_document(self, user: dict, title: str, text: str, source: str, modality: list[str]) -> dict:
        text, phi = scrub_text(text)
        doc = Document(doc_id_for(title + text), title, text, source, modality or ["general"])
        n = self.pipeline.add_documents([doc])
        self.pipeline.save()
        # Persist the source so the document survives an index rebuild.
        clean = lambda v: re.sub(r"[\r\n]+", " ", v).strip()  # noqa: E731
        (self.uploads_dir / f"{doc.doc_id}.md").write_text(
            f"---\nid: {doc.doc_id}\ntitle: {clean(title)}\nsource: {clean(source)}\n"
            f"modality: {', '.join(doc.modality)}\n---\n{text}\n", encoding="utf-8")
        (self.s.index_dir / "kb_fingerprint").write_text(self._kb_fingerprint())
        self.db.audit(user["username"], "human", "kb_add_document", doc.doc_id, {"chunks": n, "phi_redactions": phi})
        return {"doc_id": doc.doc_id, "chunks": n, **asdict(doc)}
