"""HTTP API and static clinical workspace."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, get_settings
from ..reports.lifecycle import SignOffError
from ..service import MODALITIES, MedRAGService, NotFound
from .auth import PERMISSIONS, TokenError, issue_token, verify_token

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD = 20 * 1024 * 1024


class LoginIn(BaseModel):
    username: str
    password: str


class PatientIn(BaseModel):
    patient_id: str = Field(..., min_length=1, description="Hospital identifier; stored only as an HMAC pseudonym")
    age: int | None = Field(None, ge=0, le=130)
    sex: str = "U"


class AnalyzeIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    variant: str = Field("full", pattern="^(full|no_retrieval|single_pass|text_only)$")


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    modality: str | None = None


class SearchIn(BaseModel):
    query: str = Field(..., min_length=2)
    k: int = Field(5, ge=1, le=50)
    modality: str | None = None
    strategy: str = Field("hybrid+rerank", pattern="^(dense|sparse|hybrid|hybrid\\+rerank)$")


class DecisionIn(BaseModel):
    comment: str = ""


class KBDocIn(BaseModel):
    title: str = Field(..., min_length=3)
    text: str = Field(..., min_length=20)
    source: str = "uploaded"
    modality: list[str] = ["general"]


def create_app(settings: Settings | None = None, service: MedRAGService | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    s = settings or get_settings()
    svc = service or MedRAGService(s)
    app = FastAPI(title="MedRAG", version=__version__,
                  description="Agentic multimodal RAG for evidence-grounded clinical decision support. "
                              "Drafts require clinician sign-off.")
    app.state.service = svc

    def current_user(authorization: str = Header("")) -> dict:
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            data = verify_token(authorization[7:], s.secret_key)
        except (TokenError, ValueError, KeyError) as e:
            raise HTTPException(401, f"invalid token: {e}") from e
        user = svc.db.get_user(data["sub"])
        if not user:
            raise HTTPException(401, "unknown user")
        return user

    def require(perm: str):
        def dep(user: dict = Depends(current_user)) -> dict:
            if user["role"] not in PERMISSIONS[perm]:
                svc.db.audit(user["username"], "human", "permission_denied", perm, {"role": user["role"]})
                raise HTTPException(403, f"role '{user['role']}' lacks permission '{perm}'")
            return user
        return dep

    def nf(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except NotFound as e:
            raise HTTPException(404, str(e)) from e
        except SignOffError as e:
            raise HTTPException(403, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "mode": s.mode, "kb": svc.pipeline.stats()["chunks"],
                "vlm": svc.agent.vlm.name, "generator": svc.agent.generator.name}

    @app.post("/api/auth/login")
    def login(body: LoginIn):
        user = svc.db.authenticate(body.username, body.password)
        if not user:
            svc.db.audit(body.username[:64], "human", "login_failed", "auth", {})
            raise HTTPException(401, "invalid credentials")
        svc.db.audit(user["username"], "human", "login", "auth", {})
        return {"token": issue_token(user["username"], user["role"], s.secret_key, s.token_ttl_seconds),
                "user": {"username": user["username"], "role": user["role"], "full_name": user["full_name"]}}

    @app.get("/api/me")
    def me(user: dict = Depends(current_user)):
        return {"username": user["username"], "role": user["role"], "full_name": user["full_name"],
                "permissions": sorted(p for p, roles in PERMISSIONS.items() if user["role"] in roles)}

    # -------------------------------------------------------------- patients
    @app.post("/api/patients")
    def create_patient(body: PatientIn, user: dict = Depends(require("ingest"))):
        return nf(svc.register_patient, user, body.patient_id, body.age, body.sex)

    @app.get("/api/patients")
    def list_patients(user: dict = Depends(require("read"))):
        return svc.db.list_patients()

    # --------------------------------------------------------------- studies
    @app.post("/api/studies")
    async def create_study(patient_ref: str = Form(...), modality: str = Form(...), clinical_note: str = Form(""),
                           file: UploadFile = File(...), user: dict = Depends(require("ingest"))):
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "image too large")
        return nf(svc.ingest_study, user, patient_ref, modality, data, clinical_note)

    @app.get("/api/studies")
    def list_studies(patient_ref: str | None = None, user: dict = Depends(require("read"))):
        return svc.db.list_studies(patient_ref)

    @app.get("/api/studies/{sid}")
    def get_study(sid: str, user: dict = Depends(require("read"))):
        st = svc.db.get_study(sid)
        if not st:
            raise HTTPException(404, "study not found")
        st.pop("image_key", None)
        st["reports"] = svc.db.list_reports(sid)
        return st

    @app.get("/api/studies/{sid}/image")
    def study_image(sid: str, user: dict = Depends(require("read"))):
        return Response(nf(svc.study_image, sid), media_type="image/png")

    @app.post("/api/studies/{sid}/analyze")
    def analyze(sid: str, body: AnalyzeIn, user: dict = Depends(require("analyze"))):
        return nf(svc.analyze, user, sid, body.question, body.variant)

    # --------------------------------------------------------------- reports
    @app.get("/api/reports")
    def list_reports(study_id: str | None = None, user: dict = Depends(require("read"))):
        return svc.db.list_reports(study_id)

    @app.get("/api/reports/{rid}")
    def get_report(rid: str, user: dict = Depends(require("read"))):
        rep = svc.db.get_report(rid)
        if not rep:
            raise HTTPException(404, "report not found")
        return rep

    @app.post("/api/reports/{rid}/signoff")
    def sign_off(rid: str, body: DecisionIn, user: dict = Depends(require("sign_off"))):
        return nf(svc.sign_off, user, rid, body.comment)

    @app.post("/api/reports/{rid}/reject")
    def reject(rid: str, body: DecisionIn, user: dict = Depends(require("sign_off"))):
        return nf(svc.reject, user, rid, body.comment)

    @app.get("/api/reports/{rid}/pdf")
    def report_pdf(rid: str, user: dict = Depends(require("read"))):
        pdf = nf(svc.report_pdf, user, rid)
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="medrag-{rid}.pdf"'})

    # ------------------------------------------------------- retrieval / chat
    @app.post("/api/search")
    def search(body: SearchIn, user: dict = Depends(require("read"))):
        res = svc.pipeline.search(body.query, k=body.k, modality=body.modality, strategy=body.strategy)
        return {"query": res.query, "strategy": res.strategy, "timings_ms": res.timings_ms,
                "hits": [h.to_dict() for h in res.hits]}

    @app.post("/api/ask")
    def ask(body: AskIn, user: dict = Depends(require("read"))):
        if body.modality and body.modality not in MODALITIES:
            raise HTTPException(400, "unknown modality")
        return svc.ask(user, body.question, body.modality)

    @app.get("/api/kb")
    def kb_stats(user: dict = Depends(require("read"))):
        return svc.pipeline.stats()

    @app.post("/api/kb/documents")
    def kb_add(body: KBDocIn, user: dict = Depends(require("kb_write"))):
        return svc.add_kb_document(user, body.title, body.text, body.source, body.modality)

    @app.get("/api/samples")
    def samples(user: dict = Depends(require("read"))):
        return {"xray": ["normal", "consolidation", "effusion", "cardiomegaly", "pneumothorax"],
                "fundus": ["normal", "diabetic_retinopathy", "glaucoma"], "histopathology": ["benign", "carcinoma"]}

    @app.get("/api/samples/{modality}/{finding}")
    def sample_image(modality: str, finding: str, seed: int = 0, user: dict = Depends(require("read"))):
        """Synthetic demo image with *fake* burned-in identifiers (exercises de-identification)."""
        from ..imaging import synthetic

        if modality not in synthetic.GENERATORS or finding not in samples(user)[modality]:
            raise HTTPException(404, "unknown sample")
        img = synthetic.GENERATORS[modality](finding, seed=seed, burn_text=True)
        return Response(synthetic.to_png_bytes(img, PatientName="John Doe", PatientID="00123456"),
                        media_type="image/png")

    # ----------------------------------------------------------------- audit
    @app.get("/api/audit")
    def audit(limit: int = 200, entity: str | None = None, user: dict = Depends(require("audit"))):
        return svc.db.audit_entries(min(limit, 1000), entity)

    @app.get("/api/audit/verify")
    def audit_verify(user: dict = Depends(require("audit"))):
        return svc.db.verify_audit_chain()

    # ------------------------------------------------------------- frontend
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app
