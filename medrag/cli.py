"""Command-line interface: ``medrag serve | ingest | eval | samples | demo``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT_DIR, get_settings


def cmd_serve(a):
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=a.host, port=a.port)


def cmd_llm_server(a):
    import uvicorn

    from .llm_server import create_llm_app

    uvicorn.run(create_llm_app(a.model, max_pixels=a.max_pixels * 28 * 28, threads=a.threads), host=a.host, port=a.port)


def cmd_ingest(a):
    from .retrieval import RetrievalPipeline

    s = get_settings()
    p = RetrievalPipeline(s)
    n = p.ingest_directory(Path(a.path) if a.path else None)
    p.save()
    print(json.dumps({"chunks": n, **p.stats()}, indent=2))


def cmd_eval(a):
    from .eval.runner import run, to_markdown

    real = Path(a.real_xray) if a.real_xray else None
    variants = tuple(a.variants.split(",")) if a.variants else None
    res = run(out_dir=Path(a.out) if a.out else None, real_xray=real, real_n=a.n,
              **({"real_variants": variants} if variants else {}))
    print(to_markdown(res))
    if a.strict and not all(g["pass"] for g in res["gates"].values()):
        print("Release gate FAILED", file=sys.stderr)
        sys.exit(1)


def cmd_kb(a):
    from .kb import pmc
    from .kb.importer import import_file

    if a.kb_cmd == "fetch-pmc":
        out = Path(a.out)
        m = pmc.build(out, per_topic=a.per_topic) if a.discover else pmc.build_pinned(out)
        print(f"{len(m)} CC BY articles written to {out} (see ATTRIBUTION.md); restart the server to re-index")
    elif a.kb_cmd == "import":
        path = import_file(Path(a.file), Path(a.out), title=a.title, source=a.source, modality=a.modality)
        print(f"wrote {path}; restart the server (or run 'medrag ingest') to index it")


def cmd_data(a):
    from .datasets import prepare_iu_xray

    prepare_iu_xray(Path(a.out), n=a.n)


def cmd_samples(a):
    from .imaging import synthetic

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [("xray", f) for f in ("normal", "consolidation", "effusion", "cardiomegaly", "pneumothorax")]
    cases += [("fundus", f) for f in ("normal", "diabetic_retinopathy", "glaucoma")]
    cases += [("histopathology", f) for f in ("benign", "carcinoma")]
    for mod, f in cases:
        img = synthetic.GENERATORS[mod](f, seed=7, burn_text=True)
        path = out / f"{mod}_{f}.png"
        path.write_bytes(synthetic.to_png_bytes(img, PatientName="John Doe", PatientID="00123456"))
        print(path)


def cmd_demo(a):
    """Run one full end-to-end case in the terminal (no server needed)."""
    from .imaging import synthetic
    from .reports.lifecycle import SignOffError
    from .service import MedRAGService

    svc = MedRAGService()
    clin = svc.db.get_user("clinician")
    pt = svc.register_patient(clin, "MRN-DEMO-001", 67, "M")
    png = synthetic.to_png_bytes(synthetic.chest_xray("effusion", seed=3, burn_text=True), PatientName="John Doe")
    st = svc.ingest_study(clin, pt["patient_ref"], "xray", png, "Pt John Doe, MRN 00123456, breathless for 2 weeks.")
    print("Study", st["id"], "de-identification:", json.dumps(st["deid_summary"]))
    print("Stored clinical note:", st["clinical_note"])
    rep = svc.analyze(clin, st["id"], "Progressive breathlessness - please review the chest radiograph.")
    c = rep["content"]
    print(f"\nReport {rep['id']}  status={rep['status']}  iterations={c['iterations']}")
    for cl in c["draft"]["claims"]:
        cites = "".join(f"[{i}]" for i in cl["citations"])
        print(f"  [{cl['section']}] {cl['text']} {cites}")
    try:
        svc.agent_attempt_signoff(rep["id"])
    except SignOffError as e:
        print("\nAgent sign-off attempt refused:", e)
    rep = svc.sign_off(clin, rep["id"], "Agree with draft.")
    print("Clinician sign-off ->", rep["status"], "by", rep["signed_by"])
    out = ROOT_DIR / "var" / f"{rep['id']}.pdf"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(svc.report_pdf(clin, rep["id"]))
    print("PDF written to", out)
    print("Audit chain:", svc.db.verify_audit_chain())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="medrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="run the API + clinical workspace")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("llm-server", help="serve a vision-language model with an OpenAI-compatible API (no GPU needed)")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-pixels", type=int, default=256, help="image budget in 28x28 patches (speed vs detail)")
    p.add_argument("--threads", type=int, default=None)
    p.set_defaults(fn=cmd_llm_server)
    p = sub.add_parser("ingest", help="(re)build the knowledge-base index")
    p.add_argument("--path")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("eval", help="run retrieval + clinical evaluation with ablations")
    p.add_argument("--out")
    p.add_argument("--strict", action="store_true", help="exit non-zero if a release gate fails")
    p.add_argument("--real-xray", help="folder prepared by 'medrag data iu-xray' to also evaluate on real X-rays")
    p.add_argument("--n", type=int, default=20, help="number of real studies to evaluate")
    p.add_argument("--variants", help="comma-separated subset of: full,single_pass,text_only,no_retrieval")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("kb", help="build the knowledge base")
    kb = p.add_subparsers(dest="kb_cmd", required=True)
    q = kb.add_parser("fetch-pmc", help="download the curated CC BY PubMed Central articles")
    q.add_argument("--out", default=str(ROOT_DIR / "data" / "knowledge_base_pmc"))
    q.add_argument("--discover", action="store_true", help="search per topic instead of using the curated list")
    q.add_argument("--per-topic", type=int, default=3)
    q = kb.add_parser("import", help="import a guideline document (PDF, TXT or Markdown)")
    q.add_argument("file")
    q.add_argument("--title", required=True)
    q.add_argument("--source", default="imported document")
    q.add_argument("--modality", default="general", help="xray, fundus, histopathology or general")
    q.add_argument("--out", default=str(ROOT_DIR / "data" / "knowledge_base_local"))
    p.set_defaults(fn=cmd_kb)
    p = sub.add_parser("data", help="prepare evaluation datasets")
    p.add_argument("dataset", choices=["iu-xray"])
    p.add_argument("--out", default=str(ROOT_DIR / "var" / "datasets" / "iu_xray"))
    p.add_argument("--n", type=int, default=150)
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("samples", help="write synthetic sample images (with fake burned-in PHI)")
    p.add_argument("--out", default=str(ROOT_DIR / "data" / "samples"))
    p.set_defaults(fn=cmd_samples)
    p = sub.add_parser("demo", help="run an end-to-end case in the terminal")
    p.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
