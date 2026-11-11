"""Run the retrieval and clinical evaluations and write JSON + Markdown results.

Offline-mode numbers only demonstrate that the harness works; they are *not*
clinical results. Real figures require ``MEDRAG_MODE=real`` and credentialed
gold data (see docs/EVALUATION.md).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

from ..agent import AgentConfig, MedRAGAgent
from ..config import ROOT_DIR, Settings, get_settings
from ..imaging import synthetic
from ..retrieval import RetrievalPipeline
from ..retrieval.pipeline import STRATEGIES, load_or_build
from .metrics import SemanticSimilarity, hallucination_rate, mean, p95, recall_at_k, reciprocal_rank, tfidf_cosine

GOLD_DIR = ROOT_DIR / "data" / "gold"
VARIANTS = ("full", "single_pass", "text_only", "no_retrieval")

# Release-blocking thresholds, agreed before the run.
THRESHOLDS = {"retrieval_recall_at_5_min": 0.80, "full_hallucination_rate_max": 0.10}


def evaluate_retrieval(pipeline: RetrievalPipeline, gold: list[dict], ks=(1, 3, 5), strategies=STRATEGIES) -> dict:
    out = {}
    for strategy in strategies:
        recalls = {k: [] for k in ks}
        rrs, lat = [], []
        for item in gold:
            res = pipeline.search(item["query"], k=max(ks) * 2, strategy=strategy)
            docs = list(dict.fromkeys(h.doc_id for h in res.hits))  # doc-level relevance
            rel = set(item["relevant"])
            for k in ks:
                recalls[k].append(recall_at_k(docs, rel, k))
            rrs.append(reciprocal_rank(docs, rel))
            lat.append(res.timings_ms["total_ms"])
        out[strategy] = {**{f"recall@{k}": round(mean(v), 4) for k, v in recalls.items()},
                         "mrr": round(mean(rrs), 4), "latency_mean_ms": round(mean(lat), 3),
                         "latency_p95_ms": round(p95(lat), 3), "queries": len(gold)}
    return out


def _report_text(state) -> str:
    return " ".join(c.text for c in state.draft.claims) if state.draft else ""


def evaluate_clinical(agent: MedRAGAgent, cases: list[dict], similarity=None) -> dict:
    similarity = similarity or tfidf_cosine
    out = {}
    for variant in VARIANTS:
        rates, sims, accs, lat, iters, verified = [], [], [], [], [], []
        for case in cases:
            img = synthetic.GENERATORS[case["modality"]](case["finding"], seed=case.get("seed", 0))
            t = time.perf_counter()
            st = agent.run(case["question"], modality=case["modality"], image_png=synthetic.to_png_bytes(img),
                           config=AgentConfig.variant(variant))
            lat.append((time.perf_counter() - t) * 1000)
            text = _report_text(st)
            flags = [bool(c.supported) for c in st.draft.claims if c.source != "image"]
            rates.append(hallucination_rate(flags))
            sims.append(similarity(text, case["reference"]))
            accs.append(1.0 if case["expected_concept"].lower() in text.lower() else 0.0)
            iters.append(st.iterations)
            verified.append(1.0 if st.status == "verified" else 0.0)
        out[variant] = {"hallucination_rate": round(mean(rates), 4), "report_similarity": round(mean(sims), 4),
                        "finding_accuracy": round(mean(accs), 4), "verified_rate": round(mean(verified), 4),
                        "mean_iterations": round(mean(iters), 3), "latency_mean_ms": round(mean(lat), 2),
                        "latency_p95_ms": round(p95(lat), 2), "cases": len(cases)}
    return out


def verifier_id(agent: MedRAGAgent) -> str:
    return f"{type(agent.verifier).__name__}/v{getattr(agent.verifier, 'RULES_VERSION', 1)}"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_real_xray(agent: MedRAGAgent, data_dir: Path, n: int = 20, variants=VARIANTS, similarity=None,
                       log=print, checkpoint: Path | None = None) -> dict:
    """Clinical evaluation on real IU X-ray studies with radiologist reports (see medrag/datasets.py).

    Clinical efficacy = finding labels extracted from the generated report vs. labels from the radiologist's report
    (same rule-based labeller for both), reported as micro precision / recall / F1 over abnormal findings plus
    accuracy on the normal-vs-abnormal decision.
    """
    from ..datasets import CONCEPTS, image_bytes, is_normal, label_report, load_cases

    similarity = similarity or tfidf_cosine
    cases = load_cases(data_dir, n)
    # Checkpointing: long CPU/GPU runs can be interrupted, so every finished variant is saved and skipped on re-run
    # (only if it was produced with the same cases and the same models).
    key = {"cases": [c["id"] for c in cases], "vlm": agent.vlm.name, "generator": agent.generator.name,
           "verifier": verifier_id(agent)}
    out: dict = {}
    if checkpoint and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("key") == key:
            out = saved["results"]
            log(f"  resuming: variants already done: {list(out)}")
    for variant in variants:
        if variant in out:
            continue
        tp = fp = fn = errors = 0
        normal_ok, rates, sims, lat, iters = [], [], [], [], []
        per_case = []
        for i, case in enumerate(cases, start=1):
            t = time.perf_counter()
            try:
                st = agent.run("Please review this chest radiograph and report the findings.", modality="xray",
                               image_png=image_bytes(data_dir, case), config=AgentConfig.variant(variant))
            except Exception as e:  # pragma: no cover - infrastructure failure outside the agent
                errors += 1
                log(f"  [{variant}] {i}/{len(cases)} {case['id']} ERROR {e}")
                continue
            if st.errors:
                errors += 1
            lat.append((time.perf_counter() - t) * 1000)
            gen_text = _report_text(st)
            pred = label_report(" ".join(c.text for c in st.draft.claims if c.section in ("findings", "impression")))
            gold = case["labels"]
            for c in CONCEPTS:
                tp += int(pred[c] and gold[c])
                fp += int(pred[c] and not gold[c])
                fn += int(gold[c] and not pred[c])
            normal_ok.append(float(is_normal(pred) == is_normal(gold)))
            flags = [bool(c.supported) for c in st.draft.claims if c.source != "image"]
            rates.append(hallucination_rate(flags))
            sims.append(similarity(gen_text, case["reference"]))
            iters.append(st.iterations)
            draft = "\n".join(f"[{c.section}] {c.text} " + "".join(f"[{i}]" for i in c.citations)
                              + ("" if c.supported is not False else " (UNSUPPORTED)") for c in st.draft.claims)
            per_case.append({"id": case["id"], "errors": st.errors, "gold": [k for k, v in gold.items() if v],
                             "pred": [k for k, v in pred.items() if v], "status": st.status,
                             "iterations": st.iterations, "findings": [f.text for f in st.findings][:3],
                             "reference": case["reference"], "draft": draft,
                             "sources": sorted({e["title"] for e in st.evidence})})
            log(f"  [{variant}] {i}/{len(cases)} {case['id']} gold={per_case[-1]['gold']} pred={per_case[-1]['pred']}"
                f" {lat[-1] / 1000:.0f}s")
        p, r, f1 = _prf(tp, fp, fn)
        out[variant] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                        "normal_abnormal_accuracy": round(mean(normal_ok), 4),
                        "hallucination_rate": round(mean(rates), 4), "report_similarity": round(mean(sims), 4),
                        "mean_iterations": round(mean(iters), 3), "latency_mean_ms": round(mean(lat), 1),
                        "latency_p95_ms": round(p95(lat), 1), "cases": len(cases), "errors": errors,
                        "per_case": per_case}
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"key": key, "results": out}, indent=1))
    return {v: out[v] for v in variants if v in out}


REVIEW_COLUMNS = ["accuracy_1to5", "grounding_1to5", "clinical_usefulness_1to5", "harmful_error_yes_no", "comments"]


def write_review_pack(results: dict, path: Path, variant: str = "full") -> Path | None:
    """CSV for blinded clinician review of real-data drafts (phase 7 of the plan): rate, then send back."""
    import csv

    cases = (results.get("real_xray") or {}).get(variant, {}).get("per_case", [])
    if not cases:
        return None
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "image_file", "radiologist_report", "medrag_draft", "agent_status", "sources",
                    *REVIEW_COLUMNS])
        for c in cases:
            w.writerow([c["id"], f"{c['id']}.png", c["reference"], c["draft"], c["status"], "; ".join(c["sources"]),
                        *[""] * len(REVIEW_COLUMNS)])
    return path


def to_markdown(results: dict) -> str:
    lines = [f"# MedRAG evaluation ({results['mode']} mode)", "",
             "> Offline-mode figures exercise the harness with deterministic stand-ins and synthetic images; "
             "they are **not** clinical results." if results["mode"] != "real" else "", "",
             f"Corpus: {results['corpus']['documents']} documents, {results['corpus']['chunks']} chunks "
             f"({results['corpus']['embedder']} + {results['corpus']['reranker']}).", ""]
    names = {"sample": "Retrieval: sample corpus (26 queries, complete labels)",
             "pmc": "Retrieval: full corpus incl. PubMed Central (31 developer-annotated queries)"}
    for set_name, table in results["retrieval"].items():
        lines += [f"## {names.get(set_name, set_name)}", "",
                  "| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in table.items():
            lines.append(f"| {k} | {v['recall@1']:.3f} | {v['recall@3']:.3f} | {v['recall@5']:.3f} | {v['mrr']:.3f} | "
                         f"{v['latency_mean_ms']:.2f} | {v['latency_p95_ms']:.2f} |")
        lines.append("")
    if results.get("models"):
        m = results["models"]
        lines += ["", f"Models: embedder `{m['embedder']}`, re-ranker `{m['reranker']}`, VLM `{m['vlm']}`, generator "
                      f"`{m['generator']}`, verifier `{m['verifier']}`."]
    if results.get("clinical"):
        lines += ["", "## Clinical drafting (ablations)", "",
                  "| Variant | Ungrounded-claim rate | Report similarity | Finding accuracy | Verified | Iterations "
                  "| Mean ms |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in results["clinical"].items():
            lines.append(f"| {k} | {v['hallucination_rate']:.3f} | {v['report_similarity']:.3f} | "
                         f"{v['finding_accuracy']:.3f} | {v['verified_rate']:.3f} | {v['mean_iterations']:.2f} | "
                         f"{v['latency_mean_ms']:.1f} |")
    if results.get("real_xray"):
        lines += ["", f"## Real chest X-rays: IU X-ray ({results['mode']} mode, "
                      f"{next(iter(results['real_xray'].values()))['cases']} studies with radiologist reports)", "",
                  "| Variant | Precision | Recall | F1 | Normal/abnormal acc. | Ungrounded rate | Similarity | "
                  "Iterations | Mean s |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in results["real_xray"].items():
            lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                         f"{v['normal_abnormal_accuracy']:.3f} | {v['hallucination_rate']:.3f} | "
                         f"{v['report_similarity']:.3f} | {v['mean_iterations']:.2f} | "
                         f"{v['latency_mean_ms'] / 1000:.1f} |")
    lines += ["", "## Release gates", ""]
    for k, v in results["gates"].items():
        lines.append(f"- {k}: **{'PASS' if v['pass'] else 'FAIL'}** (value {v['value']}, threshold {v['threshold']})")
    return "\n".join(lines) + "\n"


def run(settings: Settings | None = None, out_dir: Path | None = None, real_xray: Path | None = None,
        real_n: int = 20, real_variants=VARIANTS, synthetic: bool = True, retrieval_strategies=STRATEGIES) -> dict:
    s = settings or get_settings()
    pipeline = load_or_build(s)
    agent = MedRAGAgent(pipeline, s)
    report_gold = json.loads((GOLD_DIR / "report_gold.json").read_text())
    retrieval = {}
    # (a) controlled set: sample corpus only, so relevance labels are complete
    sample_kb = ROOT_DIR / "data" / "knowledge_base"
    sample = RetrievalPipeline(dataclasses.replace(s, kb_dirs=[sample_kb]), embedder=pipeline.embedder,
                               reranker=pipeline.reranker)
    sample.ingest_directory()
    retrieval["sample"] = evaluate_retrieval(sample, json.loads((GOLD_DIR / "retrieval_gold.json").read_text()),
                                             strategies=retrieval_strategies)
    # (b) the full configured corpus, against the PubMed Central question set
    pmc_gold = GOLD_DIR / "retrieval_gold_pmc.json"
    pmc_docs = {c.doc_id for c in pipeline.chunks.values() if c.doc_id.startswith("pmc-")}
    if pmc_gold.exists() and pmc_docs:
        items = [i for i in json.loads(pmc_gold.read_text())["items"] if set(i["relevant"]) <= pmc_docs]
        retrieval["pmc"] = evaluate_retrieval(pipeline, items, strategies=retrieval_strategies)
    sim = None
    if not s.offline:  # pragma: no cover
        sim = SemanticSimilarity(os.environ.get("MEDRAG_SIM_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"))
    results = {"mode": s.mode, "retrieval": retrieval,
               "clinical": evaluate_clinical(agent, report_gold, sim) if synthetic else {},
               "corpus": pipeline.stats() | {"titles": None},
               "models": {"embedder": s.embed_model if not s.offline else "hashing (offline)",
                          "reranker": s.rerank_model if not s.offline else "lexical (offline)",
                          "vlm": agent.vlm.name, "generator": agent.generator.name,
                          "verifier": verifier_id(agent)}}
    if real_xray is not None and (real_xray / "cases.jsonl").exists():
        results["real_xray"] = evaluate_real_xray(agent, real_xray, real_n, real_variants, sim,
                                                  checkpoint=(out_dir or ROOT_DIR / "eval_results")
                                                  / "real_xray_checkpoint.json")
    gate_set = "pmc" if "pmc" in retrieval else "sample"
    r5 = results["retrieval"][gate_set]["hybrid+rerank"]["recall@5"]
    clinical_full = results["clinical"].get("full") or (results.get("real_xray") or {}).get("full") or {}
    hr = clinical_full.get("hallucination_rate", 0.0)
    results["gates"] = {
        f"retrieval_recall_at_5 ({gate_set} set)": {"value": r5, "threshold": THRESHOLDS["retrieval_recall_at_5_min"],
                                  "pass": r5 >= THRESHOLDS["retrieval_recall_at_5_min"]},
        "full_hallucination_rate": {"value": hr, "threshold": THRESHOLDS["full_hallucination_rate_max"],
                                    "pass": hr <= THRESHOLDS["full_hallucination_rate_max"]},
    }
    out_dir = out_dir or (ROOT_DIR / "eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.md").write_text(to_markdown(results))
    write_review_pack(results, out_dir / "clinician_review_sheet.csv")
    return results
