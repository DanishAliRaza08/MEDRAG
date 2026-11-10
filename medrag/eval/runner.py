"""Run the retrieval and clinical evaluations and write JSON + Markdown results.

Offline-mode numbers only demonstrate that the harness works; they are *not*
clinical results. Real figures require ``MEDRAG_MODE=real`` and credentialed
gold data (see docs/EVALUATION.md).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..agent import AgentConfig, MedRAGAgent
from ..config import ROOT_DIR, Settings, get_settings
from ..imaging import synthetic
from ..retrieval import RetrievalPipeline
from ..retrieval.pipeline import STRATEGIES
from .metrics import SemanticSimilarity, hallucination_rate, mean, p95, recall_at_k, reciprocal_rank, tfidf_cosine

GOLD_DIR = ROOT_DIR / "data" / "gold"
VARIANTS = ("full", "single_pass", "text_only", "no_retrieval")

# Release-blocking thresholds, agreed before the run.
THRESHOLDS = {"retrieval_recall_at_5_min": 0.80, "full_hallucination_rate_max": 0.10}


def evaluate_retrieval(pipeline: RetrievalPipeline, gold: list[dict], ks=(1, 3, 5)) -> dict:
    out = {}
    for strategy in STRATEGIES:
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


def to_markdown(results: dict) -> str:
    lines = [f"# MedRAG evaluation ({results['mode']} mode)", "",
             "> Offline-mode figures exercise the harness with deterministic stand-ins and synthetic images; "
             "they are **not** clinical results." if results["mode"] != "real" else "", "",
             "## Retrieval", "", "| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |",
             "|---|---|---|---|---|---|---|"]
    for k, v in results["retrieval"].items():
        lines.append(f"| {k} | {v['recall@1']:.3f} | {v['recall@3']:.3f} | {v['recall@5']:.3f} | {v['mrr']:.3f} | "
                     f"{v['latency_mean_ms']:.2f} | {v['latency_p95_ms']:.2f} |")
    lines += ["", "## Clinical drafting (ablations)", "",
              "| Variant | Ungrounded-claim rate | Report similarity | Finding accuracy | Verified | Iterations "
              "| Mean ms |",
              "|---|---|---|---|---|---|---|"]
    for k, v in results["clinical"].items():
        lines.append(f"| {k} | {v['hallucination_rate']:.3f} | {v['report_similarity']:.3f} | "
                     f"{v['finding_accuracy']:.3f} | {v['verified_rate']:.3f} | {v['mean_iterations']:.2f} | "
                     f"{v['latency_mean_ms']:.1f} |")
    lines += ["", "## Release gates", ""]
    for k, v in results["gates"].items():
        lines.append(f"- {k}: **{'PASS' if v['pass'] else 'FAIL'}** (value {v['value']}, threshold {v['threshold']})")
    return "\n".join(lines) + "\n"


def run(settings: Settings | None = None, out_dir: Path | None = None) -> dict:
    s = settings or get_settings()
    pipeline = RetrievalPipeline(s)
    pipeline.ingest_directory()
    agent = MedRAGAgent(pipeline, s)
    retrieval_gold = json.loads((GOLD_DIR / "retrieval_gold.json").read_text())
    report_gold = json.loads((GOLD_DIR / "report_gold.json").read_text())
    sim = None
    if not s.offline:  # pragma: no cover
        sim = SemanticSimilarity(os.environ.get("MEDRAG_SIM_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"))
    results = {"mode": s.mode, "retrieval": evaluate_retrieval(pipeline, retrieval_gold),
               "clinical": evaluate_clinical(agent, report_gold, sim)}
    r5 = results["retrieval"]["hybrid+rerank"]["recall@5"]
    hr = results["clinical"]["full"]["hallucination_rate"]
    results["gates"] = {
        "retrieval_recall_at_5": {"value": r5, "threshold": THRESHOLDS["retrieval_recall_at_5_min"],
                                  "pass": r5 >= THRESHOLDS["retrieval_recall_at_5_min"]},
        "full_hallucination_rate": {"value": hr, "threshold": THRESHOLDS["full_hallucination_rate_max"],
                                    "pass": hr <= THRESHOLDS["full_hallucination_rate_max"]},
    }
    out_dir = out_dir or (ROOT_DIR / "eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.md").write_text(to_markdown(results))
    return results
