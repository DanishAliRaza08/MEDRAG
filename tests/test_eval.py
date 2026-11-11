import json

from medrag.eval.metrics import hallucination_rate, p95, recall_at_k, reciprocal_rank, tfidf_cosine
from medrag.eval.runner import GOLD_DIR, evaluate_retrieval, run


def test_metric_definitions():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3 and reciprocal_rank(["x"], {"a"}) == 0
    assert hallucination_rate([True, False, True, True]) == 0.25 and hallucination_rate([]) == 0
    assert p95(list(range(101))) == 95
    assert tfidf_cosine("pleural effusion", "pleural effusion") > 0.99 > tfidf_cosine("pleural effusion", "glaucoma")


def test_gold_sets_are_disjoint_from_kb():
    gold = json.loads((GOLD_DIR / "report_gold.json").read_text())
    kb = " ".join(p.read_text() for p in (GOLD_DIR.parent / "knowledge_base").glob("*.md"))
    assert all(case["reference"] not in kb for case in gold)


def test_retrieval_eval(pipeline):
    gold = json.loads((GOLD_DIR / "retrieval_gold.json").read_text())
    res = evaluate_retrieval(pipeline, gold)
    assert set(res) == {"dense", "sparse", "hybrid", "hybrid+rerank"}
    assert res["hybrid+rerank"]["recall@5"] >= 0.8


def test_full_eval_run_and_ablation_ordering(tmp_path, settings):
    res = run(settings, tmp_path)
    c = res["clinical"]
    assert c["full"]["hallucination_rate"] < c["no_retrieval"]["hallucination_rate"]
    assert c["full"]["finding_accuracy"] > c["text_only"]["finding_accuracy"]
    assert all(g["pass"] for g in res["gates"].values())
    assert (tmp_path / "results.md").exists() and (tmp_path / "results.json").exists()


def test_real_xray_eval_checkpoints_and_resumes(tmp_path, pipeline):
    """A finished variant is saved and not recomputed when the run is restarted."""
    import json as _json

    from medrag.agent import MedRAGAgent
    from medrag.datasets import label_report
    from medrag.eval.runner import evaluate_real_xray
    from medrag.imaging import synthetic

    data = tmp_path / "data"
    data.mkdir()
    cases = []
    for i, (finding, ref) in enumerate([("effusion", "Left pleural effusion."), ("normal", "Lungs are clear.")]):
        (data / f"c{i}.png").write_bytes(synthetic.to_png_bytes(synthetic.chest_xray(finding, seed=i)))
        cases.append({"id": f"c{i}", "image": f"c{i}.png", "reference": ref, "labels": label_report(ref)})
    (data / "cases.jsonl").write_text("\n".join(_json.dumps(c) for c in cases))
    agent = MedRAGAgent(pipeline)
    ck = tmp_path / "ck.json"
    first = evaluate_real_xray(agent, data, 2, ("full",), checkpoint=ck, log=lambda *a: None)
    assert first["full"]["cases"] == 2 and first["full"]["recall"] == 1.0  # effusion found, normal kept normal
    calls = []
    agent.run = lambda *a, **k: calls.append(1)  # would crash if the finished variant were recomputed
    again = evaluate_real_xray(agent, data, 2, ("full",), checkpoint=ck, log=lambda *a: None)
    assert again == first and calls == []
