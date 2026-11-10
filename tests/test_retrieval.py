import numpy as np
import pytest

from medrag.retrieval.chunking import Document, chunk_text, parse_markdown
from medrag.retrieval.embeddings import HashingEmbedder
from medrag.retrieval.fusion import reciprocal_rank_fusion
from medrag.retrieval.rerank import LexicalReranker
from medrag.retrieval.sparse import BM25Index
from medrag.retrieval.vector_store import InMemoryVectorStore, PayloadNotAllowed
from medrag.textutils import overlap_ratio, split_sentences, stem


def test_stem_and_overlap():
    assert stem("opacities") == stem("opacity")
    assert overlap_ratio("pleural effusion", "A pleural effusion is fluid") == 1.0
    assert overlap_ratio("pleural effusion", "pneumonia") == 0.0


def test_split_sentences():
    assert split_sentences("One here. Two there! Three?") == ["One here.", "Two there!", "Three?"]


def test_chunking_respects_sentences_and_size():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(200))
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 10
    assert all(c.endswith(".") for c in chunks)
    assert all(len(c.split()) <= 50 for c in chunks)


def test_chunking_overlap_repeats_trailing_sentence():
    text = " ".join(f"Fact {i} is about lungs." for i in range(30))
    a, b = chunk_text(text, max_tokens=20, overlap_tokens=6)[:2]
    assert a.split(". ")[-1].rstrip(".") in b


def test_parse_markdown_front_matter():
    d = parse_markdown("---\nid: x1\ntitle: T\nmodality: xray, fundus\n---\n# Head\nBody text here", "fb")
    assert (d.doc_id, d.title, d.modality) == ("x1", "T", ["xray", "fundus"])
    assert "Body text here." in d.text


def test_hashing_embedder_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed(["pleural effusion"]), e.embed(["pleural effusion"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    sim_related = float(e.embed(["pleural effusions"])[0] @ a[0])
    sim_unrelated = float(e.embed(["glaucoma cup disc"])[0] @ a[0])
    assert sim_related > sim_unrelated


def test_bm25_ranks_exact_terms_first():
    idx = BM25Index()
    idx.add("a", "Xpert MTB/RIF testing for tuberculosis")
    idx.add("b", "pneumonia antibiotics")
    assert idx.search("xpert tuberculosis")[0][0] == "a"
    idx.remove("a")
    assert idx.search("xpert") == []


def test_rrf_fusion():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60))
    assert max(fused, key=fused.get) == "b"


def test_reranker_prefers_full_query_coverage():
    s = LexicalReranker().score("pleural effusion aspiration",
                                ["pleural effusion aspiration is done", "pleural plaque"])
    assert s[0] > s[1]


def test_vector_store_rejects_non_allowlisted_payload():
    vs = InMemoryVectorStore(4)
    with pytest.raises(PayloadNotAllowed):
        vs.upsert("x", np.ones(4), {"text": "t", "patient_name": "John Doe"})
    vs.upsert("x", np.ones(4) / 2, {"text": "t"})
    assert len(vs) == 1


def test_vector_store_dimension_check():
    with pytest.raises(ValueError):
        InMemoryVectorStore(4).upsert("x", np.ones(3), {})


@pytest.mark.parametrize("query,doc", [
    ("CURB-65 severity score", "kb-cap"),
    ("blunting of the costophrenic angle", "kb-effusion"),
    ("cup-to-disc ratio glaucoma", "kb-glaucoma"),
    ("Nottingham grading breast", "kb-breast-ca"),
])
def test_hybrid_search_top1(pipeline, query, doc):
    assert pipeline.search(query).hits[0].doc_id == doc


def test_modality_filter(pipeline):
    hits = pipeline.search("screening referral", modality="fundus", k=10).hits
    assert hits and all("fundus" in h.modality or "general" in h.modality for h in hits)


@pytest.mark.parametrize("strategy", ["dense", "sparse", "hybrid", "hybrid+rerank"])
def test_all_strategies_return_ranked_hits(pipeline, strategy):
    res = pipeline.search("pneumothorax chest drain", strategy=strategy)
    assert res.hits and [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    assert res.timings_ms["total_ms"] >= 0


def test_unknown_strategy(pipeline):
    with pytest.raises(ValueError):
        pipeline.search("x", strategy="magic")


def test_pipeline_persistence_and_reingest(tmp_path, settings):
    from medrag.retrieval import RetrievalPipeline

    p = RetrievalPipeline(settings)
    p.add_documents([Document("d1", "Doc", "Alpha beta gamma delta. Epsilon zeta eta theta.")])
    p.add_documents([Document("d1", "Doc", "Completely new content about sarcoidosis.")])
    assert len(p.chunks) == 1 and p.search("sarcoidosis").hits[0].doc_id == "d1"
    p.save(tmp_path / "idx")
    q = RetrievalPipeline(settings)
    assert q.load(tmp_path / "idx") and q.search("sarcoidosis").hits[0].doc_id == "d1"
