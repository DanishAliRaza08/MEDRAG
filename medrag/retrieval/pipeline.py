"""The hybrid retrieval pipeline.

query -> metadata pre-filter -> dense top-N  \
                                              > RRF fusion -> cross-encoder re-rank -> top-K
query -> metadata pre-filter -> sparse top-N /

Dense and sparse indexes are always written together so they never drift.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from .chunking import Chunk, Document, chunk_document, load_directory
from .embeddings import BGEM3Embedder, HashingEmbedder
from .fusion import reciprocal_rank_fusion
from .rerank import CrossEncoderReranker, LexicalReranker
from .sparse import BM25Index
from .vector_store import InMemoryVectorStore, QdrantVectorStore


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    modality: list[str]
    score: float
    rank: int
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    strategy: str = "hybrid+rerank"


STRATEGIES = ("dense", "sparse", "hybrid", "hybrid+rerank")


class RetrievalPipeline:
    def __init__(self, settings: Settings | None = None, embedder=None, reranker=None, vector_store=None):
        self.settings = settings or get_settings()
        s = self.settings
        if embedder is None:
            embedder = HashingEmbedder() if s.offline else BGEM3Embedder(s.embed_model)
        if reranker is None:
            reranker = LexicalReranker() if s.offline else CrossEncoderReranker(s.rerank_model)
        self.embedder = embedder
        self.reranker = reranker
        if vector_store is None:
            if s.vector_backend == "qdrant":  # pragma: no cover
                vector_store = QdrantVectorStore(embedder.dim, s.qdrant_url)
            else:
                vector_store = InMemoryVectorStore(embedder.dim)
        self.vectors = vector_store
        self.sparse = BM25Index()
        self.chunks: dict[str, Chunk] = {}

    # ------------------------------------------------------------------ ingest
    def add_documents(self, docs: list[Document]) -> int:
        new_chunks: list[Chunk] = []
        for doc in docs:
            for old in [c for c in self.chunks if self.chunks[c].doc_id == doc.doc_id]:
                del self.chunks[old]
                self.sparse.remove(old)
            new_chunks += chunk_document(doc, self.settings.chunk_tokens, self.settings.chunk_overlap)
        if not new_chunks:
            return 0
        vecs = self.embedder.embed([f"{c.title}. {c.text}" for c in new_chunks])
        for c, v in zip(new_chunks, vecs, strict=True):
            self.vectors.upsert(c.chunk_id, v, c.payload())
            self.sparse.add(c.chunk_id, f"{c.title}. {c.text}")
            self.chunks[c.chunk_id] = c
        return len(new_chunks)

    def ingest_directory(self, path: Path | None = None) -> int:
        return self.add_documents(load_directory(path or self.settings.kb_dir))

    def stats(self) -> dict:
        docs = {c.doc_id for c in self.chunks.values()}
        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
            "titles": sorted({c.title for c in self.chunks.values()}),
            "embedder": type(self.embedder).__name__,
            "reranker": type(self.reranker).__name__,
        }

    # ------------------------------------------------------------------ search
    def _allowed(self, modality: str | None, age_band: str | None) -> set[str] | None:
        if not modality and not age_band:
            return None
        out = set()
        for cid, c in self.chunks.items():
            if modality and modality not in c.modality and "general" not in c.modality:
                continue
            if age_band and c.age_band not in ("any", age_band):
                continue
            out.add(cid)
        return out

    def search(
        self,
        query: str,
        k: int | None = None,
        modality: str | None = None,
        age_band: str | None = None,
        strategy: str = "hybrid+rerank",
    ) -> SearchResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        k = k or self.settings.top_k
        n = max(self.settings.candidate_k, k)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        allowed = self._allowed(modality, age_band)

        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []
        if strategy != "sparse":
            t = time.perf_counter()
            qv = self.embedder.embed([query])[0]
            dense = self.vectors.search(qv, n, allowed)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
        if strategy != "dense":
            t = time.perf_counter()
            sparse = self.sparse.search(query, n, allowed)
            timings["sparse_ms"] = (time.perf_counter() - t) * 1000

        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        sparse_rank = {cid: i + 1 for i, (cid, _) in enumerate(sparse)}
        if strategy == "dense":
            ranked = dense
        elif strategy == "sparse":
            ranked = sparse
        else:
            t = time.perf_counter()
            ranked = reciprocal_rank_fusion([[c for c, _ in dense], [c for c, _ in sparse]], self.settings.rrf_k)
            timings["fusion_ms"] = (time.perf_counter() - t) * 1000
            if strategy == "hybrid+rerank" and ranked:
                t = time.perf_counter()
                cands = [cid for cid, _ in ranked[:n]]
                scores = self.reranker.score(query, [self.chunks[c].text for c in cands])
                ranked = sorted(zip(cands, scores, strict=True), key=lambda x: -x[1])
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        hits = []
        for i, (cid, score) in enumerate(ranked[:k]):
            c = self.chunks[cid]
            hits.append(
                SearchHit(cid, c.doc_id, c.title, c.text, c.source, c.modality, float(score), i + 1,
                          dense_rank.get(cid), sparse_rank.get(cid))
            )
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return SearchResult(query, hits, timings, strategy)

    # ------------------------------------------------------------- persistence
    def save(self, path: Path | None = None) -> None:
        path = path or self.settings.index_dir
        path.mkdir(parents=True, exist_ok=True)
        self.vectors.save(path / "dense")
        (path / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks.values()]))

    def load(self, path: Path | None = None) -> bool:
        path = path or self.settings.index_dir
        f = path / "chunks.json"
        if not f.exists():
            return False
        chunks = [Chunk(**d) for d in json.loads(f.read_text())]
        if isinstance(self.vectors, InMemoryVectorStore):
            self.vectors = InMemoryVectorStore.load(path / "dense", self.embedder.dim)
            if len(self.vectors) != len(chunks):
                return False
        for c in chunks:
            self.chunks[c.chunk_id] = c
            self.sparse.add(c.chunk_id, f"{c.title}. {c.text}")
        return True
