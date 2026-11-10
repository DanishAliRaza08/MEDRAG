"""Cross-encoder re-rankers.

``LexicalReranker`` is the offline stand-in: it jointly scores the query and
passage by idf-weighted term coverage plus a bonus for matched bigrams, which
imitates the way a cross-encoder rewards passages that answer the whole query.
``CrossEncoderReranker`` loads a real cross-encoder (bge-reranker-v2-m3).
"""

from __future__ import annotations

import math
from typing import Protocol

from ..textutils import content_terms


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class LexicalReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        q = content_terms(query)
        if not q or not passages:
            return [0.0] * len(passages)
        docs = [content_terms(p) for p in passages]
        n = len(docs)
        qset = list(dict.fromkeys(q))
        df = {t: sum(1 for d in docs if t in d) for t in qset}
        qbigrams = set(zip(q, q[1:], strict=False))
        scores = []
        for d in docs:
            dset = set(d)
            weights = {t: math.log(1 + (n + 1) / (df[t] + 0.5)) for t in qset}
            total = sum(weights.values()) or 1.0
            coverage = sum(w for t, w in weights.items() if t in dset) / total
            dbigrams = set(zip(d, d[1:], strict=False))
            bigram = len(qbigrams & dbigrams) / len(qbigrams) if qbigrams else 0.0
            density = sum(1 for t in d if t in weights) / (len(d) + 20)
            scores.append(coverage + 0.5 * bigram + 0.5 * density)
        return scores


class CrossEncoderReranker:  # pragma: no cover - requires model weights
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def score(self, query, passages):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]
