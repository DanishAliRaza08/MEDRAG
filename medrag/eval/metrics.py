"""Metric definitions (exactly as specified in the synopsis)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..textutils import content_terms


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def p95(xs) -> float:
    xs = list(xs)
    return float(np.percentile(xs, 95)) if xs else 0.0


def hallucination_rate(supported_flags: list[bool]) -> float:
    """Unsupported claims / total claims (lower is better)."""
    return (sum(1 for s in supported_flags if not s) / len(supported_flags)) if supported_flags else 0.0


def tfidf_cosine(a: str, b: str, corpus: list[str] | None = None) -> float:
    """Offline stand-in for semantic report similarity."""
    docs = [content_terms(t) for t in (corpus or [a, b])]
    n = len(docs)
    df = Counter(t for d in docs for t in set(d))

    def vec(text):
        tf = Counter(content_terms(text))
        return {t: c * math.log(1 + n / (1 + df.get(t, 0))) for t, c in tf.items()}

    va, vb = vec(a), vec(b)
    dot = sum(v * vb.get(t, 0.0) for t, v in va.items())
    na, nb = math.sqrt(sum(v * v for v in va.values())), math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


class SemanticSimilarity:  # pragma: no cover - requires model weights
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model_name)

    def __call__(self, a: str, b: str) -> float:
        va, vb = self.m.encode([a, b], normalize_embeddings=True)
        return float(np.dot(va, vb))
