"""Text embedders.

``HashingEmbedder`` is the deterministic offline stand-in: it hashes stemmed
unigrams, bigrams and character trigrams into a fixed-size, L2-normalised
vector. ``BGEM3Embedder`` loads BAAI/bge-m3 lazily on first use (real mode).
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from ..textutils import content_terms


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _h(s: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(s.encode(), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % dim, 1.0 if (v >> 63) & 1 else -1.0


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        terms = content_terms(text)
        feats: list[tuple[str, float]] = [(t, 1.0) for t in terms]
        feats += [(a + "_" + b, 0.7) for a, b in zip(terms, terms[1:], strict=False)]
        for t in terms:
            padded = f"#{t}#"
            feats += [("c3:" + padded[i : i + 3], 0.25) for i in range(len(padded) - 2)]
        for f, w in feats:
            idx, sign = _h(f, self.dim)
            vec[idx] += sign * w
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts])


class BGEM3Embedder:  # pragma: no cover - requires model weights
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        self.dim = 1024

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
