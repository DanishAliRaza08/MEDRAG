"""A small Okapi BM25 keyword index (the sparse half of hybrid retrieval)."""

from __future__ import annotations

import math
from collections import Counter

from ..textutils import content_terms


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, item_id: str, text: str) -> None:
        if item_id in self.ids:
            self.remove(item_id)
        terms = content_terms(text)
        tf = Counter(terms)
        self.ids.append(item_id)
        self.tfs.append(tf)
        self.lengths.append(len(terms))
        self.df.update(tf.keys())

    def remove(self, item_id: str) -> None:
        i = self.ids.index(item_id)
        self.df.subtract(self.tfs[i].keys())
        del self.ids[i], self.tfs[i], self.lengths[i]

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        n = len(self.ids)
        avgdl = sum(self.lengths) / n or 1.0
        q = content_terms(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            if allowed is not None and self.ids[i] not in allowed:
                continue
            s = 0.0
            dl = self.lengths[i]
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
