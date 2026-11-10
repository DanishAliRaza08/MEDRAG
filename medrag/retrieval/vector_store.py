"""Vector stores with an enforced payload allowlist.

The allowlist is a privacy guarantee: any payload field that is not explicitly
permitted is rejected, so patient identifiers can never reach the vector
database even if a caller tries to attach them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GUIDELINE_FIELDS = frozenset(
    {"chunk_id", "doc_id", "title", "text", "source", "modality", "age_band", "position", "kind"}
)
CASE_FIELDS = frozenset({"study_id", "modality", "age_band", "finding_summary", "kind"})
ALLOWED_PAYLOAD_FIELDS = GUIDELINE_FIELDS | CASE_FIELDS


class PayloadNotAllowed(ValueError):
    pass


def check_payload(payload: dict) -> None:
    bad = set(payload) - ALLOWED_PAYLOAD_FIELDS
    if bad:
        raise PayloadNotAllowed(f"payload fields not on allowlist: {sorted(bad)}")


class InMemoryVectorStore:
    """Exact cosine search over a numpy matrix; persisted as .npy + .json."""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def upsert(self, item_id: str, vector: np.ndarray, payload: dict) -> None:
        check_payload(payload)
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.shape[1]}")
        if item_id in self.ids:
            i = self.ids.index(item_id)
            self.matrix[i] = vector[0]
            self.payloads[i] = dict(payload)
            return
        self.ids.append(item_id)
        self.payloads.append(dict(payload))
        self.matrix = np.vstack([self.matrix, vector])

    def get(self, item_id: str) -> dict:
        return self.payloads[self.ids.index(item_id)]

    def search(self, vector: np.ndarray, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.matrix @ np.asarray(vector, dtype=np.float32).reshape(-1)
        order = np.argsort(-sims)
        out = []
        for i in order:
            if allowed is not None and self.ids[i] not in allowed:
                continue
            out.append((self.ids[i], float(sims[i])))
            if len(out) >= k:
                break
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.matrix)
        (path / "payloads.json").write_text(json.dumps({"ids": self.ids, "payloads": self.payloads}))

    @classmethod
    def load(cls, path: Path, dim: int) -> InMemoryVectorStore:
        store = cls(dim)
        if (path / "vectors.npy").exists():
            store.matrix = np.load(path / "vectors.npy").astype(np.float32)
            data = json.loads((path / "payloads.json").read_text())
            store.ids, store.payloads = data["ids"], data["payloads"]
            if store.matrix.shape[1] != dim:
                return cls(dim)
        return store


class QdrantVectorStore:  # pragma: no cover - requires a running Qdrant
    """Same interface backed by Qdrant (used in real / containerised mode)."""

    def __init__(self, dim: int, url: str, collection: str = "medrag_guidelines"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dim, self.collection = dim, collection
        self.client = QdrantClient(url=url)
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return self.client.count(self.collection).count

    @staticmethod
    def _pid(item_id: str) -> str:
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, item_id))

    def upsert(self, item_id, vector, payload):
        from qdrant_client.models import PointStruct

        check_payload(payload)
        self.client.upsert(
            self.collection,
            [PointStruct(id=self._pid(item_id), vector=list(map(float, vector)), payload={**payload, "_id": item_id})],
        )
        self._cache[item_id] = dict(payload)

    def get(self, item_id):
        if item_id not in self._cache:
            rec = self.client.retrieve(self.collection, [self._pid(item_id)])[0]
            self._cache[item_id] = {k: v for k, v in rec.payload.items() if k != "_id"}
        return self._cache[item_id]

    def search(self, vector, k=10, allowed=None):
        res = self.client.search(self.collection, query_vector=list(map(float, vector)), limit=k * 4)
        out = []
        for r in res:
            iid = r.payload["_id"]
            if allowed is None or iid in allowed:
                out.append((iid, float(r.score)))
        return out[:k]

    def save(self, path):
        pass
