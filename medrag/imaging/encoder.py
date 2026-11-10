"""Image encoders and the similar-case index.

``ThumbnailEncoder`` (offline) builds a small, normalised descriptor from a
grey-level thumbnail and colour histograms. ``BiomedCLIPEncoder`` (real mode)
uses BiomedCLIP through open_clip.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ..retrieval.vector_store import InMemoryVectorStore


class ThumbnailEncoder:
    dim = 32 * 32 + 48

    def encode(self, image_png: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_png)).convert("RGB")
        g = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32).ravel()
        g = (g - g.mean()) / (g.std() + 1e-6)
        rgb = np.asarray(img.resize((64, 64)), dtype=np.int32)
        hist = np.concatenate([np.histogram(rgb[..., c], bins=16, range=(0, 256))[0] for c in range(3)])
        hist = hist.astype(np.float32) / hist.sum() * 8
        v = np.concatenate([g / np.sqrt(len(g)), hist])
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class BiomedCLIPEncoder:  # pragma: no cover - requires model weights
    dim = 512

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._m = None

    def encode(self, image_png: bytes) -> np.ndarray:
        import open_clip
        import torch

        if self._m is None:
            model, preprocess = open_clip.create_model_from_pretrained(self.model_name)
            self._m = (model.eval(), preprocess)
        model, preprocess = self._m
        img = preprocess(Image.open(io.BytesIO(image_png)).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            v = model.encode_image(img)[0].numpy()
        return (v / np.linalg.norm(v)).astype(np.float32)


class CaseIndex:
    """Similar prior-case search over de-identified studies."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.store = InMemoryVectorStore(encoder.dim)

    def add(self, study_id: str, image_png: bytes, modality: str, age_band: str, finding_summary: str) -> None:
        self.store.upsert(study_id, self.encoder.encode(image_png),
                          {"study_id": study_id, "modality": modality, "age_band": age_band,
                           "finding_summary": finding_summary, "kind": "case"})

    def similar(self, image_png: bytes, modality: str, k: int = 3, exclude: str | None = None) -> list[dict]:
        allowed = {i for i, p in zip(self.store.ids, self.store.payloads, strict=True)
                   if p["modality"] == modality and i != exclude}
        if not allowed:
            return []
        hits = self.store.search(self.encoder.encode(image_png), k, allowed)
        return [{**self.store.get(i), "similarity": round(s, 4)} for i, s in hits]
