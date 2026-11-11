"""Vision-language model interface.

``OfflineVLM`` wraps the deterministic analysers; ``QwenVLM`` sends the
de-identified image to Qwen2.5-VL served by vLLM.
"""

from __future__ import annotations

import io
from typing import Protocol

from PIL import Image

from ..llm import ChatClient, extract_json, image_part
from .analysis import Finding, analyze

MODALITY_NAMES = {"xray": "chest radiograph", "fundus": "retinal fundus photograph",
                  "histopathology": "histopathology slide (H&E)"}


class VisionModel(Protocol):
    name: str

    def describe(self, image_png: bytes, modality: str, question: str = "") -> list[Finding]: ...


class OfflineVLM:
    name = "offline-heuristic-vlm"

    def describe(self, image_png: bytes, modality: str, question: str = "") -> list[Finding]:
        return analyze(Image.open(io.BytesIO(image_png)), modality)


VLM_PROMPT = """You are assisting a clinician by describing a de-identified {mname}.
Clinical question: {question}
Report the ABNORMAL findings that are directly visible, one per item, each with a short medical concept name
(for example: pleural effusion, consolidation, cardiomegaly, pneumothorax, atelectasis, nodule). Do not list normal
structures and do not diagnose beyond what is visible. If the image looks normal, return exactly one item whose
concept is "normal".
Return only JSON: {{"findings": [{{"text": "...", "concept": "...", "location": "...", "confidence": 0.0-1.0}}]}}"""

NORMAL_CONCEPT = {"xray": "normal chest radiograph", "fundus": "normal fundus", "histopathology": "benign tissue"}


def normalise_findings(findings: list[Finding], modality: str) -> list[Finding]:
    """Turn negated or "normal" items into a normal concept so they never drive evidence search as abnormalities."""
    from ..textutils import asserts

    out = []
    for f in findings:
        concept = (f.concept or "").strip().lower()
        is_normal_word = not concept or concept.startswith(("normal", "no ", "unremarkable", "none"))
        negated_only = concept in f.text.lower() and not asserts(f.text, concept)  # "no evidence of effusion"
        if is_normal_word or negated_only:
            f.concept = NORMAL_CONCEPT.get(modality, "normal")
        out.append(f)
    abnormal = [f for f in out if not f.concept.startswith("normal") and f.concept != "benign tissue"]
    return abnormal or out[:1] or [Finding("No abnormality identified.", NORMAL_CONCEPT.get(modality, "normal"))]


class QwenVLM:  # pragma: no cover - requires a served model
    name = "qwen2.5-vl"

    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()

    def describe(self, image_png: bytes, modality: str, question: str = "") -> list[Finding]:
        prompt = VLM_PROMPT.format(mname=MODALITY_NAMES.get(modality, modality), question=question or "n/a")
        out = self.client.chat([{"role": "user", "content": [image_part(image_png), {"type": "text", "text": prompt}]}])
        try:
            data = extract_json(out)
        except Exception:  # not JSON: keep the free text as a single finding rather than failing the case
            return normalise_findings([Finding(out.strip()[:500], "", "", 0.3)], modality)
        items = data.get("findings", []) if isinstance(data, dict) else data
        findings = [Finding(str(f.get("text", "")), str(f.get("concept", "")), str(f.get("location", "")),
                            float(f.get("confidence", 0.5) or 0.5)) for f in items if isinstance(f, dict)]
        return normalise_findings(findings, modality)
