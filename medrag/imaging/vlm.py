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
List only findings that are directly visible in the image. Do not diagnose beyond what is visible.
Return JSON: {{"findings": [{{"text": "...", "concept": "short medical term", "location": "...",
"confidence": 0.0-1.0}}]}}"""


class QwenVLM:  # pragma: no cover - requires a served model
    name = "qwen2.5-vl"

    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()

    def describe(self, image_png: bytes, modality: str, question: str = "") -> list[Finding]:
        prompt = VLM_PROMPT.format(mname=MODALITY_NAMES.get(modality, modality), question=question or "n/a")
        out = self.client.chat([{"role": "user", "content": [image_part(image_png), {"type": "text", "text": prompt}]}])
        data = extract_json(out)
        return [Finding(f["text"], f.get("concept", ""), f.get("location", ""), float(f.get("confidence", 0.5)))
                for f in data.get("findings", [])]
