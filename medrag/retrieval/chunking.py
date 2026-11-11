"""Sentence-boundary chunking with token overlap.

Passages of roughly ``max_tokens`` whitespace tokens are produced without ever
cutting a sentence in half; the last sentences of each chunk are repeated at
the start of the next one so that context spanning a boundary is not lost.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..textutils import split_sentences


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "unknown"
    modality: list[str] = field(default_factory=lambda: ["general"])
    age_band: str = "any"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    modality: list[str]
    age_band: str
    position: int

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "modality": list(self.modality),
            "age_band": self.age_band,
            "position": self.position,
            "kind": "guideline",
        }


def _ntok(s: str) -> int:
    return len(s.split())


def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 60) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        n = _ntok(sent)
        if current and size + n > max_tokens:
            chunks.append(" ".join(current))
            # carry trailing sentences forward as overlap
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + _ntok(prev) > overlap_tokens:
                    break
                carry.insert(0, prev)
                carried += _ntok(prev)
            current, size = carry, carried
        current.append(sent)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: Document, max_tokens: int = 500, overlap_tokens: int = 60) -> list[Chunk]:
    out = []
    for i, text in enumerate(chunk_text(doc.text, max_tokens, overlap_tokens)):
        cid = f"{doc.doc_id}#{i}"
        out.append(Chunk(cid, doc.doc_id, doc.title, text, doc.source, doc.modality, doc.age_band, i))
    return out


_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def parse_markdown(text: str, fallback_id: str) -> Document:
    """Parse a markdown file with an optional simple ``key: value`` front matter."""
    meta: dict[str, str] = {}
    m = _FRONT_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        text = text[m.end():]
    # Drop markdown heading markers / bullets but keep the words.
    body_lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue  # headings are structure, not statements: never index them as sentences
        line = re.sub(r"^\s*([-*]|\d+\.)\s+", "", line).strip()
        if line:
            if not line.endswith((".", "!", "?", ":")):
                line += "."
            body_lines.append(line)
    body = " ".join(body_lines)
    title = meta.get("title") or fallback_id
    modality = [x.strip() for x in meta.get("modality", "general").split(",") if x.strip()]
    doc_id = meta.get("id") or fallback_id
    return Document(doc_id, title, body, meta.get("source", "unknown"), modality, meta.get("age_band", "any"))


def load_directory(path: Path) -> list[Document]:
    docs = []
    for p in sorted(Path(path).glob("*.md")):
        if p.stem.isupper():  # README.md, ATTRIBUTION.md, ... are not evidence
            continue
        docs.append(parse_markdown(p.read_text(encoding="utf-8"), p.stem))
    return docs


def doc_id_for(text: str) -> str:
    return "doc-" + hashlib.sha1(text.encode()).hexdigest()[:10]
