"""Import guideline documents (PDF, plain text, Markdown) into a knowledge-base folder.

Typical use: a hospital adds its licensed NICE / WHO / local protocol PDFs. Text is extracted, cleaned (page
furniture, hyphenation, repeated headers), PHI-scrubbed and written as markdown with front matter, so it is indexed
like every other knowledge-base document and cited with its source.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path

from ..deid import scrub_text
from ..retrieval.chunking import doc_id_for

MODALITIES = ("xray", "fundus", "histopathology", "general")


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    pages = [(p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages]
    # lines that repeat on most pages are running headers/footers
    counts = Counter(line.strip() for page in pages for line in set(page.splitlines()) if line.strip())
    furniture = {line for line, c in counts.items()
                 if len(pages) >= 3 and c >= 0.6 * len(pages) and len(line.split()) <= 8}
    lines = []
    for page in pages:
        for line in page.splitlines():
            s = line.strip()
            if not s or s in furniture or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", s, re.I):
                continue
            lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"(?<![.:!?])\n(?=[a-z(])", " ", text)  # join wrapped lines inside sentences
    return text


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return pdf_text(data)
    if name.endswith((".txt", ".md", ".markdown")):
        return data.decode("utf-8", errors="replace")
    raise ValueError("supported formats: .pdf, .txt, .md")


def to_markdown(title: str, text: str, source: str, modality: str) -> tuple[str, str]:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    text, _ = scrub_text(text)
    clean = lambda v: re.sub(r"[\r\n:]+", " ", v).strip()  # noqa: E731
    doc_id = doc_id_for(title + text)
    md = f"---\nid: {doc_id}\ntitle: {clean(title)}\nsource: {clean(source)}\nmodality: {modality}\n---\n{text}\n"
    return doc_id, md


def import_bytes(filename: str, data: bytes, out_dir: Path, title: str, source: str = "imported document",
                 modality: str = "general") -> Path:
    text = extract_text(filename, data)
    if len(text.split()) < 20:
        raise ValueError("no extractable text found (scanned PDFs need OCR first)")
    doc_id, md = to_markdown(title, text, source, modality)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc_id}.md"
    path.write_text(md, encoding="utf-8")
    return path


def import_file(path: Path, out_dir: Path, title: str, source: str = "imported document",
                modality: str = "general") -> Path:
    return import_bytes(path.name, path.read_bytes(), out_dir, title, source, modality)
