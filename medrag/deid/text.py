"""Rule-based PHI scrubbing for free text (clinical notes, questions).

Covers the HIPAA Safe-Harbour categories that occur in free text: names
introduced by a label or title, record numbers, dates, phone numbers, e-mail
addresses, national ID numbers (incl. Pakistani CNIC), URLs, IPs, addresses
and ages over 89.
"""

from __future__ import annotations

import re

_NAME = r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3}"
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("URL", re.compile(r"\bhttps?://\S+", re.I)),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("CNIC", re.compile(r"\b\d{5}-\d{7}-\d\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:MRN|MR\s*No|Medical\s+Record(?:\s+Number)?|Hospital\s+No|Acc(?:ession)?\s*#?|ID)"
                       r"\s*[:#]?\s*[A-Z0-9-]{4,}\b", re.I)),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{3,4})[\s.-]?\d{3}[\s.-]?\d{3,4}\b")),
    ("DATE", re.compile(
        r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b", re.I)),
    ("NAME", re.compile(rf"\b(?:Name|Patient|Pt|Mr|Mrs|Ms|Miss|Dr|Doctor)\.?\s*[:\-]?\s+{_NAME}")),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Block|"
                           r"Colony|Town|Boulevard|Blvd)\b\.?")),
    ("AGE", re.compile(r"\b(?:9\d|1[0-2]\d)\s*(?:-|\s)?(?:years?|yrs?|y/?o)(?:[\s-]old)?\b", re.I)),
]


def scrub_text(text: str) -> tuple[str, dict[str, int]]:
    """Return (scrubbed_text, counts_per_category)."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts
    for label, pat in PATTERNS:
        text, n = pat.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


def contains_phi(text: str) -> bool:
    return bool(scrub_text(text)[1])
