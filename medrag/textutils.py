"""Small, dependency-free text utilities shared by retrieval and verification."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have having he her
    his how i if in into is it its itself may me might more most must my no nor not of on or other our
    out over own same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your also any each per via vs within without about above after again against all am both
    during few further here just only once s t don now patient patients""".split()
)

# Collapse common inflections so that "opacities" matches "opacity" and so on.
_SUFFIXES = ("ations", "ation", "ities", "ity", "ies", "ing", "ed", "es", "s")


def stem(token: str) -> str:
    for suf in _SUFFIXES:
        if len(token) > len(suf) + 3 and token.endswith(suf):
            base = token[: -len(suf)]
            if suf == "ies":
                base += "y"
            return base
    return token


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Lower-cased, stemmed tokens with stopwords removed."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def overlap_ratio(claim: str, evidence: str) -> float:
    """Fraction of the claim's content terms that occur in the evidence."""
    c = set(content_terms(claim))
    if not c:
        return 1.0
    e = set(content_terms(evidence))
    return len(c & e) / len(c)
