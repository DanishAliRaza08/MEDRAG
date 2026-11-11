"""Report drafting and question answering.

``ExtractiveGenerator`` (offline) writes every non-image claim by selecting a
sentence from the retrieved evidence and citing it; if no evidence supports a
visual finding it says so in an *uncited* claim, which the verifier will flag
and which drives another retrieval round. ``LLMGenerator`` (real mode) asks
the served model for a JSON report constrained to the numbered evidence.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from ..imaging.analysis import Finding
from ..llm import ChatClient, extract_json
from ..textutils import content_terms, overlap_ratio, split_sentences
from .state import Claim, Draft

RECOMMEND_CUES = ("recommend", "should", "referred", "refer ", "offer", "arrange", "advised")
DIFFERENTIAL_CUES = ("differential", "distinguish", "mimic", "causes of")
_NEGATIONS = ("no ", "without ", "absence of ", "not ", "free of ")


def _negated(sentence: str, concept: str) -> bool:
    """True if the concept only occurs in a negated phrase, e.g. "no nuclear pleomorphism"."""
    s = sentence.lower()
    head = concept.lower().split()[0] if concept else ""
    if not head or head not in s:
        return False
    for i in [j for j in range(len(s)) if s.startswith(head, j)]:
        window = s[max(0, i - 25) : i]
        if not any(n in window for n in _NEGATIONS):
            return False
    return True


class Generator(Protocol):
    name: str

    def draft(self, question: str, evidence: list[dict], findings: list[Finding], similar: list[dict]) -> Draft: ...

    def answer(self, question: str, evidence: list[dict]) -> dict: ...


# Review-article "meta" sentences describe the paper, not the medicine; they must never become clinical claims.
_META = re.compile(r"^(we|our|this (review|article|paper|study|guideline|document)|in this (review|article|paper)|"
                   r"here,? we|the (aim|purpose|objective) of|the present (review|study)|herein|a recent study|"
                   # sentences that open with a pronoun depend on context the reader cannot see
                   r"they|it|these|this|those|such|its|their|he|she)\b", re.I)
_NOISE = re.compile(r"\b(et al|fig(ure)?\.?\s*\d|table\s*\d|doi|https?://)", re.I)


def _sentences(evidence: list[dict]) -> list[tuple[int, str]]:
    out = []
    for i, ev in enumerate(evidence, start=1):
        for s in split_sentences(ev["text"]):
            n = len(s.split())
            if 5 <= n <= 60 and not _META.match(s) and not _NOISE.search(s) and s[-1] in ".!":
                out.append((i, s))
    return out


def _best(sentences: list[tuple[int, str]], query: str, used: set[str], cues=None,
          must: list[str] | None = None, must_overlap: float = 0.5) -> tuple[int, str] | None:
    """Pick the unused evidence sentence that best matches ``query``.

    ``cues``: only sentences containing one of these phrases are eligible.
    ``must``: concepts; the sentence must contain more than ``must_overlap`` of the terms of at
    least one of them, and must not mention that concept only in negated form.
    """
    qterms = set(content_terms(query))
    concepts = [(m, set(content_terms(m))) for m in (must or []) if content_terms(m)]
    best, best_score = None, 0.0
    for idx, s in sentences:
        if s in used or (cues and not any(c in s.lower() for c in cues)):
            continue
        sterms = set(content_terms(s))
        if not sterms:
            continue
        matched = [m for m, mt in concepts if len(mt & sterms) / len(mt) > must_overlap and not _negated(s, m)]
        if concepts and not matched:
            continue
        score = len(qterms & sterms) / (len(qterms) or 1) + 1e-3 / (1 + len(sterms))
        if any(m.lower() in s.lower() for m in matched):
            score += 0.5  # the sentence names the exact finding
        if score > best_score:
            best, best_score = (idx, s), score
    return best


class ExtractiveGenerator:
    name = "offline-extractive"

    def draft(self, question, evidence, findings, similar):
        claims: list[Claim] = []
        used: set[str] = set()
        sents = _sentences(evidence)
        for f in findings:
            claims.append(Claim(f.text, "findings", [], f.concept, "image"))
        abnormal = [f for f in findings if not f.concept.startswith("normal") and f.concept != "unsupported modality"]
        for f in abnormal:
            hit = _best(sents, f"{f.concept} {f.text} {question}", used, must=[f.concept])
            if hit:
                used.add(hit[1])
                claims.append(Claim(hit[1], "impression", [hit[0]], f.concept))
            else:
                # Nothing retrieved supports this finding yet: the claim is left uncited on purpose.
                claims.append(Claim(f"The {f.concept} requires guideline-directed management.", "impression",
                                    [], f.concept))
        focus = " ".join([question] + [f.concept for f in abnormal])
        must = [f.concept for f in abnormal] or [f.concept for f in findings] or None
        if not abnormal:
            n_imp = 1 if findings else 2
            for _ in range(n_imp):
                hit = _best(sents, focus + " " + " ".join(f.concept for f in findings), used)
                if hit:
                    used.add(hit[1])
                    claims.append(Claim(hit[1], "impression", [hit[0]]))
        # A differential only makes sense for an abnormality (or a text-only question).
        hit = _best(sents, focus, used, DIFFERENTIAL_CUES, must=must) if (abnormal or not findings) else None
        if hit:
            used.add(hit[1])
            claims.append(Claim(hit[1], "differential", [hit[0]]))
        for _ in range(2):
            hit = _best(sents, focus, used, RECOMMEND_CUES, must=must)
            if hit:
                used.add(hit[1])
                claims.append(Claim(hit[1], "recommendations", [hit[0]]))
        if not evidence:
            claims.append(Claim("Management should follow standard clinical practice.", "recommendations", []))
        concepts = [f.concept for f in findings if not f.concept.startswith("normal")]
        summary = ("Draft for clinician review. Image findings: " + ", ".join(concepts) + "." if concepts
                   else "Draft for clinician review.")
        if similar:
            summary += f" {len(similar)} similar prior case(s) retrieved for comparison."
        return Draft(claims, summary, self.name)

    def answer(self, question, evidence):
        sents = _sentences(evidence)
        used: set[str] = set()
        parts, cites = [], []
        for _ in range(3):
            hit = _best(sents, question, used)
            if not hit:
                break
            used.add(hit[1])
            parts.append(f"{hit[1]} [{hit[0]}]")
            cites.append(hit[0])
        if not parts:
            return {"answer": "No supporting evidence was found in the knowledge base for this question.",
                    "citations": []}
        return {"answer": " ".join(parts), "citations": sorted(set(cites))}


DRAFT_PROMPT = """You are a clinical decision-support assistant drafting a report for a clinician to review.
Use ONLY the numbered evidence passages and the image findings below. Every impression, differential or
recommendation claim MUST cite at least one evidence number. Never invent citations or facts.
If the evidence does not support a statement, omit it.

Clinical question: {question}

Image findings:
{findings}

Evidence:
{evidence}

Return JSON: {{"summary": "...", "claims": [{{"text": "...", "section": "impression|differential|recommendations",
"citations": [1], "concept": "..."}}]}}"""


class LLMGenerator:  # pragma: no cover - requires a served model
    name = "llm"

    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()

    MAX_SENTENCES = 4  # per passage: keeps prompts short (fast on CPU, fewer distractions for small models)

    @classmethod
    def _ev(cls, evidence, focus: str = ""):
        """Numbered evidence, each passage compressed to its sentences most relevant to the question/findings.

        Citations still refer to the full passage, and verification checks claims against the full text."""
        terms = set(content_terms(focus))
        lines = []
        for i, e in enumerate(evidence, start=1):
            sents = split_sentences(e["text"])
            if terms and len(sents) > cls.MAX_SENTENCES:
                ranked = sorted(range(len(sents)), key=lambda k: -len(terms & set(content_terms(sents[k]))))
                sents = [sents[k] for k in sorted(ranked[: cls.MAX_SENTENCES])]
            lines.append(f"[{i}] ({e['title']}) " + " ".join(sents))
        return "\n".join(lines) or "(none)"

    def _json_chat(self, prompt: str, max_tokens: int):
        out = self.client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
        try:
            return extract_json(out)
        except (ValueError, json.JSONDecodeError):
            fix = self.client.chat([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": out},
                                    {"role": "user", "content": "That was not valid JSON. Reply with only the JSON."}],
                                   max_tokens=max_tokens)
            return extract_json(fix)

    def draft(self, question, evidence, findings, similar):
        fl = "\n".join(f"- {f.text}" for f in findings) or "- (no image)"
        focus = question + " " + " ".join(f.concept + " " + f.text for f in findings)
        try:
            data = self._json_chat(DRAFT_PROMPT.format(question=question, findings=fl,
                                                       evidence=self._ev(evidence, focus)), max_tokens=700)
        except (ValueError, json.JSONDecodeError):
            data = {"summary": "The model did not return a structured draft; image findings only.", "claims": []}
        claims = [Claim(f.text, "findings", [], f.concept, "image") for f in findings]
        n = len(evidence)
        sections = {"impression", "differential", "recommendations"}
        for c in data.get("claims", []) if isinstance(data, dict) else []:
            if not isinstance(c, dict) or not str(c.get("text", "")).strip():
                continue
            cites = [int(x) for x in c.get("citations", []) or [] if str(x).isdigit() and 1 <= int(x) <= n]
            section = c.get("section", "impression") if c.get("section") in sections else "impression"
            claims.append(Claim(str(c["text"]), section, cites, str(c.get("concept", ""))))
        return Draft(claims, str(data.get("summary", "")), f"llm:{self.client.served_model}")

    def answer(self, question, evidence):
        out = self.client.chat([{"role": "user", "content":
            "Answer the clinical question using ONLY the evidence; cite as [n]. If unsupported, say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{self._ev(evidence, question)}\n\n"
            'Return JSON {"answer": "...", "citations": [n]}'}])
        try:
            return extract_json(out)
        except (ValueError, json.JSONDecodeError):
            return {"answer": out, "citations": []}


def claim_support(claim: Claim, evidence: list[dict]) -> float:
    """Best lexical support score of a claim against its cited passages."""
    if not claim.citations:
        return 0.0
    scores = [overlap_ratio(claim.text, evidence[i - 1]["text"]) for i in claim.citations if 0 < i <= len(evidence)]
    return max(scores, default=0.0)
