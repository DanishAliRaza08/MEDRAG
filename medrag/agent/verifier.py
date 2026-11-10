"""The critic: checks every claim in a draft against the retrieved evidence.

* Image-finding claims are grounded in the image itself (source = "image").
* Every other claim must cite at least one retrieved passage and share at
  least ``threshold`` of its content terms with a cited passage.
"""

from __future__ import annotations

from ..textutils import content_terms
from .generator import claim_support
from .state import Draft, Verification


class Verifier:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, draft: Draft, evidence: list[dict]) -> Verification:
        unsupported, supported = [], 0
        missing: list[str] = []
        for c in draft.claims:
            if c.source == "image":
                c.supported, c.support_score = True, 1.0
                supported += 1
                continue
            score = claim_support(c, evidence)
            c.support_score = round(score, 3)
            c.supported = score >= self.threshold
            if c.supported:
                supported += 1
            else:
                unsupported.append(c.text)
                if c.concept and c.concept not in missing:
                    missing.append(c.concept)
        total = supported + len(unsupported)
        rate = len(unsupported) / total if total else 0.0
        return Verification(supported, len(unsupported), unsupported, round(rate, 4), not unsupported, missing)

    @staticmethod
    def reformulate(question: str, verification: Verification, previous: list[str]) -> str:
        """Build a follow-up query focused on what the draft could not support."""
        focus = verification.missing_concepts or [
            " ".join(content_terms(t)[:6]) for t in verification.unsupported_claims[:2]
        ]
        query = " ".join(focus) + " diagnosis management guideline"
        if query in previous:
            query = question + " " + query
        return query
