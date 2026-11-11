"""The critic: checks every claim in a draft against the retrieved evidence.

* Image-finding claims are grounded in the image itself (source = "image").
* Every other claim must be grounded in the sources the synopsis allows: the cited evidence passages AND the image
  findings of this study. A statement such as "the radiograph shows cardiomegaly" can only ever be supported by
  the image findings (no guideline describes this patient), so the findings are always available as premises.
* ``Verifier`` (lexical): every other claim must cite at least one retrieved passage and share at least
  ``threshold`` of its content terms with a cited passage. Deterministic and dependency-free.
* ``NLIVerifier`` (semantic): a natural-language-inference cross-encoder decides whether the cited passage
  *entails* the claim. It accepts faithful paraphrases that share few words, and it rejects claims that copy
  words from the evidence but contradict it (e.g. a dropped or added "no").
"""

from __future__ import annotations

from ..textutils import content_terms
from .generator import claim_support
from .state import Claim, Draft, Verification


class Verifier:
    # bump when the grounding rules change, so stored evaluation checkpoints are not reused across rule changes
    RULES_VERSION = 2  # v2: image findings are premises (synopsis: evidence AND image findings)

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, draft: Draft, evidence: list[dict]) -> Verification:
        unsupported, supported = [], 0
        missing: list[str] = []
        findings = [c.text for c in draft.claims if c.source == "image" and c.text.strip()]
        for c in draft.claims:
            if c.source == "image":
                c.supported, c.support_score = True, 1.0
                supported += 1
                continue
            c.supported, score = self.judge(c, evidence, findings)
            c.support_score = round(score, 3)
            if c.supported:
                supported += 1
            else:
                unsupported.append(c.text)
                if c.concept and c.concept not in missing:
                    missing.append(c.concept)
        total = supported + len(unsupported)
        rate = len(unsupported) / total if total else 0.0
        return Verification(supported, len(unsupported), unsupported, round(rate, 4), not unsupported, missing)

    def judge(self, claim: Claim, evidence: list[dict], findings: list[str] | None = None) -> tuple[bool, float]:
        from ..textutils import overlap_ratio

        score = claim_support(claim, evidence)
        if findings:  # restating / building on this study's image findings
            score = max(score, overlap_ratio(claim.text, " ".join(findings)))
        return score >= self.threshold, score

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


class NLIVerifier(Verifier):
    """Entailment-based critic (cross-encoder/nli-deberta-v3-small by default; loaded on first use).

    NLI models are trained on short premises: given a claim and its source sentence alone the model says
    "entailment", but add two unrelated sentences and it says "neutral". So each cited passage is split into single
    sentences and adjacent sentence pairs. The claim is supported if ANY premise entails it (probability >=
    ``entail_threshold``). Only when nothing entails it, a premise that is about the same thing (>= 50% term
    overlap) and contradicts it (>= ``contradict_threshold``) marks it as contradicting its source.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5,
                 entail_threshold: float = 0.5, contradict_threshold: float = 0.8):
        super().__init__(threshold)
        self.model_name = model_name
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        self._model = None
        self._labels: dict[str, int] = {}

    def _load(self):  # pragma: no cover - requires model weights
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            id2label = self._model.model.config.id2label
            self._labels = {v.lower(): int(k) for k, v in id2label.items()}
        return self._model

    @staticmethod
    def _windows(text: str, size: int = 2) -> list[str]:
        """Every sentence, plus every run of up to ``size`` adjacent sentences (for claims that span two)."""
        from ..textutils import split_sentences

        sents = split_sentences(text) or [text]
        out = list(sents)
        for n in range(2, size + 1):
            out += [" ".join(sents[i: i + n]) for i in range(0, len(sents) - n + 1)]
        return out

    def probabilities(self, premises: list[str], hypothesis: str) -> list[tuple[float, float]]:
        """(entailment, contradiction) probability per premise."""
        import numpy as np

        model = self._load()
        logits = np.asarray(model.predict([(p, hypothesis) for p in premises]))
        logits = logits.reshape(len(premises), -1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return [(float(r[self._labels["entailment"]]), float(r[self._labels["contradiction"]])) for r in probs]

    def judge(self, claim: Claim, evidence: list[dict], findings: list[str] | None = None) -> tuple[bool, float]:
        from ..textutils import split_sentences

        cited = []
        for i in claim.citations:
            if 0 < i <= len(evidence):
                cited += self._windows(evidence[i - 1]["text"])
        findings = list(findings or [])
        # premises: cited passages, the image findings, and each finding joined with each cited sentence (a claim may
        # combine both: "the radiograph shows X [image], which suggests Y [evidence]")
        single = [s for i in claim.citations if 0 < i <= len(evidence)
                  for s in split_sentences(evidence[i - 1]["text"])]
        premises = cited + findings + [f"{f} {s}" for f in findings for s in single]
        if not premises:
            return False, 0.0
        from ..textutils import overlap_ratio

        scores = self.probabilities(premises, claim.text)
        entail = max(s[0] for s in scores)
        if entail >= self.entail_threshold:
            return True, entail
        contra = max((s[1] for p, s in zip(premises, scores, strict=True) if overlap_ratio(claim.text, p) >= 0.5),
                     default=0.0)
        if contra >= self.contradict_threshold:
            return False, -contra  # negative score marks an explicit contradiction of the cited source
        return False, entail
