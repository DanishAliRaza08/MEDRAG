"""The critic: checks every claim in a draft against the retrieved evidence.

* Image-finding claims are grounded in the image itself (source = "image").
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
            c.supported, score = self.judge(c, evidence)
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

    def judge(self, claim: Claim, evidence: list[dict]) -> tuple[bool, float]:
        score = claim_support(claim, evidence)
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

    The cited passage is split into windows of three sentences (NLI models are trained on short premises); the claim
    is supported when some window entails it with probability >= ``entail_threshold`` and no window contradicts it
    with probability >= ``contradict_threshold``.
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
    def _windows(text: str, size: int = 3) -> list[str]:
        from ..textutils import split_sentences

        sents = split_sentences(text)
        if len(sents) <= size:
            return [text]
        return [" ".join(sents[i: i + size]) for i in range(0, len(sents) - size + 1)]

    def probabilities(self, premises: list[str], hypothesis: str) -> list[tuple[float, float]]:
        """(entailment, contradiction) probability per premise."""
        import numpy as np

        model = self._load()
        logits = np.asarray(model.predict([(p, hypothesis) for p in premises]))
        logits = logits.reshape(len(premises), -1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return [(float(r[self._labels["entailment"]]), float(r[self._labels["contradiction"]])) for r in probs]

    def judge(self, claim: Claim, evidence: list[dict]) -> tuple[bool, float]:
        if not claim.citations:
            return False, 0.0
        premises = []
        for i in claim.citations:
            if 0 < i <= len(evidence):
                premises += self._windows(evidence[i - 1]["text"])
        if not premises:
            return False, 0.0
        scores = self.probabilities(premises, claim.text)
        entail = max(s[0] for s in scores)
        contra = max(s[1] for s in scores)
        if contra >= self.contradict_threshold and contra > entail:
            return False, -contra  # negative score marks an explicit contradiction
        return entail >= self.entail_threshold, entail
