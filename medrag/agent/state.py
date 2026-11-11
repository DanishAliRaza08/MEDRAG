"""Agent state and the structured draft report produced by the agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..imaging.analysis import Finding


@dataclass
class Claim:
    text: str
    section: str  # findings | impression | differential | recommendations
    citations: list[int] = field(default_factory=list)  # 1-based indices into evidence
    concept: str = ""
    source: str = "evidence"  # "image" for visual findings
    supported: bool | None = None
    support_score: float | None = None


@dataclass
class Draft:
    claims: list[Claim] = field(default_factory=list)
    summary: str = ""
    generator: str = ""

    def section(self, name: str) -> list[Claim]:
        return [c for c in self.claims if c.section == name]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verification:
    supported: int
    unsupported: int
    unsupported_claims: list[str]
    hallucination_rate: float
    passed: bool
    missing_concepts: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    question: str
    modality: str | None = None
    age_band: str | None = None
    image_png: bytes | None = None
    study_id: str | None = None

    needs_image: bool = False
    queries: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    similar_cases: list[dict] = field(default_factory=list)
    draft: Draft | None = None
    verification: Verification | None = None
    iterations: int = 0
    status: str = "pending"  # pending | verified | needs_review | unverified
    trace: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # model/service failures; any error forces needs_review

    def log(self, node: str, **info) -> None:
        self.trace.append({"step": len(self.trace) + 1, "node": node, **info})

    def result(self) -> dict:
        return {
            "question": self.question,
            "modality": self.modality,
            "status": self.status,
            "iterations": self.iterations,
            "queries": self.queries,
            "evidence": self.evidence,
            "findings": [f.to_dict() for f in self.findings],
            "similar_cases": self.similar_cases,
            "draft": self.draft.to_dict() if self.draft else None,
            "verification": asdict(self.verification) if self.verification else None,
            "trace": self.trace,
            "errors": self.errors,
        }
