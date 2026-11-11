"""The agent as an explicit, dependency-free state machine.

    START -> route -> retrieve -> [inspect_image] -> draft -> verify -> END
                         ^                                       |
                         +------- reformulated query ------------+   (bounded)

The best outcome the agent can reach on its own is a *verified draft*. Signing
off a report is outside the agent entirely (see ``medrag.reports.lifecycle``).
``build_langgraph`` expresses the identical graph with LangGraph if installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..imaging.vlm import OfflineVLM, QwenVLM
from ..retrieval import RetrievalPipeline
from .generator import ExtractiveGenerator, LLMGenerator
from .state import AgentState, Claim, Draft
from .verifier import NLIVerifier, Verifier

IMAGE_TERMS = ("image", "x-ray", "xray", "radiograph", "film", "scan", "fundus", "slide", "photo", "findings",
               "show", "see", "visible", "opacity", "lesion")
MODALITY_TERMS = {
    "xray": ("chest", "x-ray", "xray", "radiograph", "lung", "pneumonia", "effusion", "cxr"),
    "fundus": ("fundus", "retina", "retinal", "eye", "optic", "diabetic retinopathy", "glaucoma"),
    "histopathology": ("biopsy", "histology", "histopathology", "slide", "tumour", "tumor", "carcinoma"),
}


@dataclass
class AgentConfig:
    use_retrieval: bool = True
    use_image: bool = True
    self_verify: bool = True
    max_iterations: int = 3
    top_k: int = 5

    @classmethod
    def variant(cls, name: str, **kw) -> AgentConfig:
        presets = {
            "full": {},
            "no_retrieval": {"use_retrieval": False, "self_verify": False},
            "single_pass": {"self_verify": False},
            "text_only": {"use_image": False},
        }
        if name not in presets:
            raise ValueError(f"unknown variant {name!r}")
        return cls(**{**presets[name], **kw})


class MedRAGAgent:
    def __init__(self, pipeline: RetrievalPipeline, settings: Settings | None = None, vlm=None, generator=None,
                 case_index=None):
        self.s = settings or get_settings()
        self.pipeline = pipeline
        self.vlm = vlm or (OfflineVLM() if self.s.offline else QwenVLM())
        self.generator = generator or (ExtractiveGenerator() if self.s.offline else LLMGenerator())
        kind = self.s.verifier or ("lexical" if self.s.offline else "nli")
        self.verifier = (NLIVerifier(self.s.nli_model, self.s.support_threshold) if kind == "nli"
                         else Verifier(self.s.support_threshold))
        self.case_index = case_index

    # ------------------------------------------------------------------ nodes
    def route(self, st: AgentState, cfg: AgentConfig) -> str:
        q = st.question.lower()
        if not st.modality:
            for m, terms in MODALITY_TERMS.items():
                if any(t in q for t in terms):
                    st.modality = m
                    break
        st.needs_image = bool(cfg.use_image and st.image_png is not None)
        st.queries.append(st.question)
        st.log("route", needs_image=st.needs_image, modality=st.modality,
               image_requested=any(t in q for t in IMAGE_TERMS))
        if cfg.use_retrieval:
            return "retrieve"
        return "inspect_image" if st.needs_image else "draft"

    def retrieve(self, st: AgentState, cfg: AgentConfig) -> str:
        query = st.queries[-1]
        res = self.pipeline.search(query, k=cfg.top_k, modality=st.modality, age_band=st.age_band)
        seen = {e["chunk_id"] for e in st.evidence}
        added = 0
        for h in res.hits:
            if h.chunk_id not in seen:
                st.evidence.append(h.to_dict())
                added += 1
        st.log("retrieve", query=query, hits=[h.chunk_id for h in res.hits], added=added,
               timings_ms={k: round(v, 2) for k, v in res.timings_ms.items()})
        if st.needs_image and not st.findings:
            return "inspect_image"
        return "draft"

    def inspect_image(self, st: AgentState, cfg: AgentConfig) -> str:
        t = time.perf_counter()
        try:
            st.findings = self.vlm.describe(st.image_png, st.modality or "unknown", st.question)
        except Exception as e:  # model server down / timeout: continue text-only, but never as "verified"
            st.findings = []
            st.errors.append(f"vision model failed: {e}")
            st.log("error", stage="inspect_image", error=str(e)[:200])
        if self.case_index is not None and st.modality:
            st.similar_cases = self.case_index.similar(st.image_png, st.modality, k=3, exclude=st.study_id)
        st.log("inspect_image", model=self.vlm.name, findings=[f.concept for f in st.findings],
               similar_cases=len(st.similar_cases), ms=round((time.perf_counter() - t) * 1000, 2))
        if cfg.use_retrieval and st.iterations == 0:
            # Image findings sharpen the evidence search before the first draft.
            abnormal = [f.concept for f in st.findings if not f.concept.startswith("normal")]
            # a normal study still needs evidence about what "normal" means for this modality
            concepts = abnormal or [f.concept for f in st.findings if f.concept != "unsupported modality"]
            if concepts:
                st.queries.append(st.question + " " + " ".join(concepts))
                return "retrieve"
        return "draft"

    def draft(self, st: AgentState, cfg: AgentConfig) -> str:
        st.iterations += 1
        try:
            st.draft = self.generator.draft(st.question, st.evidence, st.findings, st.similar_cases)
        except Exception as e:
            st.errors.append(f"drafting failed: {e}")
            st.log("error", stage="draft", error=str(e)[:200])
            st.draft = Draft([Claim(f.text, "findings", [], f.concept, "image") for f in st.findings],
                             "Drafting failed; image findings only. Requires clinician review.", "error")
        st.log("draft", iteration=st.iterations, claims=len(st.draft.claims), generator=st.draft.generator)
        return "verify"

    def verify(self, st: AgentState, cfg: AgentConfig) -> str:
        v = self.verifier.check(st.draft, st.evidence)
        st.verification = v
        st.log("verify", passed=v.passed, unsupported=v.unsupported, hallucination_rate=v.hallucination_rate)
        if st.errors:  # a degraded run is never presented as verified
            st.status = "needs_review"
            st.log("flag", reason="a model or service failed during this analysis")
            return "END"
        if v.passed:
            st.status = "verified"
            return "END"
        if not cfg.self_verify:
            st.status = "unverified"
            return "END"
        if not cfg.use_retrieval or st.iterations >= cfg.max_iterations:
            st.status = "needs_review"
            st.log("flag", reason="unsupported claims remain after bounded self-correction")
            return "END"
        st.queries.append(self.verifier.reformulate(st.question, v, st.queries))
        return "retrieve"

    # -------------------------------------------------------------------- run
    def run(self, question: str, *, modality: str | None = None, age_band: str | None = None,
            image_png: bytes | None = None, study_id: str | None = None,
            config: AgentConfig | None = None) -> AgentState:
        cfg = config or AgentConfig(max_iterations=self.s.max_iterations)
        st = AgentState(question=question, modality=modality, age_band=age_band, image_png=image_png,
                        study_id=study_id)
        nodes = {"route": self.route, "retrieve": self.retrieve, "inspect_image": self.inspect_image,
                 "draft": self.draft, "verify": self.verify}
        node, steps = "route", 0
        t0 = time.perf_counter()
        while node != "END":
            steps += 1
            if steps > 6 * (cfg.max_iterations + 2):  # hard guard against a runaway loop
                st.status = "needs_review"
                st.log("abort", reason="step budget exhausted")
                break
            node = nodes[node](st, cfg)
        st.log("end", status=st.status, total_ms=round((time.perf_counter() - t0) * 1000, 2))
        return st

    def ask(self, question: str, modality: str | None = None, k: int = 5) -> dict:
        res = self.pipeline.search(question, k=k, modality=modality)
        evidence = [h.to_dict() for h in res.hits]
        out = self.generator.answer(question, evidence)
        return {**out, "evidence": evidence}


def build_langgraph(agent: MedRAGAgent, cfg: AgentConfig):  # pragma: no cover - optional dependency
    """Same control flow compiled with LangGraph (``pip install langgraph``)."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(dict)

    def wrap(fn):
        def _node(state: dict) -> dict:
            st = state["st"]
            return {"st": st, "next": fn(st, cfg)}
        return _node

    for name in ("route", "retrieve", "inspect_image", "draft", "verify"):
        g.add_node(name, wrap(getattr(agent, name)))
        g.add_conditional_edges(name, lambda s: s["next"] if s["next"] != "END" else END)
    g.set_entry_point("route")
    return g.compile()
