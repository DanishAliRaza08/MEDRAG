import pytest

from medrag.agent import AgentConfig, MedRAGAgent
from medrag.agent.generator import ExtractiveGenerator, _negated
from medrag.agent.state import Claim, Draft
from medrag.agent.verifier import Verifier
from medrag.config import ROOT_DIR, Settings
from medrag.imaging import synthetic

S = Settings(mode="offline", kb_dirs=[ROOT_DIR / "data" / "knowledge_base"])


def png(mod="xray", finding="consolidation", seed=1):
    return synthetic.to_png_bytes(synthetic.GENERATORS[mod](finding, seed=seed))


class HallucinatingGenerator(ExtractiveGenerator):
    """Adds an unsupported claim until the agent has retrieved evidence about 'sarcoidosis'."""

    name = "hallucinating"

    def draft(self, question, evidence, findings, similar):
        d = super().draft(question, evidence, findings, similar)
        if not any("sarcoidosis" in e["text"].lower() for e in evidence):
            d.claims.append(Claim("Bilateral hilar lymphadenopathy suggests sarcoidosis.", "impression", [1],
                                  "sarcoidosis"))
        return d


class AlwaysWrongGenerator(ExtractiveGenerator):
    def draft(self, *a):
        d = super().draft(*a)
        d.claims.append(Claim("The moon is made of cheese.", "impression", [], "lunar dairy"))
        return d


def nodes(state):
    return [t["node"] for t in state.trace]


def test_full_flow_order_and_verified(pipeline):
    st = MedRAGAgent(pipeline, S).run("cough and fever, review chest x-ray", modality="xray", image_png=png())
    assert nodes(st)[:4] == ["route", "retrieve", "inspect_image", "retrieve"]
    assert nodes(st)[-3:] == ["draft", "verify", "end"]
    assert st.status == "verified" and st.verification.hallucination_rate == 0.0
    assert any(c.section == "impression" and c.citations for c in st.draft.claims)


def test_every_non_image_claim_cites_valid_evidence(pipeline):
    st = MedRAGAgent(pipeline, S).run("diabetic screening", modality="fundus",
                                      image_png=png("fundus", "diabetic_retinopathy"))
    for c in st.draft.claims:
        if c.source != "image":
            assert c.citations and all(1 <= i <= len(st.evidence) for i in c.citations)


def test_no_image_skips_inspection(pipeline):
    st = MedRAGAgent(pipeline, S).run("how is pneumonia severity assessed", modality="xray")
    assert "inspect_image" not in nodes(st) and st.findings == []


def test_text_only_variant_ignores_image(pipeline):
    st = MedRAGAgent(pipeline, S).run("review", modality="xray", image_png=png(),
                                      config=AgentConfig.variant("text_only"))
    assert "inspect_image" not in nodes(st)


def test_no_retrieval_baseline_is_ungrounded(pipeline):
    st = MedRAGAgent(pipeline, S).run("review", modality="xray", image_png=png(),
                                      config=AgentConfig.variant("no_retrieval"))
    assert "retrieve" not in nodes(st) and st.evidence == []
    assert st.verification.hallucination_rate > 0.5 and st.status == "unverified"


def test_self_verification_loop_recovers():
    from medrag.retrieval import RetrievalPipeline
    from medrag.retrieval.chunking import Document

    pipeline = RetrievalPipeline(S)
    pipeline.ingest_directory()
    pipeline.add_documents([Document("kb-sarc", "Sarcoidosis", "Sarcoidosis is a granulomatous disease in which "
                                     "bilateral hilar lymphadenopathy is typical.", modality=["xray"])])
    agent = MedRAGAgent(pipeline, S, generator=HallucinatingGenerator())
    st = agent.run("review chest x-ray", modality="xray", image_png=png())
    assert st.iterations == 2 and st.status == "verified"
    assert "sarcoidosis" in st.queries[-1]
    assert nodes(st).count("verify") == 2


def test_single_pass_does_not_loop(pipeline):
    agent = MedRAGAgent(pipeline, S, generator=AlwaysWrongGenerator())
    st = agent.run("review", modality="xray", image_png=png(), config=AgentConfig.variant("single_pass"))
    assert st.iterations == 1 and st.status == "unverified"


@pytest.mark.parametrize("max_it", [1, 2, 3])
def test_loop_is_bounded_and_flags_for_review(pipeline, max_it):
    agent = MedRAGAgent(pipeline, S, generator=AlwaysWrongGenerator())
    st = agent.run("review", modality="xray", image_png=png(), config=AgentConfig(max_iterations=max_it))
    assert st.iterations == max_it and st.status == "needs_review"
    assert nodes(st)[-2] == "flag"


def test_verifier_flags_uncited_and_mismatched_claims():
    ev = [{"text": "A pleural effusion causes blunting of the costophrenic angle."}]
    d = Draft([Claim("finding", "findings", source="image"),
               Claim("A pleural effusion causes blunting.", "impression", [1]),
               Claim("Uncited statement about effusion.", "impression", []),
               Claim("Glaucoma needs eye drops.", "recommendations", [1], "glaucoma")])
    v = Verifier(0.5).check(d, ev)
    assert (v.supported, v.unsupported) == (2, 2) and v.missing_concepts == ["glaucoma"]
    assert v.hallucination_rate == 0.5 and not v.passed


def test_reformulate_targets_missing_concepts():
    from medrag.agent.state import Verification

    q = Verifier.reformulate("q", Verification(1, 1, ["x"], 0.5, False, ["pleural effusion"]), ["q"])
    assert q.startswith("pleural effusion")


def test_negation_detection():
    assert _negated("Benign tissue shows no nuclear pleomorphism.", "nuclear pleomorphism")
    assert not _negated("Carcinoma shows nuclear pleomorphism.", "nuclear pleomorphism")


def test_route_infers_modality(pipeline):
    st = MedRAGAgent(pipeline, S).run("diabetic retinopathy referral criteria")
    assert st.modality == "fundus"


def test_variant_presets():
    assert AgentConfig.variant("no_retrieval").use_retrieval is False
    assert AgentConfig.variant("single_pass").self_verify is False
    with pytest.raises(ValueError):
        AgentConfig.variant("nope")


def test_ask_returns_cited_answer(pipeline):
    out = MedRAGAgent(pipeline, S).ask("When should a repeat chest radiograph be done after pneumonia?")
    assert out["citations"] and "[" in out["answer"]


class BrokenVLM:
    name = "broken"

    def describe(self, *a, **k):
        raise TimeoutError("model server timed out")


class BrokenGenerator(ExtractiveGenerator):
    def draft(self, *a, **k):
        raise ConnectionError("LLM endpoint unavailable")


@pytest.mark.parametrize("kw", [{"vlm": BrokenVLM()}, {"generator": BrokenGenerator()}])
def test_model_failure_degrades_to_needs_review_never_verified(pipeline, kw):
    st = MedRAGAgent(pipeline, S, **kw).run("review", modality="xray", image_png=png())
    assert st.status == "needs_review" and st.errors
    assert "error" in nodes(st) and nodes(st)[-1] == "end"
