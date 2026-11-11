"""PubMed Central corpus builder, report labeller and NLI verifier (no network or model weights needed)."""

import json

import pytest

from medrag.agent.state import Claim, Draft
from medrag.agent.verifier import NLIVerifier
from medrag.config import ROOT_DIR
from medrag.datasets import is_normal, label_report, split_report
from medrag.kb import pmc
from medrag.retrieval.chunking import load_directory

JATS = b"""<?xml version="1.0"?>
<pmc-articleset><article xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:ali="http://www.niso.org/schemas/ali/1.0/">
<front><journal-meta><journal-title-group><journal-title>Chest Reviews</journal-title></journal-title-group></journal-meta>
<article-meta><title-group><article-title>Pleural effusion: a review</article-title></title-group>
<contrib-group><contrib contrib-type="author"><name><surname>Khan</surname><given-names>Ayesha</given-names></name></contrib>
<contrib contrib-type="author"><name><surname>Lee</surname><given-names>Bo</given-names></name></contrib></contrib-group>
<pub-date><year>2023</year></pub-date>
<permissions><license><ali:license_ref>https://creativecommons.org/licenses/by/4.0/</ali:license_ref>
<license-p>Open access.</license-p></license></permissions>
<abstract><p>Pleural effusion is common and has many causes that need a structured approach.</p></abstract>
</article-meta></front>
<body><sec><title>Diagnosis</title>
<p>Thoracic ultrasound is recommended before aspiration <xref ref-type="bibr">[12]</xref> because it reduces complications and guides sampling safely.</p>
<table-wrap><table><tr><td>should not appear</td></tr></table></table-wrap></sec>
<sec><title>Funding</title><p>This work received funding from a charity and the authors thank everybody involved.</p></sec>
</body></article></pmc-articleset>"""


def test_jats_parse_keeps_text_after_citations_and_skips_admin_sections():
    a = pmc.parse(JATS, "123")
    assert a.title == "Pleural effusion: a review" and a.journal == "Chest Reviews" and a.year == "2023"
    assert a.authors == "Khan A et al." and a.license == "CC BY 4.0"
    # regression: text AFTER a citation used to be dropped (Element.clear() wipes the tail)
    assert "because it reduces complications and guides sampling safely." in a.body
    assert "[12]" not in a.body and "should not appear" not in a.body and "charity" not in a.body


def test_jats_rejects_non_cc_by_licences():
    nc = JATS.replace(b"licenses/by/4.0", b"licenses/by-nc/4.0")
    assert pmc.parse(nc, "1") is None


def test_markdown_front_matter_roundtrip(tmp_path):
    md = pmc.to_markdown(pmc.parse(JATS, "123"), "xray", "effusion")
    (tmp_path / "pmc123.md").write_text(md)
    (tmp_path / "ATTRIBUTION.md").write_text("# Attribution\nnot evidence")
    docs = load_directory(tmp_path)
    assert [d.doc_id for d in docs] == ["pmc-123"]  # upper-case files are not indexed
    assert docs[0].modality == ["xray"] and "PMC123" in docs[0].source and "CC BY 4.0" in docs[0].source
    assert "Diagnosis." not in docs[0].text  # headings are structure, not sentences


def test_committed_pmc_corpus_is_cc_by_and_attributed():
    folder = ROOT_DIR / "data" / "knowledge_base_pmc"
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    assert len(manifest) >= 25
    assert all("creativecommons.org/licenses/by/" in m["license_url"] for m in manifest)
    attribution = (folder / "ATTRIBUTION.md").read_text()
    assert all(f"PMC{m['pmcid']}" in attribution for m in manifest)
    assert {m["modality"] for m in manifest} == {"xray", "fundus", "histopathology"}


def test_pmc_gold_set_answers_are_in_the_relevant_articles():
    gold = json.loads((ROOT_DIR / "data" / "gold" / "retrieval_gold_pmc.json").read_text())
    for item in gold["items"]:
        doc = ROOT_DIR / "data" / "knowledge_base_pmc" / (item["relevant"][0].replace("pmc-", "pmc") + ".md")
        assert item["check"].lower() in doc.read_text().lower(), item["query"]


@pytest.mark.parametrize("text,expected", [
    ("No pleural effusion or pneumothorax. Heart size normal.", set()),
    ("The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax.",
     set()),
    ("Small volume effusions and basilar airspace disease.", {"pleural effusion", "consolidation"}),
    ("Mild cardiomegaly. Previously identified small right pneumothorax is not visualized.", {"cardiomegaly"}),
    ("Stable 1 cm right apical pneumothorax. Minimal atelectasis left base.", {"pneumothorax", "atelectasis"}),
    ("Calcified XXXX nodule in the right upper lobe.", {"nodule"}),
])
def test_report_labeller_handles_negation(text, expected):
    labels = label_report(text)
    assert {k for k, v in labels.items() if v} == expected
    assert is_normal(labels) == (not expected)


def test_split_report():
    assert split_report("FINDINGS: Heart normal. IMPRESSION: No acute disease.") == ("Heart normal", "No acute disease")


class StubNLI(NLIVerifier):
    """Pretends to be an NLI model: entailed if the hypothesis is a substring, contradicted if it adds a 'not'."""

    def probabilities(self, premises, hypothesis):
        out = []
        for p in premises:
            if " not " in hypothesis and hypothesis.replace(" not", "") in p:
                out.append((0.01, 0.97))
            elif hypothesis.rstrip(".") in p or "ultrasound" in hypothesis.lower() and "ultrasound" in p.lower():
                out.append((0.93, 0.01))
            else:
                out.append((0.05, 0.20))
        return out


def test_nli_verifier_accepts_paraphrase_rejects_contradiction():
    ev = [{"text": "Thoracic ultrasound is recommended to confirm a pleural effusion. Light's criteria distinguish "
                   "transudate from exudate. Aspiration is guided by imaging. Fluid is sent for cytology."}]
    v = StubNLI()
    para = Claim("Chest ultrasound should be used to confirm fluid around the lung.", "impression", [1])
    contra = Claim("Thoracic ultrasound is not recommended to confirm a pleural effusion.", "impression", [1])
    uncited = Claim("Thoracic ultrasound is recommended to confirm a pleural effusion.", "impression", [])
    image = Claim("Blunted left costophrenic angle.", "findings", source="image")
    res = v.check(Draft([para, contra, uncited, image]), ev)
    assert (para.supported, contra.supported, uncited.supported, image.supported) == (True, False, False, True)
    assert contra.support_score < 0  # explicit contradiction is marked with a negative score
    assert res.unsupported == 2 and not res.passed


def test_nli_windows_split_long_passages():
    text = " ".join(f"Sentence number {i} is here." for i in range(7))
    w = NLIVerifier._windows(text, 3)
    assert len(w) == 5 and all(len(x.split(". ")) == 3 for x in w)
