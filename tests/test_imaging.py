import pytest

from medrag.imaging import synthetic
from medrag.imaging.analysis import analyze
from medrag.imaging.encoder import CaseIndex, ThumbnailEncoder
from medrag.imaging.vlm import OfflineVLM


@pytest.mark.parametrize("modality,finding,concept", [
    ("xray", "normal", "normal chest radiograph"),
    ("xray", "consolidation", "consolidation"),
    ("xray", "effusion", "pleural effusion"),
    ("xray", "cardiomegaly", "cardiomegaly"),
    ("xray", "pneumothorax", "pneumothorax"),
    ("fundus", "normal", "normal fundus"),
    ("fundus", "diabetic_retinopathy", "diabetic retinopathy"),
    ("fundus", "glaucoma", "optic disc cupping"),
    ("histopathology", "benign", "benign tissue"),
    ("histopathology", "carcinoma", "carcinoma"),
])
def test_offline_vlm_findings(modality, finding, concept):
    for seed in (0, 5):
        found = [f.concept for f in analyze(synthetic.GENERATORS[modality](finding, seed=seed), modality)]
        assert concept in found


def test_unknown_modality():
    f = OfflineVLM().describe(synthetic.to_png_bytes(synthetic.chest_xray()), "ct")
    assert f[0].concept == "unsupported modality"


def test_similar_case_index_finds_same_pathology():
    idx = CaseIndex(ThumbnailEncoder())
    for i, f in enumerate(["consolidation", "cardiomegaly", "pneumothorax"]):
        idx.add(f"s{i}", synthetic.to_png_bytes(synthetic.chest_xray(f, seed=i)), "xray", "40-64", f)
    idx.add("f0", synthetic.to_png_bytes(synthetic.fundus()), "fundus", "40-64", "normal")
    hits = idx.similar(synthetic.to_png_bytes(synthetic.chest_xray("cardiomegaly", seed=9)), "xray", k=3)
    assert hits[0]["finding_summary"] == "cardiomegaly"
    assert all(h["modality"] == "xray" for h in hits)
