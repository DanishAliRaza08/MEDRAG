import io

import numpy as np
import pytest
from PIL import Image

from medrag.deid import age_band, contains_phi, deidentify_image, pseudonymize, scrub_text
from medrag.imaging import synthetic


@pytest.mark.parametrize("text,label", [
    ("Patient: John Smith presented", "NAME"),
    ("MRN: 00123456", "MRN"),
    ("seen on 12/03/2024", "DATE"),
    ("born 3 March 1958", "DATE"),
    ("call 0300-1234567", "PHONE"),
    ("email john.smith@example.com", "EMAIL"),
    ("CNIC 35202-1234567-1", "CNIC"),
    ("a 94 year old man", "AGE"),
    ("lives at 12 Canal Road", "ADDRESS"),
])
def test_scrub_text_categories(text, label):
    out, counts = scrub_text(text)
    assert label in counts and f"[{label}]" in out


def test_scrub_keeps_clinical_content():
    out, counts = scrub_text("Cough and fever for 3 days, right lower zone crackles.")
    assert counts == {} and "crackles" in out
    assert not contains_phi("pleural effusion on the left")


def test_pseudonym_stable_irreversible_and_salted():
    a = pseudonymize("mrn-001", "s1")
    assert a == pseudonymize(" MRN-001 ", "s1")
    assert a != pseudonymize("mrn-001", "s2")
    assert "001" not in a and a.startswith("pt_")
    with pytest.raises(ValueError):
        pseudonymize("  ", "s")


@pytest.mark.parametrize("age,band", [(5, "0-17"), (30, "18-39"), (50, "40-64"), (95, "65+"), (None, "unknown")])
def test_age_band(age, band):
    assert age_band(age) == band


def test_image_metadata_removed():
    png = synthetic.to_png_bytes(synthetic.chest_xray(seed=1), PatientName="John Doe", PatientID="00123456")
    res = deidentify_image(png, use_ocr=False)
    assert "PatientName" in res.metadata_removed
    assert b"John Doe" not in res.png and b"00123456" not in res.png
    assert "PatientName" not in Image.open(io.BytesIO(res.png)).info


@pytest.mark.parametrize("modality,finding", [("xray", "consolidation"), ("fundus", "diabetic_retinopathy"),
                                              ("histopathology", "carcinoma"), ("histopathology", "benign")])
def test_burned_in_text_removed_without_false_positives(modality, finding):
    gen = synthetic.GENERATORS[modality]
    clean = deidentify_image(synthetic.to_png_bytes(gen(finding, seed=4)), use_ocr=False)
    assert clean.masked_regions == []  # no false positives on text-free images
    burned = gen(finding, seed=4, burn_text=True)
    res = deidentify_image(synthetic.to_png_bytes(burned), use_ocr=False)
    assert res.masked_regions
    out = np.asarray(Image.open(io.BytesIO(res.png)).convert("L")).astype(int)
    orig = np.asarray(burned.convert("L")).astype(int)
    base = np.asarray(gen(finding, seed=4).convert("L")).astype(int)
    glyphs = (np.abs(orig - base) > 40) & ((orig > 200) | (orig < 60))
    if modality == "histopathology":
        glyphs &= orig < 60  # black text on a white label
    assert glyphs.sum() > 50
    assert ((out[glyphs]) == 0).all()  # every glyph pixel blacked out


def test_deid_preserves_diagnostic_content():
    from medrag.imaging.analysis import analyze

    res = deidentify_image(synthetic.to_png_bytes(synthetic.chest_xray("effusion", seed=2, burn_text=True)), False)
    concepts = [f.concept for f in analyze(Image.open(io.BytesIO(res.png)), "xray")]
    assert concepts == ["pleural effusion"]


def test_invalid_image_raises():
    with pytest.raises(Exception):  # noqa: B017
        deidentify_image(b"not an image")
