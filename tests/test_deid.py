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
    # regression: words that merely start with "id"/"acc" are not identifiers
    for text in ("No cardiopulmonary abnormality identified.", "Accompanied by mild atelectasis.",
                 "Accession of fluid is not seen."):
        assert scrub_text(text)[1] == {}, text
    assert "MRN" in scrub_text("Accession 12345678")[1] and "MRN" in scrub_text("ID: AB12345")[1]
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


def _make_dicom(img, photometric="MONOCHROME2", **tags):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    arr = np.asarray(img.convert("L"), dtype=np.uint16) * 16  # 12-bit range, like a real radiograph
    if photometric == "MONOCHROME1":
        arr = 4095 - arr
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = SecondaryCaptureImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    for k, v in tags.items():
        setattr(ds, k, v)
    ds.Modality, ds.Rows, ds.Columns = "CR", arr.shape[0], arr.shape[1]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = arr.tobytes()
    ds.add_new(0x00091001, "LO", "private vendor note: John Doe")
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


PHI_TAGS = dict(PatientName="Doe^John", PatientID="MRN00123456", PatientBirthDate="19600102",
                InstitutionName="St Mary Hospital", ReferringPhysicianName="Smith^Anna", AccessionNumber="ACC987654",
                BurnedInAnnotation="YES")


@pytest.mark.parametrize("photometric", ["MONOCHROME2", "MONOCHROME1"])
def test_dicom_header_and_pixels_deidentified(photometric):
    pytest.importorskip("pydicom")
    from medrag.imaging.analysis import analyze

    dcm = _make_dicom(synthetic.chest_xray("effusion", seed=3, burn_text=True), photometric, **PHI_TAGS)
    res = deidentify_image(dcm, use_ocr=False)
    for needle in (b"Doe^John", b"MRN00123456", b"19600102", b"St Mary", b"Smith^Anna", b"ACC987654", b"John Doe"):
        assert needle not in res.png
    assert {"dicom:PatientName", "dicom:PatientID", "dicom:PatientBirthDate", "dicom:InstitutionName",
            "dicom:private-tags", "dicom:burned-in-annotation-flag"} <= set(res.metadata_removed)
    assert res.masked_regions  # burned-in text found in the DICOM pixels
    # grey levels rendered the right way round (MONOCHROME1 inverted back): strongly positively correlated with the
    # original radiograph outside the masked text regions
    out = np.asarray(Image.open(io.BytesIO(res.png)).convert("L"), dtype=np.float32)
    ref = np.asarray(synthetic.chest_xray("effusion", seed=3).convert("L"), dtype=np.float32)
    keep = out > 0
    assert np.corrcoef(out[keep], ref[keep])[0, 1] > 0.98
    assert "pleural effusion" in [f.concept for f in analyze(Image.open(io.BytesIO(res.png)), "xray")]


def test_deidentified_dicom_copy():
    pytest.importorskip("pydicom")
    import pydicom

    from medrag.deid.dicom import deidentified_dataset

    dcm = _make_dicom(synthetic.chest_xray(seed=1), **PHI_TAGS)
    original_uid = pydicom.dcmread(io.BytesIO(dcm)).StudyInstanceUID
    ds = deidentified_dataset(dcm, "pt_abc")
    assert ds.PatientName == "pt_abc" and ds.PatientBirthDate == "" and ds.InstitutionName == ""
    assert ds.StudyInstanceUID != original_uid and ds.PatientIdentityRemoved == "YES"
    assert not any(el.tag.is_private for el in ds)
