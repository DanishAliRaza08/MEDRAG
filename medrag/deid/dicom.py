"""DICOM input handling with header de-identification.

MedRAG never stores a DICOM header: it renders the pixel data to an 8-bit image, which then goes through the same
pixel-level de-identification as any upload. This module (1) renders the pixels correctly (rescale slope/intercept,
VOI LUT / windowing, MONOCHROME1 inversion, first frame of multi-frame objects) and (2) reports which identifying
attributes of the DICOM PS3.15 Basic Application Level Confidentiality Profile were present and therefore dropped,
so the audit trail records what was removed. ``deidentified_dataset`` additionally produces a header-scrubbed copy
for workflows that must keep DICOM.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Subset of PS3.15 Annex E Table E.1-1 (Basic Profile): attributes that directly or indirectly identify the patient,
# the staff or the institution.
IDENTIFYING_KEYWORDS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientSex", "PatientAge", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName", "MedicalRecordLocator",
    "EthnicGroup", "Occupation", "AdditionalPatientHistory", "PatientComments", "MilitaryRank", "BranchOfService",
    "AccessionNumber", "StudyID", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime",
    "SeriesTime", "AcquisitionTime", "ContentTime", "InstitutionName", "InstitutionAddress",
    "InstitutionalDepartmentName", "ReferringPhysicianName", "PerformingPhysicianName", "NameOfPhysiciansReadingStudy",
    "OperatorsName", "RequestingPhysician", "StationName", "DeviceSerialNumber", "StudyInstanceUID",
    "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID", "StudyDescription", "SeriesDescription",
    "RequestedProcedureDescription", "PerformedProcedureStepDescription", "ImageComments", "RequestAttributesSequence",
]


def is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def _render(ds) -> Image.Image:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    arr = ds.pixel_array
    if getattr(ds, "NumberOfFrames", 1) and int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # colour (e.g. secondary capture, fundus, pathology)
        return Image.fromarray(arr[..., :3].astype(np.uint8), "RGB")
    arr = apply_modality_lut(arr, ds)
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:  # pragma: no cover - malformed VOI
        pass
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip((arr - lo) / max(1e-6, hi - lo), 0, 1) * 255
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 255 - arr  # MONOCHROME1 stores inverted grey levels
    return Image.fromarray(arr.astype(np.uint8), "L")


def load_dicom(data: bytes) -> tuple[Image.Image, list[str], bool]:
    """Return (image, removed attribute names, burned_in_flag)."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data))
    present = [k for k in IDENTIFYING_KEYWORDS if k in ds and str(ds.data_element(k).value).strip()]
    private = any(el.tag.is_private for el in ds)
    burned = str(getattr(ds, "BurnedInAnnotation", "")).upper() == "YES"
    removed = [f"dicom:{k}" for k in present] + (["dicom:private-tags"] if private else [])
    return _render(ds), removed, burned


def deidentified_dataset(data: bytes, pseudonym: str = "ANONYMOUS"):
    """A header-scrubbed copy of the dataset (identifying attributes blanked, private tags removed)."""
    import pydicom
    from pydicom.uid import generate_uid

    ds = pydicom.dcmread(io.BytesIO(data))
    for k in IDENTIFYING_KEYWORDS:
        if k in ds:
            if k.endswith("UID"):
                setattr(ds, k, generate_uid())
            elif ds.data_element(k).VR == "SQ":
                delattr(ds, k)
            else:
                setattr(ds, k, "")
    ds.PatientName = pseudonym
    ds.PatientID = pseudonym
    ds.remove_private_tags()
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "MedRAG PS3.15 basic profile subset"
    return ds
