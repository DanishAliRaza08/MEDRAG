"""Real evaluation data: the Indiana University chest X-ray collection (IU X-ray / OpenI).

The collection pairs real chest radiographs with the radiologists' reports (already de-identified by the
providers: identifiers appear as "XXXX"). It is downloaded on demand from a Hugging Face mirror of the public
collection; images are NOT committed to the repository. Every image is still passed through MedRAG's own
de-identification before it is written to disk, exactly as an uploaded study would be.

A rule-based labeller with negation handling (in the spirit of the CheXpert/NegEx labellers) turns each report into
finding labels, so a generated report can be scored for clinical correctness, not only for wording.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .deid import deidentify_image, scrub_text
from .textutils import asserts

HF_REPO = "MLforHealthcare/Indiana_University_Chest_X-ray_Collection"
HF_FILE = "data/test-00000-of-00001.parquet"

# concept -> phrases that assert it (matched on word boundaries, lower-case)
CONCEPTS: dict[str, list[str]] = {
    "pleural effusion": ["pleural effusion", "effusion", "effusions", "blunting of the costophrenic",
                         "blunted costophrenic", "pleural fluid"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged", "heart size is enlarged",
                     "enlarged cardiac silhouette", "cardiac enlargement", "heart size is mildly enlarged",
                     "mildly enlarged heart", "heart is mildly enlarged"],
    "consolidation": ["consolidation", "airspace disease", "airspace opacity", "airspace opacities", "infiltrate",
                      "infiltrates", "pneumonia"],
    "pneumothorax": ["pneumothorax", "pneumothoraces"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "nodule": ["nodule", "nodules", "mass"],
}
def label_report(text: str) -> dict[str, int]:
    """Return {concept: 1 present, 0 absent/negated} for every concept (uncertain mentions count as present)."""
    out = {c: 0 for c in CONCEPTS}
    for concept, phrases in CONCEPTS.items():
        out[concept] = int(any(asserts(text, ph) for ph in phrases))
    return out


def is_normal(labels: dict[str, int]) -> bool:
    return not any(labels.values())


def split_report(report: str) -> tuple[str, str]:
    m = re.search(r"FINDINGS:(.*?)(?:IMPRESSION:|$)(.*)", report, re.S)
    if not m:
        return "", report.strip()
    return m.group(1).strip(" ."), m.group(2).strip(" .")


def prepare_iu_xray(out_dir: Path, n: int = 100, seed: int = 0, log=print) -> list[dict]:
    """Download the IU X-ray test split and write ``n`` de-identified cases + ``cases.jsonl`` to ``out_dir``."""
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    table = pq.read_table(path)
    idx = list(range(table.num_rows))
    random.Random(seed).shuffle(idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, seen_reports = [], set()
    for i in idx:
        if len(cases) >= n:
            break
        row = table.slice(i, 1).to_pylist()[0]
        report = row["report"] or ""
        findings, impression = split_report(report)
        if not impression or report in seen_reports:
            continue  # skip empty reports and duplicate views of the same study
        seen_reports.add(report)
        deid = deidentify_image(row["image"]["bytes"], use_ocr=False)
        case_id = f"iu{i:04d}"
        (out_dir / f"{case_id}.png").write_bytes(deid.png)
        reference = scrub_text(f"{findings} {impression}".strip())[0]
        labels = label_report(reference)
        cases.append({"id": case_id, "image": f"{case_id}.png", "modality": "xray", "findings": findings,
                      "impression": impression, "reference": reference, "labels": labels,
                      "normal": is_normal(labels), "masked_regions": len(deid.masked_regions)})
    with open(out_dir / "cases.jsonl", "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    stats = {k: sum(c["labels"][k] for c in cases) for k in CONCEPTS}
    log(f"wrote {len(cases)} cases to {out_dir}; positives per finding: {stats}; "
        f"normal: {sum(c['normal'] for c in cases)}")
    (out_dir / "README.md").write_text(
        "# IU X-ray evaluation subset\n\nIndiana University chest X-ray collection (Demner-Fushman et al., J Am Med "
        "Inform Assoc 2016; distributed by the U.S. National Library of Medicine via Open-i), downloaded from the "
        f"Hugging Face mirror `{HF_REPO}`. Used for research evaluation only; do not redistribute. Images were "
        "passed through MedRAG de-identification.\n")
    return cases


def load_cases(path: Path, n: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in (path / "cases.jsonl").read_text().splitlines() if line.strip()]
    return cases[:n] if n else cases


def image_bytes(path: Path, case: dict) -> bytes:
    return (path / case["image"]).read_bytes()
