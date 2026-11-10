"""Deterministic image analysis used by the *offline* vision stand-in.

These heuristics are tuned to the synthetic images in ``synthetic.py``; on a
real image they only provide a rough, clearly-labelled placeholder. In real
mode, Qwen2.5-VL replaces this module entirely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass
class Finding:
    text: str
    concept: str
    location: str = ""
    confidence: float = 0.5
    source: str = "image"

    def to_dict(self) -> dict:
        return asdict(self)


def _components(mask: np.ndarray, min_size: int = 3) -> list[int]:
    """Sizes of 4-connected components in a boolean mask (pure Python BFS)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist(), strict=True):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        n = 0
        while q:
            y, x = q.popleft()
            n += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if n >= min_size:
            sizes.append(n)
    return sizes


def _zone(a: np.ndarray, x0, x1, y0, y1) -> np.ndarray:
    h, w = a.shape[:2]
    return a[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def analyze_chest_xray(img: Image.Image) -> list[Finding]:
    a = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    out: list[Finding] = []
    # Patient right is on the image left by radiographic convention.
    lower_r, lower_l = _zone(a, 0.17, 0.36, 0.56, 0.69).mean(), _zone(a, 0.64, 0.83, 0.56, 0.69).mean()
    base_r, base_l = _zone(a, 0.17, 0.36, 0.73, 0.83).mean(), _zone(a, 0.64, 0.83, 0.73, 0.83).mean()
    upper_r, upper_l = _zone(a, 0.17, 0.36, 0.28, 0.45), _zone(a, 0.64, 0.83, 0.28, 0.45)

    for side, base_a, base_b in (("right", base_r, base_l), ("left", base_l, base_r)):
        if base_a - base_b > 25:
            out.append(Finding(
                f"Homogeneous basal opacity with blunting of the {side} costophrenic angle, "
                "in keeping with a pleural effusion.",
                "pleural effusion", f"{side} base", 0.8))
    for side, lo_a, lo_b, ba, bb in (("right", lower_r, lower_l, base_r, base_l),
                                     ("left", lower_l, lower_r, base_l, base_r)):
        if lo_a - lo_b > 20 and not (ba - bb > 25):
            out.append(Finding(
                f"Patchy airspace opacity in the {side} lower zone, suggestive of consolidation.",
                "consolidation", f"{side} lower zone", 0.75))
    for side, up in (("right", upper_r), ("left", upper_l)):
        if (up < 22).mean() > 0.3:
            out.append(Finding(
                f"Hyperlucent {side} apical region devoid of lung markings, suspicious for a pneumothorax.",
                "pneumothorax", f"{side} apex", 0.7))

    row = a[int(0.74 * 256)]
    body = np.nonzero(row > 70)[0]
    heart = np.nonzero(row > 152)[0]
    if len(body) and len(heart):
        # widest contiguous bright run around the centre is the heart shadow
        centre = 128
        runs = np.split(heart, np.nonzero(np.diff(heart) > 1)[0] + 1)
        run = max(runs, key=lambda r: len(r) if r[0] <= centre + 20 and r[-1] >= centre - 20 else 0)
        ctr = (run[-1] - run[0]) / max(1, body[-1] - body[0])
        if ctr > 0.5:
            out.append(Finding(
                f"Enlarged cardiac silhouette (estimated cardiothoracic ratio {ctr:.2f}), "
                "consistent with cardiomegaly.",
                "cardiomegaly", "cardiac silhouette", 0.7))
    if not out:
        out.append(Finding("No focal consolidation, pleural effusion or pneumothorax identified; "
                           "cardiac silhouette within normal limits.", "normal chest radiograph", "", 0.6))
    return out


def analyze_fundus(img: Image.Image) -> list[Finding]:
    a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    yy, xx = np.mgrid[0:256, 0:256]
    inner = (yy - 128) ** 2 + (xx - 128) ** 2 < (0.40 * 256) ** 2  # ignore the retina rim
    out: list[Finding] = []
    haem = inner & (R < 120) & (R > 50) & (G < 30)
    n_haem = len(_components(haem, 2))
    exud = (R > 210) & (G > 190) & (B < 140)
    n_exud = len(_components(exud, 1))
    if n_haem >= 3:
        out.append(Finding(f"Multiple ({n_haem}) dot-blot retinal haemorrhages / microaneurysms.",
                           "retinal haemorrhages", "posterior pole", 0.75))
    if n_exud >= 3:
        out.append(Finding(f"Hard exudates ({n_exud} foci) near the macula.", "hard exudates", "macula", 0.7))
    disc = (R > 235) & (G > 215) & (B > 170)
    cup = (R > 250) & (G > 245) & (B > 230)
    if disc.sum() > 30:
        cdr = float(np.sqrt(cup.sum() / disc.sum()))
        if cdr >= 0.6:
            out.append(Finding(f"Enlarged optic cup with cup-to-disc ratio of approximately {cdr:.1f}, "
                               "suspicious for glaucomatous optic neuropathy.", "optic disc cupping", "optic disc",
                               0.65))
    if out and any(f.concept in ("retinal haemorrhages", "hard exudates") for f in out):
        out.append(Finding("Pattern of haemorrhages and exudates consistent with diabetic retinopathy.",
                           "diabetic retinopathy", "posterior pole", 0.7))
    if not out:
        out.append(Finding("Optic disc, vessels and macula appear unremarkable; no haemorrhages or exudates.",
                           "normal fundus", "", 0.6))
    return out


def analyze_histopathology(img: Image.Image) -> list[Finding]:
    a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    nuclei = (R < 140) & (B > R) & (G < 90)
    frac = float(nuclei.mean())
    sizes = _components(nuclei, 3)
    cv = float(np.std(sizes) / np.mean(sizes)) if len(sizes) > 2 else 0.0
    if frac > 0.18 and cv > 0.35:
        return [
            Finding(f"Densely crowded nuclei (nuclear area fraction {frac:.2f}) with marked variation in nuclear "
                    "size (nuclear pleomorphism).", "nuclear pleomorphism", "", 0.7),
            Finding("Architecture suggestive of an invasive carcinoma; histological grading requires "
                    "pathologist review.", "carcinoma", "", 0.6),
        ]
    return [Finding(f"Evenly spaced, uniform nuclei (nuclear area fraction {frac:.2f}) without pleomorphism; "
                    "no features of malignancy.", "benign tissue", "", 0.6)]


ANALYZERS = {"xray": analyze_chest_xray, "fundus": analyze_fundus, "histopathology": analyze_histopathology}


def analyze(img: Image.Image, modality: str) -> list[Finding]:
    fn = ANALYZERS.get(modality)
    if fn is None:
        return [Finding(f"No offline analyser for modality '{modality}'.", "unsupported modality", "", 0.0)]
    return fn(img)
