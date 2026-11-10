"""Deterministic synthetic medical images for development, demos and tests.

These are *not* real patient images. They are cartoons with just enough
structure (lung fields, a heart shadow, an optic disc, nuclei...) for the
offline vision stand-in to produce meaningful, reproducible findings. Each
generator can burn fake patient identifiers into the pixels so that the
de-identification stage can be exercised.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FAKE_BURNED_TEXT = "NAME: JOHN DOE  MRN: 00123456  DOB: 01/02/1960"


def _burn(img: Image.Image, text: str, fill, box_fill=None) -> Image.Image:
    d = ImageDraw.Draw(img)
    if box_fill is not None:
        d.rectangle([4, 4, 8 + 6 * len(text) + 4, 22], fill=box_fill)
    d.text((8, 8), text, fill=fill)
    d.text((8, img.height - 18), "ACC# 987654  ST. MARY HOSPITAL", fill=fill)
    return img


def chest_xray(finding: str = "normal", seed: int = 0, burn_text: bool = False, size: int = 512) -> Image.Image:
    """finding in {normal, consolidation, effusion, cardiomegaly, pneumothorax}."""
    rng = np.random.default_rng(seed)
    W = H = size
    img = Image.new("L", (W, H), 25)
    d = ImageDraw.Draw(img)
    d.ellipse([0.06 * W, 0.12 * H, 0.94 * W, 1.10 * H], fill=95)  # body / soft tissue
    d.ellipse([0.12 * W, 0.20 * H, 0.46 * W, 0.86 * H], fill=48)  # patient right lung (image left)
    d.ellipse([0.54 * W, 0.20 * H, 0.88 * W, 0.86 * H], fill=48)  # patient left lung (image right)
    d.rectangle([0.47 * W, 0.10 * H, 0.53 * W, 1.0 * H], fill=150)  # spine / mediastinum
    heart_w = {"cardiomegaly": 0.66}.get(finding, 0.30)
    hc = 0.50 if finding == "cardiomegaly" else 0.53
    d.ellipse([(hc - heart_w / 2) * W, 0.55 * H, (hc + heart_w / 2) * W, 0.85 * H], fill=165)
    if finding == "consolidation":
        for _ in range(9):
            cx, cy = rng.uniform(0.20, 0.36) * W, rng.uniform(0.57, 0.68) * H
            r = rng.uniform(0.03, 0.06) * W
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.uniform(125, 150)))
    elif finding == "effusion":
        d.polygon(
            [(0.56 * W, 0.74 * H), (0.70 * W, 0.71 * H), (0.88 * W, 0.66 * H), (0.90 * W, 0.90 * H),
             (0.55 * W, 0.90 * H)],
            fill=140,
        )
    elif finding == "pneumothorax":
        d.ellipse([0.70 * W, 0.20 * H, 0.90 * W, 0.62 * H], fill=8)
    img = img.filter(ImageFilter.GaussianBlur(3))
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 3, (H, W))
    img = Image.fromarray(np.clip(arr, 0, 240).astype(np.uint8), "L")
    if burn_text:
        img = _burn(img, FAKE_BURNED_TEXT, fill=255)
    return img


def fundus(finding: str = "normal", seed: int = 0, burn_text: bool = False, size: int = 512) -> Image.Image:
    """finding in {normal, diabetic_retinopathy, glaucoma}."""
    rng = np.random.default_rng(seed)
    W = H = size
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0.05 * W, 0.05 * H, 0.95 * W, 0.95 * H], fill=(200, 85, 45))
    disc = (0.66 * W, 0.50 * H)
    for ang in np.linspace(0, 2 * np.pi, 9, endpoint=False):
        x2 = disc[0] + np.cos(ang) * 0.26 * W
        y2 = disc[1] + np.sin(ang) * 0.26 * H
        d.line([disc, (x2, y2)], fill=(155, 35, 25), width=4)
    r = 0.07 * W
    d.ellipse([disc[0] - r, disc[1] - r, disc[0] + r, disc[1] + r], fill=(250, 232, 190))
    cup = 0.82 if finding == "glaucoma" else 0.35
    d.ellipse([disc[0] - r * cup, disc[1] - r * cup, disc[0] + r * cup, disc[1] + r * cup], fill=(255, 252, 240))
    if finding == "diabetic_retinopathy":
        for _ in range(18):
            x, y = rng.uniform(0.2, 0.55) * W, rng.uniform(0.25, 0.75) * H
            rr = rng.uniform(3, 6)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(95, 15, 10))
        for _ in range(12):
            x, y = rng.uniform(0.3, 0.55) * W, rng.uniform(0.35, 0.65) * H
            rr = rng.uniform(2, 4)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(235, 215, 90))
    if burn_text:
        img = _burn(img, FAKE_BURNED_TEXT, fill=(255, 255, 255))
    return img


def histopathology(finding: str = "benign", seed: int = 0, burn_text: bool = False, size: int = 512) -> Image.Image:
    """finding in {benign, carcinoma}."""
    rng = np.random.default_rng(seed)
    W = H = size
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[..., 0], base[..., 1], base[..., 2] = 236, 178, 202
    base += rng.normal(0, 6, (H, W, 1))
    img = Image.fromarray(np.clip(base, 0, 244).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    if finding == "carcinoma":
        n, rmin, rmax = 520, 4, 12
    else:
        n, rmin, rmax = 130, 5, 6
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        rx = rng.uniform(rmin, rmax)
        ry = rx * rng.uniform(0.7, 1.0)
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(int(rng.uniform(70, 100)), 45, int(rng.uniform(130, 150))))
    if burn_text:
        img = _burn(img, FAKE_BURNED_TEXT, fill=(0, 0, 0), box_fill=(255, 255, 255))
    return img


GENERATORS = {"xray": chest_xray, "fundus": fundus, "histopathology": histopathology}


def to_png_bytes(img: Image.Image, **info: str) -> bytes:
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    for k, v in info.items():
        meta.add_text(k, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()
