"""Image de-identification.

1. *Metadata*: the image is decoded and re-encoded from raw pixels into a fresh
   PNG, which discards EXIF, PNG text chunks, ICC profiles and any DICOM header.
2. *Burned-in text*: tiles containing text-like strokes (many near-saturated
   foreground runs per row, the signature of rendered glyphs) are detected and
   blacked out. If Tesseract OCR is installed its word boxes are masked as well.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

TILE = 8


@dataclass
class ImageDeidResult:
    png: bytes
    width: int
    height: int
    masked_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    metadata_removed: list[str] = field(default_factory=list)
    ocr_used: bool = False


def _load(data: bytes) -> tuple[Image.Image, list[str]]:
    if data[128:132] == b"DICM":  # pragma: no cover - optional pydicom path
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        arr = (255 * (arr - arr.min()) / max(1.0, float(arr.max() - arr.min()))).astype(np.uint8)
        return Image.fromarray(arr), ["dicom-header"]
    img = Image.open(io.BytesIO(data))
    removed = sorted(k for k in img.info if k not in ("dpi",))
    try:
        if img.getexif():
            removed.append("exif")
    except Exception:  # pragma: no cover
        pass
    return img, removed


def _stroke_rows(mask: np.ndarray) -> np.ndarray:
    """For each tile: number of rows that contain >=4 foreground/background transitions."""
    h, w = mask.shape
    th, tw = h // TILE, w // TILE
    m = mask[: th * TILE, : tw * TILE].astype(np.int8)
    trans = np.abs(np.diff(m, axis=1))  # (H, W-1)
    trans = np.pad(trans, ((0, 0), (0, 1)))
    per_row = trans.reshape(th, TILE, tw, TILE).sum(axis=3)  # (th, TILE, tw)
    return (per_row >= 3).sum(axis=1)  # (th, tw)


def detect_text_tiles(img: Image.Image) -> np.ndarray:
    from PIL import ImageFilter

    gray = img.convert("L")
    g = np.asarray(gray, dtype=np.int16)
    local = np.asarray(gray.filter(ImageFilter.BoxBlur(6)), dtype=np.int16)
    bright = (g >= 225) & (g - local >= 60)  # light glyphs on a darker background
    dark = (g <= 40) & (local - g >= 60)  # dark glyphs on a lighter background
    score = np.maximum(_stroke_rows(bright), _stroke_rows(dark))
    seeds = score >= 2
    # glyphs come in horizontal runs: a seed needs a seed neighbour on the same tile row
    left = np.pad(seeds, ((0, 0), (1, 0)))[:, :-1]
    right = np.pad(seeds, ((0, 0), (0, 1)))[:, 1:]
    seeds &= left | right
    # candidate tiles hold any glyph-like pixels; grow each text line through them
    h, w = g.shape
    th, tw = h // TILE, w // TILE
    fg = (bright | dark)[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE).sum(axis=(1, 3))
    cand = (fg >= 2) | seeds
    closed = cand.copy()
    for dx in range(1, 4):  # bridge gaps between words of up to 3 tiles
        closed |= np.roll(cand, dx, 1) & np.roll(cand, -dx, 1)
    mask = np.zeros_like(seeds)
    stack = list(zip(*np.nonzero(seeds), strict=True))
    while stack:
        y, x = stack.pop()
        if mask[y, x]:
            continue
        mask[y, x] = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < th and 0 <= nx < tw and closed[ny, nx] and not mask[ny, nx]:
                    stack.append((ny, nx))
    # dilate by one tile so anti-aliased glyph edges are covered
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def _ocr_boxes(img: Image.Image) -> list[tuple[int, int, int, int]]:
    try:
        import pytesseract
    except Exception:
        return []
    try:  # pragma: no cover - requires the tesseract binary
        data = pytesseract.image_to_data(img.convert("L"), output_type=pytesseract.Output.DICT)
    except Exception:  # pragma: no cover
        return []
    boxes = []  # pragma: no cover
    for i, txt in enumerate(data["text"]):  # pragma: no cover
        if txt.strip() and float(data["conf"][i]) > 50:
            boxes.append((data["left"][i] - 2, data["top"][i] - 2,
                          data["left"][i] + data["width"][i] + 2, data["top"][i] + data["height"][i] + 2))
    return boxes  # pragma: no cover


def deidentify_image(data: bytes, use_ocr: bool = True) -> ImageDeidResult:
    img, removed = _load(data)
    img = img.convert("RGB") if img.mode not in ("L", "RGB") else img
    arr = np.array(img)
    tiles = detect_text_tiles(img)
    regions: list[tuple[int, int, int, int]] = []
    for ty, tx in zip(*np.nonzero(tiles), strict=True):
        y0, x0 = int(ty) * TILE, int(tx) * TILE
        arr[y0 : y0 + TILE, x0 : x0 + TILE] = 0
        regions.append((x0, y0, x0 + TILE, y0 + TILE))
    ocr_boxes = _ocr_boxes(img) if use_ocr else []
    for x0, y0, x1, y1 in ocr_boxes:  # pragma: no cover
        arr[max(0, y0) : y1, max(0, x0) : x1] = 0
        regions.append((x0, y0, x1, y1))
    clean = Image.fromarray(arr, img.mode)  # fresh image object: no metadata carried over
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return ImageDeidResult(buf.getvalue(), clean.width, clean.height, regions, removed, bool(ocr_boxes))
