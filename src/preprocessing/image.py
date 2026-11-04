"""Image pre-processing steps (the "pretreatment" stage).

Pipeline (each step can be switched off):
  1. grayscale
  2. deskew (projection-profile search, +-10 degrees)
  3. resize so that the long side is ~2400 px (Tesseract works best around 300 dpi)
  4. shadow / uneven-lighting removal
  5. light denoising + CLAHE contrast
  6. optional adaptive binarisation

`Prepared.scale` is the resize factor, so OCR boxes can be mapped back to the
coordinates of the ORIGINAL image (needed to align OCR words with FATURA labels).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TARGET_LONG_SIDE = 2400
MIN_LONG_SIDE = 1600
MAX_LONG_SIDE = 3600


@dataclass
class Prepared:
    image: np.ndarray   # grayscale uint8 image given to the OCR
    scale: float        # processed size / original size
    angle: float        # deskew angle applied (degrees), 0.0 if none


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unreadable image (unsupported or corrupted file).")
    return img


def read_image(path: str | Path) -> np.ndarray:
    # read bytes first: cv2.imread fails on non-ASCII paths on Windows
    return decode_image(Path(path).read_bytes())


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Angle (degrees) to rotate the image by so that text lines become horizontal."""
    h, w = gray.shape
    f = 800.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sh, sw = bw.shape
    center = (sw / 2.0, sh / 2.0)
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        m = cv2.getRotationMatrix2D(center, float(a), 1.0)
        rot = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST)
        profile = rot.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(profile) ** 2))   # sharp row profile = straight lines
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def rotate_keep_size(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimate of the background to flatten shadows and coloured paper."""
    bg = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def preprocess(
    img: np.ndarray,
    *,
    deskew: bool = True,
    resize: bool = True,
    remove_shadow: bool = True,
    denoise: bool = True,
    contrast: bool = True,
    binarize: bool = False,
) -> Prepared:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape

    angle = 0.0
    if deskew:
        a = estimate_skew(gray)
        if abs(a) >= 0.5:
            gray = rotate_keep_size(gray, a)
            angle = a

    scale = 1.0
    long_side = max(h, w)
    if resize and (long_side < MIN_LONG_SIDE or long_side > MAX_LONG_SIDE):
        scale = TARGET_LONG_SIDE / long_side
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)

    if remove_shadow:
        gray = remove_shadows(gray)
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    if contrast:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Prepared(image=gray, scale=scale, angle=angle)


def remove_table_lines(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Erase long horizontal / vertical rules (table borders). Tesseract reads a bordered table much worse than the same
    table without its lines. -> (image without lines, True when lines were found)."""
    h, w = gray.shape[:2]
    mask = (gray < 238).astype(np.uint8) * 255                      # anything that is not paper white
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 45)))
    lines = cv2.bitwise_or(cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk), cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk))
    if int((lines > 0).sum()) < 0.0005 * h * w:
        return gray, False
    out = gray.copy()
    out[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 255
    return out, True
