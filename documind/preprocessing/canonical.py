"""Stage A of the OCR reconstruction: bring a PICTURE to the form the models were trained on.

The detectors and the invoice classifier were trained on clean, upright, flat, evenly lit pages (FATURA renders, generate_receipts.py,
generate_cheques.py). A phone photo is the same page lying on a table, tilted, in perspective, maybe turned 90 degrees, with a shadow.
Everything the models learned still applies to it, once it is put back in that form:

  1. find the page in the picture (4 corners) and cut it out, flattening the perspective
  2. turn it upright (0 / 90 / 180 / 270 degrees, from Tesseract's orientation detection; only when it is sure)
  3. flatten uneven lighting (shadows, a gradient from the window) - only when the lighting really is uneven

Every step is skipped when the picture is already in that form (a scan, a PDF page, a generated image): `canonicalize` then returns the SAME
image and an identity transform, so clean inputs cost nothing and are not touched.

The boxes found on the rectified page are in rectified coordinates. The screen draws them over the ORIGINAL upload, so `Canon.map_back`
moves them back with the inverse transform.

    from documind.preprocessing.canonical import canonicalize
    canon = canonicalize(img_bgr, "receipt")
    canon.image                    # what the models should read
    canon.steps                    # ["page", "rotate90", "light"]  ([] = nothing to do)
    canon.to_original([x0, y0, x1, y1])
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import TESSERACT_CMD

MIN_PAGE_AREA = 0.30            # the page must fill at least 30 % of the picture to be trusted as "the page"
FRAMED = 0.96                   # a quad covering 96 %+ of the picture with its corners at the picture corners: already framed
OSD_MIN_CONF = float(os.getenv("DOCUMIND_ORIENT_CONF", "4.0"))      # Tesseract orientation confidence needed to turn a page (a wrong turn is worse than none)
LIGHT_UNEVEN = 0.15             # spread (5th-95th percentile) of the paper brightness, relative to its median, above which the lighting is flattened


@dataclass
class Canon:
    image: np.ndarray                                            # BGR, the page in training form (the input itself when steps == [])
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3, original pixel coordinates -> canonical coordinates
    steps: list[str] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)                               # (width, height) of the original

    @property
    def identity(self) -> bool:
        return not self.steps

    def to_original(self, box) -> list[float]:
        """[x0, y0, x1, y1] on the canonical page -> the smallest box around the same region of the original picture."""
        if self.identity:
            return [float(v) for v in box]
        inv = np.linalg.inv(self.matrix)
        x0, y0, x1, y1 = (float(v) for v in box)
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        p = np.hstack([pts, np.ones((4, 1))]) @ inv.T
        p = p[:, :2] / p[:, 2:3]
        W, H = self.size
        return [float(np.clip(p[:, 0].min(), 0, W)), float(np.clip(p[:, 1].min(), 0, H)), float(np.clip(p[:, 0].max(), 0, W)), float(np.clip(p[:, 1].max(), 0, H))]

    def map_back(self, fields: list[dict] | None = None, items: list[dict] | None = None) -> None:
        """Rewrite, in place, the `bbox` / `boxes` of extraction results from canonical to original coordinates."""
        if self.identity:
            return
        for x in list(fields or []) + list(items or []):
            if x.get("bbox"):
                x["bbox"] = self.to_original(x["bbox"])
            if x.get("boxes"):
                x["boxes"] = [self.to_original(b) for b in x["boxes"]]


# ------------------------------------------------------------------------------------------------ 1. the page
def _order(pts: np.ndarray) -> np.ndarray:
    """4 points -> top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _masks(gray: np.ndarray) -> list[np.ndarray]:
    """Two ways to separate a light page from its background: brightness (Otsu) and edges."""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)          # text and table lines inside the page must not cut it into pieces
    edges = cv2.dilate(cv2.Canny(blur, 40, 120), k, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=3)
    edges = cv2.erode(edges, k, iterations=2)                                    # undo the thickening of the edge lines: the quad must sit ON the page border
    return [bright, edges]


def find_page(img: np.ndarray, min_area: float = MIN_PAGE_AREA) -> np.ndarray | None:
    """The 4 corners (top-left, top-right, bottom-right, bottom-left, in pixels of `img`) of the page, or None when there is no clear page
    or the picture already IS the page."""
    h, w = img.shape[:2]
    f = 800.0 / max(h, w)
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    f = f if f < 1 else 1.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    sh, sw = gray.shape
    best, best_area = None, 0.0
    for mask in _masks(gray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(c)
            if area < min_area * sh * sw or area <= best_area:
                continue
            hull = cv2.convexHull(c)
            if area < 0.88 * cv2.contourArea(hull):                              # not a page: a ragged shape (a table, a hand holding it)
                continue
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
            else:                                                                # rounded / crumpled corners: the tilted rectangle around it
                rect = cv2.minAreaRect(hull)
                if cv2.contourArea(hull) < 0.85 * rect[1][0] * rect[1][1]:
                    continue
                quad = cv2.boxPoints(rect).astype(np.float32)
            quad = _order(quad)
            inside = np.zeros(gray.shape, np.uint8)
            cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
            n_out = int((inside == 0).sum())
            if n_out > 0.02 * sh * sw and float(gray[inside > 0].mean()) < float(gray[inside == 0].mean()) + 8:
                continue                                                         # the "page" is not lighter than what surrounds it: a dark photo of a table
            best, best_area = quad, area
    if best is None:
        return None
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    if best_area >= FRAMED * sh * sw and float(np.abs(best - corners).max()) <= 0.03 * max(sh, sw):
        return None                                                              # already the whole picture
    tl, tr, br, bl = best
    tol = 0.02 * max(sh, sw)
    upright = abs(tl[0] - bl[0]) < tol and abs(tr[0] - br[0]) < tol and abs(tl[1] - tr[1]) < tol and abs(bl[1] - br[1]) < tol
    if upright and best_area >= 0.80 * sh * sw:
        return None                                                              # a straight rectangle that is nearly the whole picture: it only trims a coloured edge, not a photo on a table
    return best / f


def warp_page(img: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut the page out and make it a flat rectangle at its natural resolution -> (image, 3x3 matrix original -> page)."""
    tl, tr, br, bl = quad
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    out_w, out_h = max(out_w, 32), max(out_h, 32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), m.astype(np.float64)


# ------------------------------------------------------------------------------------------------ 2. upright
def detect_rotation(img: np.ndarray, min_conf: float = OSD_MIN_CONF) -> int:
    """Degrees (0, 90, 180, 270) to turn the page CLOCKWISE to make the text upright; 0 when Tesseract is not sure (or has nothing to read)."""
    try:
        import pytesseract
        from pytesseract import Output
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        h, w = img.shape[:2]
        k = 1600.0 / max(h, w)
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.copyMakeBorder(gray, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        osd = pytesseract.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=20", output_type=Output.DICT)
        rot, conf = int(osd.get("rotate", 0)) % 360, float(osd.get("orientation_conf", 0.0))
    except Exception:                                                           # no OSD data, too little text: leave the page as it is
        return 0
    return rot if rot in (90, 180, 270) and conf >= min_conf else 0


def rotate_page(img: np.ndarray, degrees: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn by a multiple of 90 degrees clockwise -> (image, 3x3 matrix). Continuous coordinates (a pixel is a unit square)."""
    h, w = img.shape[:2]
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], np.float64)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180), np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], np.float64)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], np.float64)
    return img, np.eye(3)


# ------------------------------------------------------------------------------------------------ 3. lighting
def flatten_light(img: np.ndarray, threshold: float = LIGHT_UNEVEN) -> tuple[np.ndarray, bool]:
    """Divide the brightness by an estimate of the paper's brightness, only when that is clearly uneven. Colours are kept (V channel of HSV).

    The paper's brightness is estimated from the pixels that are not much darker than the brightest paper around them (blurred and renormalised), so ink, dark table headers and coloured banners
    do not count as "shadow".
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else None
    v = hsv[:, :, 2] if hsv is not None else img
    h, w = v.shape
    f = min(1.0, 800.0 / max(h, w))
    vs = (cv2.resize(v, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else v).astype(np.float32)
    k = max(15, int(max(vs.shape) / 10) | 1)
    paper = (vs >= 0.75 * cv2.dilate(vs, np.ones((k, k), np.uint8))).astype(np.float32)   # not much darker than the brightest paper nearby: ink and dark bands are out
    sig = max(vs.shape) / 12.0
    num, den = cv2.GaussianBlur(vs * paper, (0, 0), sig), cv2.GaussianBlur(paper, (0, 0), sig)
    bg = np.where(den > 0.02, num / np.maximum(den, 1e-3), float(np.median(vs)))
    m0, m1 = bg.shape[0] // 8, bg.shape[1] // 8
    mid = bg[m0: bg.shape[0] - m0, m1: bg.shape[1] - m1]
    if mid.size == 0 or float(np.percentile(mid, 95) - np.percentile(mid, 5)) / max(float(np.median(mid)), 1.0) < threshold:
        return img, False
    bg_full = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR) if f < 1 else bg
    flat = np.clip(v.astype(np.float32) / np.maximum(bg_full, 1.0) * 255.0, 0, 255).astype(np.uint8)
    if hsv is None:
        return flat, True
    hsv = hsv.copy()
    hsv[:, :, 2] = flat
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), True


# ------------------------------------------------------------------------------------------------ all steps
def _on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in ("0", "off", "false", "no")


def canonicalize(img: np.ndarray, doc_type: str = "invoice", *, rectify: bool | None = None, orient: bool | None = None, light: bool | None = None) -> Canon:
    """Picture (BGR) -> the page in training form. `doc_type` (invoice / receipt / cheque / contract) is kept for the steps that may depend on it.

    Switches: DOCUMIND_CANON_PAGE / DOCUMIND_CANON_ORIENT / DOCUMIND_CANON_LIGHT = 0 turn one step off (all on by default).
    """
    rectify = _on("DOCUMIND_CANON_PAGE") if rectify is None else rectify
    orient = _on("DOCUMIND_CANON_ORIENT") if orient is None else orient
    light = _on("DOCUMIND_CANON_LIGHT") if light is None else light
    h, w = img.shape[:2]
    out, m, steps = img, np.eye(3), []
    if rectify:
        quad = find_page(out)
        if quad is not None:
            out, mw = warp_page(out, quad)
            m, steps = mw @ m, steps + ["page"]
    if orient:
        deg = detect_rotation(out)
        if deg:
            out, mr = rotate_page(out, deg)
            m, steps = mr @ m, steps + [f"rotate{deg}"]
    if light:
        out, changed = flatten_light(out)
        if changed:
            steps.append("light")
    return Canon(image=out, matrix=m, steps=steps, size=(w, h))


def canonicalize_pil(pil_img, doc_type: str = "invoice"):
    """Same for a PIL RGB picture (the receipt / cheque / contract readers) -> (PIL picture, Canon). The picture is returned as is when nothing was changed."""
    from PIL import Image
    arr = np.asarray(pil_img.convert("RGB"))[:, :, ::-1].copy()
    canon = canonicalize(arr, doc_type)
    if canon.identity:
        return pil_img, canon
    return Image.fromarray(np.ascontiguousarray(canon.image[:, :, ::-1])), canon
