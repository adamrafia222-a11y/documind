"""Align OCR words with the labelled boxes of the dataset.

A word gets the label of the annotation box that covers most of it (>= 50 %),
otherwise "O" (background).

Datasets disagree on how a box is written: x,y,width,height or x0,y0,x1,y1; pixels, 0-1 or 0-1000;
sometimes relative to a different image size. `align_labels` tries every convention on each document
and keeps the one whose boxes fit the OCR words best, so a wrong guess can no longer silently
produce "0 labelled words".
"""
from __future__ import annotations

import numpy as np

from ..config import BACKGROUND_LABEL
from .loader import Annotation, to_pixels

MODES = ("xyxy", "xywh", "xyxy_flipy", "xywh_flipy", "xyxy_n1000", "xywh_n1000", "xyxy_n1", "xywh_n1", "xyxy_ref", "xywh_ref")


def _word_boxes(words: list[dict]) -> np.ndarray:
    return np.array([[w["x0"], w["y0"], w["x1"], w["y1"]] for w in words], dtype=np.float64).reshape(-1, 4)


def _cover(wb: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """cover[i, j] = share of word i that lies inside box j."""
    ix0 = np.maximum(wb[:, None, 0], ab[None, :, 0])
    iy0 = np.maximum(wb[:, None, 1], ab[None, :, 1])
    ix1 = np.minimum(wb[:, None, 2], ab[None, :, 2])
    iy1 = np.minimum(wb[:, None, 3], ab[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    w_area = np.maximum((wb[:, 2] - wb[:, 0]) * (wb[:, 3] - wb[:, 1]), 1e-6)
    return inter / w_area[:, None]


def compute_labels(words: list[dict], ann: list[tuple[str, tuple[float, float, float, float]]],
                   min_cover: float = 0.5) -> list[str]:
    """Label of every word (pure function, nothing is modified)."""
    if not words:
        return []
    if not ann:
        return [BACKGROUND_LABEL] * len(words)
    wb = _word_boxes(words)
    ab = np.array([a[1] for a in ann], dtype=np.float64).reshape(-1, 4)
    a_area = np.maximum((ab[:, 2] - ab[:, 0]) * (ab[:, 3] - ab[:, 1]), 1e-6)
    cover = _cover(wb, ab)
    # among boxes covering the word enough, prefer the smallest (most specific) one
    score = np.where(cover >= min_cover, cover - 1e-9 * a_area[None, :], -1.0)
    best = score.argmax(axis=1)
    return [ann[best[i]][0] if score[i, best[i]] > 0 else BACKGROUND_LABEL for i in range(len(words))]


def assign_labels(words: list[dict], ann: list[tuple[str, tuple[float, float, float, float]]], min_cover: float = 0.5) -> None:
    """Add a "label" key to each OCR word, in place."""
    for w, lab in zip(words, compute_labels(words, ann, min_cover)):
        w["label"] = lab


def _fit_score(wb: np.ndarray, ab: np.ndarray, min_cover: float = 0.5) -> float:
    """How well do these boxes wrap the OCR words? Sum over boxes of IoU(box, tight box of its words).

    A correct convention gives boxes that hug their words (IoU close to 1). A wrong one (e.g. x,y,w,h read
    as x0,y0,x1,y1) gives huge or empty boxes: the score stays near 0 even if some words fall inside by chance.
    """
    if len(ab) == 0 or len(wb) == 0:
        return 0.0
    ok = (ab[:, 2] > ab[:, 0]) & (ab[:, 3] > ab[:, 1])
    mask = _cover(wb, ab) >= min_cover                          # (n_words, n_boxes)
    mask &= ok[None, :]
    has = mask.any(axis=0)
    if not has.any():
        return 0.0
    inf = np.inf
    tx0 = np.where(mask, wb[:, 0:1], inf).min(axis=0)
    ty0 = np.where(mask, wb[:, 1:2], inf).min(axis=0)
    tx1 = np.where(mask, wb[:, 2:3], -inf).max(axis=0)
    ty1 = np.where(mask, wb[:, 3:4], -inf).max(axis=0)
    t = np.stack([tx0, ty0, tx1, ty1], axis=1)[has]
    a = ab[has]
    ix = np.clip(np.minimum(t[:, 2], a[:, 2]) - np.maximum(t[:, 0], a[:, 0]), 0, None)
    iy = np.clip(np.minimum(t[:, 3], a[:, 3]) - np.maximum(t[:, 1], a[:, 1]), 0, None)
    inter = ix * iy
    union = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1]) + (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) - inter
    return float(np.sum(inter / np.maximum(union, 1e-6)))


def align_labels(words: list[dict], annotations: list[Annotation], image_size: tuple[float, float],
                 min_cover: float = 0.5, drop: set[str] | frozenset[str] = frozenset()) -> dict:
    """Label the OCR words in place using the best (annotation file, coordinate convention).

    `annotations` = the same invoice in several file formats; every one is tried.
    `drop` = labels that must become background (e.g. {"TABLE"}), applied after the alignment so those
    boxes still help to find the right convention.
    Returns {"mode": ..., "source": index of the annotation used, "n_boxes", "n_labelled", "fit": score}.
    """
    W, H = image_size
    for w in words:
        w["label"] = BACKGROUND_LABEL
    info = {"mode": "none", "source": -1, "n_boxes": 0, "n_labelled": 0, "fit": 0.0}
    if not words or not annotations:
        return info
    wb = _word_boxes(words)

    best = None   # (score, source, mode, labels, boxes)
    for si, ann in enumerate(annotations):
        real = [b for b in ann.boxes if b.label != BACKGROUND_LABEL]
        if not real:
            continue
        vals = np.array([b.vals for b in real], dtype=np.float64)
        hints = {b.hint for b in real}
        modes = list(MODES)
        if "xywh" in hints and "xyxy" not in hints:          # the file tells us its shape: try that first
            modes.sort(key=lambda m: not m.startswith("xywh"))
        elif "xyxy" in hints and "xywh" not in hints:
            modes.sort(key=lambda m: not m.startswith("xyxy"))
        for mode in modes:
            if mode.endswith("_ref") and (not ann.ref_size or
                                          (abs(ann.ref_size[0] - W) < 1 and abs(ann.ref_size[1] - H) < 1)):
                continue
            ab = to_pixels(vals, mode, W, H, ann.ref_size)
            score = _fit_score(wb, ab, min_cover)
            if best is None or score > best[0] + 1e-9:         # strict '>' keeps the earlier (hinted) mode on ties
                best = (score, si, mode, [b.label for b in real], ab)

    if best is None or best[0] <= 0:
        return info
    score, si, mode, labels, ab = best
    pairs = [(lab, tuple(box)) for lab, box in zip(labels, ab)]
    out = compute_labels(words, pairs, min_cover)
    for w, lab in zip(words, out):
        w["label"] = BACKGROUND_LABEL if lab in drop else lab
    info.update(mode=mode, source=si, n_boxes=len(pairs), fit=round(score, 3),
                n_labelled=sum(w["label"] != BACKGROUND_LABEL for w in words))
    return info
