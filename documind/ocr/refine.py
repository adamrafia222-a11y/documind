"""Optional second opinion on the words Tesseract reads in a full page (the invoice pipeline): the trained recognizer re-reads the doubtful ones.

    DOCUMIND_OCR_REFINE=off        (default) nothing changes
    DOCUMIND_OCR_REFINE=lowconf    re-read the words whose Tesseract confidence is below DOCUMIND_OCR_REFINE_CONF (default 80)
    DOCUMIND_OCR_REFINE=all        re-read every word (slow on a CPU, fine on a GPU)

Tesseract still finds where the words are. For every re-read word the two readings go through documind.ocr.ensemble: when Tesseract was very unsure
(conf < 50) the trained model decides unless its reading cannot be a word; otherwise the model's reading replaces Tesseract's only if it is clearly better
(Tesseract's stays when both look fine). A replaced word gets `conf` 95 (the model gives no confidence).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def mode() -> str:
    m = os.getenv("DOCUMIND_OCR_REFINE", "off").lower()
    return m if m in ("lowconf", "all") else "off"


def refine_words(gray: np.ndarray, words: list[dict], scale: float = 1.0, engine=None) -> list[dict]:
    """words: run_ocr output (boxes in original-image coordinates; `gray` is the image they were read on, `scale` its enlargement)."""
    m = mode()
    if m == "off" or not words:
        return words
    if engine is None:
        from . import trocr
        engine = trocr.get_engine()
    if engine is None:
        return words
    limit = float(os.getenv("DOCUMIND_OCR_REFINE_CONF", "80"))
    todo = [i for i, w in enumerate(words) if m == "all" or w["conf"] < limit]
    H, W = gray.shape[:2]
    crops, idx = [], []
    for i in todo:
        w = words[i]
        x0, y0, x1, y1 = (int(round(v * scale)) for v in (w["x0"], w["y0"], w["x1"], w["y1"]))
        x0, y0, x1, y1 = max(0, x0 - 2), max(0, y0 - 2), min(W, x1 + 2), min(H, y1 + 2)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            crops.append(Image.fromarray(gray[y0:y1, x0:x1]).convert("RGB"))
            idx.append(i)
    if not crops:
        return words
    from .ensemble import choose
    for i, text in zip(idx, engine.read(crops, batch_size=64)):
        text = (text or "").strip()
        if text and " " not in text:                         # a word stays a word: a multi-word answer for one word box is a misreading
            w = words[i]
            pick = choose({"tesseract": w["text"], "trocr": text}, primary="trocr" if w["conf"] < 50 else "tesseract")
            if pick.source == "trocr" and pick.text != w["text"]:
                words[i] = {**w, "text": pick.text, "conf": 95.0, "refined": True}
    return words
