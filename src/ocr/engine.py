"""OCR with Tesseract. Returns words with boxes in ORIGINAL-image coordinates."""
from __future__ import annotations

import numpy as np
import pytesseract
from pytesseract import Output

from ..config import OCR_LANG, OCR_PSM, TESSERACT_CMD

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def run_ocr(gray: np.ndarray, *, scale: float = 1.0, lang: str = OCR_LANG, psm: int = OCR_PSM) -> list[dict]:
    data = pytesseract.image_to_data(
        gray, lang=lang, config=f"--oem 3 --psm {psm}", output_type=Output.DICT
    )
    words: list[dict] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:                      # Tesseract uses -1 for non-word boxes
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(
            {
                "text": text,
                "conf": conf,
                "x0": x / scale,
                "y0": y / scale,
                "x1": (x + w) / scale,
                "y1": (y + h) / scale,
            }
        )
    return words
