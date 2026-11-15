"""OCR with Tesseract. Returns words with boxes in ORIGINAL-image coordinates."""
from __future__ import annotations

import warnings

import numpy as np
import pytesseract
from pytesseract import Output

from ..config import OCR_LANG, OCR_PSM, TESSERACT_CMD

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

_INSTALLED: set[str] | None = None
_WARNED: set[str] = set()


def installed_langs() -> set[str]:
    """Tesseract language packs found on this machine (cached). Empty set if they cannot be listed."""
    global _INSTALLED
    if _INSTALLED is None:
        try:
            _INSTALLED = set(pytesseract.get_languages(config=""))
        except Exception:
            _INSTALLED = set()
    return _INSTALLED


def resolve_lang(lang: str | None = None) -> str:
    """Language string for Tesseract ("eng+fra"), keeping only the packs that are installed.

    A language that is not installed makes Tesseract fail on the WHOLE page, so the document came back empty (text "ignored").
    Here the missing pack is dropped with one warning, and the page is still read with the others ("eng" as the last resort).
    """
    wanted = [x for x in (lang or OCR_LANG).split("+") if x]
    have = installed_langs()
    if not have:                                   # cannot list them: let Tesseract decide
        return "+".join(wanted) or "eng"
    ok = [x for x in wanted if x in have]
    for x in wanted:
        if x not in have and x not in _WARNED:
            _WARNED.add(x)
            warnings.warn(f"Tesseract language '{x}' is not installed: skipped. Install {x}.traineddata to read that language.", stacklevel=2)
    return "+".join(ok) or ("eng" if "eng" in have else sorted(have - {"osd"})[0] if have - {"osd"} else "eng")


def run_ocr(gray: np.ndarray, *, scale: float = 1.0, lang: str = OCR_LANG, psm: int = OCR_PSM) -> list[dict]:
    data = pytesseract.image_to_data(
        gray, lang=resolve_lang(lang), config=f"--oem 3 --psm {psm}", output_type=Output.DICT
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
    from .refine import refine_words                      # optional second opinion of the trained model (DOCUMIND_OCR_REFINE, off by default)
    return refine_words(gray, words, scale)
