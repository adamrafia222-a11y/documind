"""Tesseract on one crop: the reference every other OCR engine is compared with (documind.training.evaluate_ocr)."""
from __future__ import annotations

from PIL import Image, ImageOps

from ..config import OCR_LANG, TESSERACT_CMD


def read_tesseract(img: Image.Image, psm: int = 7, lang: str | None = None, scale: float | None = None) -> str:
    """One text line -> text. Enlarged so that the text is about 40 px high (Tesseract's comfortable size), white margin added."""
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    g = img.convert("L")
    k = scale or min(4.0, max(1.0, 40.0 / max(1, g.height)))
    g = g.resize((max(1, int(g.width * k)), max(1, int(g.height * k))), Image.LANCZOS)
    g = ImageOps.expand(ImageOps.autocontrast(g, cutoff=1), border=16, fill=255)
    return " ".join(pytesseract.image_to_string(g, lang=lang or OCR_LANG, config=f"--psm {psm}").split())
