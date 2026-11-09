"""Decide what kind of document a picture / PDF is, so that it goes to the right extractor (and to the right CSV).

    from documind.inference.router import detect_type
    detect_type(data, "scan.jpg")   # {"doc_type": "receipt", "confidence": 0.84, "scores": {"invoice": 1.0, "receipt": 5.5}, "reason": "..."}

How it decides (cheap and readable, no training needed):
  * shape of the page: a receipt is a long narrow strip (height / width above ~1.65), an invoice is an A4 / Letter page (about 1.3 - 1.5)
  * words found by a quick, low-resolution Tesseract pass: "invoice", "bill to", "unit price", "due date" ... point to an invoice;
    "receipt", "cashier", "change", "visa", "merci" ... point to a receipt (English and French)
The score of each type is added up; the higher one wins and a tie goes to the invoice model. The web page shows the choice and lets the
user force the other type, so a wrong guess costs one click. Once you have labelled examples of both types you can replace `score_types`
by a trained classifier (the rest of the code does not change).
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from ..config import OCR_LANG, PDF_DPI, TESSERACT_CMD
from ..preprocessing import decode_image, pdf_to_images

TYPES = ("invoice", "receipt")

# (pattern, weight). Patterns are matched as whole words / phrases on the lower-cased text.
INVOICE_WORDS = [
    (r"invoice", 2.0), (r"facture", 2.0), (r"bill to", 1.5), (r"ship to", 1.0), (r"invoice (no|number|date|#)", 1.5), (r"due date", 1.0),
    (r"purchase order", 1.0), (r"po number", 1.0), (r"unit price", 1.0), (r"qty", 0.5), (r"quantity", 0.5), (r"sub ?total", 0.5),
    (r"gstin", 1.0), (r"vat (id|number|no)", 1.0), (r"payment terms", 1.0), (r"seller", 0.7), (r"buyer", 0.7), (r"bank details", 1.0),
    (r"amount due", 1.0), (r"bon de commande", 1.0), (r"date d.?[ée]ch[ée]ance", 1.0), (r"bill(ing)? address", 1.0),
]
RECEIPT_WORDS = [
    (r"receipt", 2.0), (r"re[çc]u", 1.5), (r"ticket", 1.5), (r"caisse", 1.5), (r"cashier", 1.5), (r"caissi[eè]re?", 1.5), (r"cash", 0.7),
    (r"change", 0.7), (r"merci", 1.0), (r"thank you", 1.0), (r"thanks", 0.7), (r"visa", 0.7), (r"mastercard", 0.7), (r"card payment", 1.0),
    (r"tpe", 0.7), (r"esp[èe]ces", 1.0), (r"monnaie", 1.0), (r"rendu", 0.7), (r"terminal", 0.5), (r"transaction", 0.5), (r"customer copy", 1.0),
    (r"bienvenue", 1.0), (r"welcome", 0.7), (r"[àa] bient[ôo]t", 1.0), (r"see you", 0.7), (r"items? sold", 1.0), (r"tendered", 1.0),
]
_INV = [(re.compile(rf"\b{p}\b"), w) for p, w in INVOICE_WORDS]
_REC = [(re.compile(rf"\b{p}\b"), w) for p, w in RECEIPT_WORDS]


def score_types(text: str, aspect: float) -> dict[str, float]:
    """Evidence for each type from the page text (any case) and the page shape (height / width)."""
    t = (text or "").lower()
    inv = sum(w for rx, w in _INV if rx.search(t))
    rec = sum(w for rx, w in _REC if rx.search(t))
    if aspect >= 1.9:
        rec += 3.0
    elif aspect >= 1.65:
        rec += 1.5
    elif aspect <= 1.0:
        inv += 1.5                                                # landscape page
    elif aspect <= 1.5:
        inv += 1.5                                                # A4 (1.41) / Letter (1.29)
    return {"invoice": round(inv, 2), "receipt": round(rec, 2)}


def _first_page(data: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        pages = pdf_to_images(data, min(PDF_DPI, 110))
        if not pages:
            raise ValueError("Empty PDF.")
        return pages[0]
    return decode_image(data)


def quick_text(img: np.ndarray, max_side: int = 1400) -> str:
    """One fast, low-resolution Tesseract pass (a second or so), only used to look for key words."""
    try:
        import pytesseract
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        h, w = img.shape[:2]
        k = max_side / max(h, w)
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        return pytesseract.image_to_string(gray, lang=OCR_LANG, config="--psm 6")
    except Exception:                                               # no Tesseract / no language data: the page shape alone decides
        return ""


def detect_type(data: bytes, filename: str = "document", available: tuple[str, ...] | list[str] = TYPES) -> dict:
    """-> {"doc_type", "confidence" (0.5..1), "scores", "reason"}; only types in `available` can be chosen."""
    img = _first_page(data, filename)
    h, w = img.shape[:2]
    aspect = h / max(w, 1)
    scores = score_types(quick_text(img), aspect)
    allowed = [t for t in TYPES if t in available] or list(TYPES)
    if len(allowed) == 1:
        return {"doc_type": allowed[0], "confidence": 1.0, "scores": scores, "reason": f"only the {allowed[0]} model is installed"}
    inv, rec = scores["invoice"], scores["receipt"]
    doc_type = "receipt" if rec > inv else "invoice"
    total = inv + rec
    conf = round(max(inv, rec) / total, 2) if total > 0 else 0.5
    return {"doc_type": doc_type, "confidence": max(conf, 0.5), "scores": scores,
            "reason": f"page shape {aspect:.2f} (height/width), invoice words {inv}, receipt words {rec}"}
