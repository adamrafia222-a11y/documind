"""Confidence of a field that was found by RULES (not by the detector), from evidence instead of a fixed number.

    evidence_confidence(words, valid, labelled, computed) -> 0.05 .. 0.92

  base         0.45 when a printed label sits next to the value ("Date d'échéance : ..."), 0.30 for a positional guess (the biggest line at the top = the seller),
               0.35 for a value computed from other amounts
  + 0.30 x     how well the OCR read the words of the value (Tesseract's own word confidence, 0..1)
  + 0.15       the value has a VALID format (a real calendar date, a well-formed e-mail, an amount that parses, an IBAN that passes mod 97 ...)
  - 0.15       the format is clearly wrong
Then documind.inference.invoice_checks.adjust_confidence raises the fields that a passing check confirms (+0.08) and lowers the ones a failing check contradicts (x 0.7).
So a clean, labelled, valid field reaches about 0.9; a name read from a blurry line stays near 0.5. The numbers are an estimate of how sure the reading is, not a probability.
"""
from __future__ import annotations

import re

MAX = 0.92


def ocr_conf(words: list[dict] | None) -> float:
    """Mean Tesseract confidence of the words, 0..1 (0.6 when unknown)."""
    c = [float(w["conf"]) for w in (words or []) if w.get("conf") is not None]
    return min(1.0, max(0.0, sum(c) / len(c) / 100.0)) if c else 0.6


def words_in_box(words: list[dict], bbox: list[float]) -> list[dict]:
    x0, y0, x1, y1 = bbox
    return [w for w in words if x0 - 2 <= (w["x0"] + w["x1"]) / 2 <= x1 + 2 and y0 - 2 <= (w["y0"] + w["y1"]) / 2 <= y1 + 2]


def evidence_confidence(words: list[dict] | None, valid: bool | None = None, labelled: bool = True, computed: bool = False) -> float:
    base = 0.35 if computed else (0.45 if labelled else 0.30)
    c = base + 0.30 * ocr_conf(words) + (0.15 if valid is True else (-0.15 if valid is False else 0.0))
    return round(min(MAX, max(0.05, c)), 2)


def valid_format(label: str, text: str) -> bool | None:
    """True / False when the label has a checkable format, None when it is free text (a name, an address)."""
    from ..features.patterns import EMAIL_RE, parse_amount
    from .invoice_checks import _iban_ok, _words_value, parse_date
    L, t = label.upper(), (text or "").strip()
    if L in ("DATE", "DUE_DATE"):
        return parse_date(t) is not None
    if L == "SELLER_EMAIL":
        return bool(EMAIL_RE.fullmatch(t))
    if L == "SELLER_SITE":
        return bool(re.search(r"\.[a-z]{2,}", t, re.I))
    if L in ("NUMBER", "PO_NUMBER"):
        return bool(re.search(r"\d", t))
    if L.startswith("GSTIN"):
        return len(re.sub(r"\D", "", t)) >= 8
    if L in ("TOTAL", "SUB_TOTAL", "TAX", "GST", "AMOUNT_DUE", "DISCOUNT"):
        return parse_amount(t) is not None or "%" in t
    if L == "TOTAL_WORDS":
        return _words_value(t) is not None
    if L == "PAYMENT_DETAILS":
        ok = _iban_ok(t)
        return ok if ok is not None else None
    return None
