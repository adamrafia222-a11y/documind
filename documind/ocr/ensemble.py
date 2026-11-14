"""Two OCR engines, one answer: when the trained model (TrOCR) and Tesseract read the same box, pick the better reading.

    from documind.ocr.ensemble import choose
    choose({"trocr": "12,50", "tesseract": "l2,5O"}, field="total").text      # -> "12,50"

How the reading is chosen, in this order (the first rule that decides wins):
  1. only one engine returned something            -> that one
  2. both read the same text                       -> that text (the strongest evidence there is)
  3. a cross-check function (`prefer`) accepts exactly one  -> that one. Example: on a cheque, the amount in words must be the amount in figures.
  4. format of the field: a date must be a date, an amount must look like an amount (12,50 DH), a phone number has 10 digits, an IBAN the length
     of its country, the amount in words must be understood as a number. The reading that fits the field clearly better (by more than 0.1) wins.
  5. otherwise the primary engine (DOCUMIND_OCR_PRIMARY, default trocr: it is the one trained on your documents)

The two engines do not make the same mistakes: Tesseract drops accents and mixes 0 / 9 / 8, a neural model can drop a character or invent one. When one of
them produces something that cannot be a date / an amount / a number, the other one is used. When both look fine and differ, the primary engine decides:
the checker is only as good as the format rules, it does not know which of two valid-looking readings is true.
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field as dc_field

STATS: Counter = Counter()            # how many readings were decided by which rule (see summary())

# every field name used by the receipt / cheque / contract readers -> the kind of value it holds
KIND = {"date": "date", "contract_date": "date", "first_payment_date": "date", "signing_date": "date", "time": "time", "phone": "phone",
        "total": "amount", "subtotal": "amount", "tax": "amount", "item_price": "amount", "amount": "amount", "amount_digits": "amount", "monthly_payment": "amount",
        "item_qty": "number", "receipt_no": "number", "receipt_number": "number", "cheque_number": "number", "account_number": "number",
        "interest_rate": "rate", "duration": "duration", "iban": "iban", "amount_words": "words", "micr_line": "micr",
        "store": "name", "store_name": "name", "bank_name": "name", "payee": "name", "lender_name": "name", "borrower_name": "name"}
MIN_DIGITS = {"item_qty": 1, "receipt_no": 4, "receipt_number": 4, "cheque_number": 5, "account_number": 20}
AMOUNT_FULL = re.compile(r"^\d[\d ]*[.,]\d{1,2}(\s?(dh|dhs|mad|eur|€))?$", re.I)
AMOUNT_ANY = re.compile(r"\d[\d ]*[.,]\d{1,2}")
TIME_FULL = re.compile(r"^\d{1,2}\s?[:hH]\s?\d{2}(\s?:\s?\d{2})?$")
RATE_FULL = re.compile(r"^\d+(?:[.,]\d+)?\s?%$")
DURATION_FULL = re.compile(r"^\d+\s?(mois|ans?|annees?)$", re.I)
ORDINARY = set(" .,'’-/&°%()€:;+#")
JUNK = set("|_~`^\\{}[]<>@*=")


def _fold(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))


def _norm(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t or "").split()).casefold()


def _frac(t: str, ok) -> float:
    return sum(1 for c in t if ok(c)) / max(1, len(t))


def plausibility(field: str | None, text: str, whitelist: str | None = None) -> float:
    """0..1: does `text` look like a value of this field? (1 = fits the format exactly)."""
    t = (text or "").strip()
    if not t:
        return 0.0
    kind = KIND.get(field or "", "text")
    digits = sum(c.isdigit() for c in t)
    if kind == "amount":
        if AMOUNT_FULL.match(t):
            return 1.0
        numeric = _frac(t, lambda c: c.isdigit() or c in ",. ")                             # a lookalike letter inside the number (l for 1, O for 0) lowers this
        return 0.7 * numeric if AMOUNT_ANY.search(t) else 0.1 * numeric
    if kind == "date":
        from ..inference.contracts import parse_date_fr
        d = parse_date_fr(t)
        return (1.0 if 1990 <= d.year <= 2100 else 0.5) if d else 0.4 if re.search(r"\d{1,2}\s?[-/.]\s?\d{1,2}", t) else 0.05
    if kind == "time":
        return 1.0 if TIME_FULL.match(t) else 0.6 if re.search(r"\d{1,2}\s?[:hH]\s?\d{2}", t) else 0.05
    if kind == "phone":
        return 1.0 if digits == 10 and _frac(t, lambda c: c.isdigit() or c in " +.-") == 1 else 0.7 if 9 <= digits <= 12 else 0.1
    if kind == "number":
        body = t.replace("N°", "").replace("n°", "")
        return _frac(body, lambda c: c.isdigit() or c == " ") * (1.0 if sum(c.isdigit() for c in body) >= MIN_DIGITS.get(field or "", 1) else 0.5)
    if kind == "rate":
        return 1.0 if RATE_FULL.match(t) else 0.5 if digits else 0.05
    if kind == "duration":
        return 1.0 if DURATION_FULL.match(_fold(t)) else 0.3 if digits else 0.05
    if kind == "iban":
        c = re.sub(r"[^A-Za-z0-9]", "", t).upper()
        from ..inference.contracts import IBAN_LENGTH
        want = IBAN_LENGTH.get(c[:2])
        return 1.0 if want and len(c) == want else 0.6 if want and abs(len(c) - want) == 1 else 0.1
    if kind == "words":
        from ..inference.cheques import _norm as words_norm, _snap, words_to_number
        if words_to_number(t) is not None:
            return 1.0
        toks = words_norm(t)
        return 0.8 * sum(_snap(w) is not None for w in toks) / max(1, len(toks))
    if kind == "micr":
        return 0.9 * _frac(t, lambda c: c.isdigit() or c in "<:. ") + (0.1 if "<" in t else 0.0)
    ordinary = _frac(t, lambda c: c.isalnum() or c in ORDINARY)                      # names, addresses, item names ...
    junk = _frac(t, lambda c: c in JUNK)
    mixed = sum(1 for w in t.split() if len(w) > 2 and any(c.isalpha() for c in w) and any(c.isdigit() for c in w))        # "T0TAL": letters and digits in one word
    score = ordinary - 2 * junk - 0.15 * mixed
    if kind == "name":
        score -= 0.5 * digits / len(t)
    return max(0.0, min(1.0, score))


@dataclass
class Choice:
    text: str
    source: str                                       # the engine whose reading was kept
    reason: str                                       # only / agreed / cross_check / format / primary / empty
    candidates: dict = dc_field(default_factory=dict)
    scores: dict = dc_field(default_factory=dict)


def choose(candidates: dict[str, str], field: str | None = None, whitelist: str | None = None, prefer=None, primary: str | None = None) -> Choice:
    """candidates: {engine name: text}. See the module docstring for the rules."""
    primary = primary or os.getenv("DOCUMIND_OCR_PRIMARY", "trocr").lower()
    cands = {k: (v or "").strip() for k, v in candidates.items()}
    live = {k: v for k, v in cands.items() if v}
    if not live:
        return Choice("", primary, "empty", cands)
    if len(live) == 1:
        k, v = next(iter(live.items()))
        STATS[f"only_{k}"] += 1
        return Choice(v, k, "only", cands)
    first = primary if primary in live else next(iter(live))
    if len({_norm(v) for v in live.values()}) == 1:
        STATS["agreed"] += 1
        return Choice(live[first], first, "agreed", cands)
    scores = {k: round(plausibility(field, v, whitelist), 3) for k, v in live.items()}
    if prefer is not None:
        ok = [k for k, v in live.items() if prefer(v)]
        if len(ok) == 1:
            STATS["cross_check"] += 1
            return Choice(live[ok[0]], ok[0], "cross_check", cands, scores)
    best = max(scores.values())
    top = [k for k, s in scores.items() if s >= best - 0.1]
    if len(top) == 1:
        STATS[f"format_{top[0]}"] += 1
        return Choice(live[top[0]], top[0], "format", cands, scores)
    k = primary if primary in top else top[0]
    STATS[f"primary_{k}"] += 1
    return Choice(live[k], k, "primary", cands, scores)


def summary() -> dict:
    """How the readings were decided so far in this process, e.g. {'agreed': 812, 'format_tesseract': 17, 'primary_trocr': 33, ...}."""
    return dict(STATS)
