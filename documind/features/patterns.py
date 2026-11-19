"""Patterns and the amount parser shared by the rules reader (documind.inference.postprocess) and the OCR reconstruction: dates, money, e-mail, URL, phone."""
from __future__ import annotations

import re

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|f[eé]vr|avr|juin|juil|ao[uû]t|sept|d[eé]c)[a-z]*"
DATE_RE = re.compile(
    r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$"                       # 20/03/2008, 2008-03-20
    rf"|^{_MONTH}\.?,?$"                                              # a month name on its own
    rf"|^\d{{1,2}}[/\-. ]{_MONTH}[/\-. ,]*\d{{2,4}}$"                 # 20-Mar-2008  (FATURA's format)
    rf"|^{_MONTH}[/\-. ]\d{{1,2}}[/\-. ,]+\d{{2,4}}$",                # Mar-20-2008, Mar 20, 2008
    re.I,
)
MONEY_RE = re.compile(r"^[-(]?[$€£]?\d[\d,.\s]*[$€£]?\)?$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"(https?://|www\.|\.com\b|\.org\b|\.net\b|\.ma\b|\.fr\b)", re.I)
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")


def parse_amount(t: str) -> float | None:
    """'$1,234.50' -> 1234.5 ; '1.234,50' -> 1234.5 ; None when it is not an amount."""
    core = re.sub(r"[^\d.,]", "", t)
    if not core or not any(c.isdigit() for c in core):
        return None
    last_dot, last_com = core.rfind("."), core.rfind(",")
    if last_dot >= 0 and last_com >= 0:
        dec = max(last_dot, last_com)
        core = core[:dec].replace(".", "").replace(",", "") + "." + core[dec + 1:]
    elif last_com >= 0:
        core = core.replace(",", ".") if len(core) - last_com - 1 in (1, 2) and core.count(",") == 1 else core.replace(",", "")
    elif core.count(".") > 1:
        core = core.replace(".", "")
    try:
        return float(core)
    except ValueError:
        return None
