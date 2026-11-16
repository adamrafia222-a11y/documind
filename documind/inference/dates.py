"""Dates in every form a cheque, a contract or an invoice uses - not only 22/01/2006:

    1 decembre 2005   1er décembre 2005   le 1 Déc. 05   December 1st, 2005   1-Dec-2005   ١ ديسمبر ٢٠٠٥   l décembre 2OO5 (OCR slips)   2005-12-01   22 01 2006

Months are known in French, English and Arabic, with or without accents, abbreviated or not; a month the OCR damaged ("decembe", "Jancier") is matched to the
closest name. Days and years may contain the usual OCR confusions (l / I / | for 1, O for 0). 2-digit years: 00-69 = 20xx, 70-99 = 19xx. d/m/y is read day first
(m/d/y only when the "day" cannot be a day: 12/25/2005). Numbers written out in letters ("vingt-deux") are NOT handled.

    parse_date("1 decembre 2005")            -> date(2005, 12, 1)       None when there is no valid date
    find_dates("Fait à Rabat, le 1er déc. 2005")  -> [Found(date(2005, 12, 1), text='1er déc. 2005', start=15, end=28, kind='named')]
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

_MONTHS_FULL = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    # Arabic: Gulf / Egypt spelling and Maghreb spelling
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "أبريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7, "اغسطس": 8, "أغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    "جانفي": 1, "فيفري": 2, "افريل": 4, "أفريل": 4, "ماي": 5, "جوان": 6, "جويلية": 7, "اوت": 8, "أوت": 8, "شتنبر": 9, "غشت": 8, "نونبر": 11, "دجنبر": 12,
}
_MONTHS_ABBR = {"janv": 1, "jan": 1, "fev": 2, "fevr": 2, "feb": 2, "mar": 3, "avr": 4, "apr": 4, "jun": 6, "juil": 7, "jul": 7, "aou": 8, "aug": 8, "sep": 9, "sept": 9,
                "oct": 10, "nov": 11, "dec": 12}
_FILLER = {"le", "la", "du", "de", "d", "of", "the", "l", "on", "en", "in", "fait", "a", "au", "at"}
_SUFFIX = {"er", "re", "e", "eme", "ere", "st", "nd", "rd", "th"}
_LOOK = str.maketrans({"l": "1", "I": "1", "i": "1", "|": "1", "!": "1", "O": "0", "o": "0", "Q": "0", "S": "5", "B": "8", "Z": "2"})
_ARABIC_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")} | {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}


@dataclass(frozen=True)
class Found:
    date: date
    text: str
    start: int
    end: int
    kind: str                       # "numeric" | "named"


def _fold1(c: str) -> str:
    """One character in, one character out (positions stay aligned with the original text): lower case, accents removed."""
    d = unicodedata.normalize("NFKD", c)
    base = "".join(x for x in d if not unicodedata.combining(x)) or c
    return base[0].lower()


def _fold(text: str) -> str:
    return "".join(_fold1(c) for c in (text or "").translate(_ARABIC_DIGITS))


def month_of(token: str) -> int | None:
    """Month number of a word: exact (full or abbreviated name), else the closest full name when the word is long enough ('decembe', 'janvlier')."""
    t = _fold(token).strip(".")
    if not t or t.isdigit():
        return None
    if t in _MONTHS_FULL:
        return _MONTHS_FULL[t]
    if t in _MONTHS_ABBR:
        return _MONTHS_ABBR[t]
    if len(t) >= 4 and t.isalpha():
        t2 = t.translate(str.maketrans({"1": "l", "0": "o", "5": "s"}))
        m = difflib.get_close_matches(t2, [k for k in _MONTHS_FULL if k.isascii()], n=1, cutoff=0.8)
        if m:
            return _MONTHS_FULL[m[0]]
    return None


def _num(token: str, lo: int, hi: int, digits: tuple[int, ...], strict: bool = False) -> int | None:
    """Digits (OCR confusions allowed unless `strict`: 'l' = 1, 'O' = 0) -> int, if the number of digits and the value fit."""
    if strict and not token.isdigit():
        return None
    t = token.translate(_LOOK) if not token.isdigit() else token
    if not t.isdigit() or len(t) not in digits:
        return None
    if token.isalpha() and token not in ("l", "I", "i", "|"):                              # a word is not a number, except the usual stand-ins for "1"
        return None
    v = int(t)
    return v if lo <= v <= hi else None


def _day(token: str, strict: bool = False) -> int | None:
    """A day of the month: 5, 05, 1er, 22nd, premier / first, and (unless `strict`) the OCR stand-ins for 1 ('l')."""
    if (v := _num(token, 1, 31, (1, 2), strict)) is not None:
        return v
    f = _fold(token)
    if f in ("premier", "premiere", "first"):
        return 1
    m = re.fullmatch(r"(\d{1,2})(?:er|re|e|eme|st|nd|rd|th)", f)
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 31 else None


def _year(v: int, digits: int) -> int:
    return v if digits == 4 else (2000 + v if v <= 69 else 1900 + v)


def _mk(d: int, m: int, y: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


_TOKEN = re.compile(r"[^\W_]+|[|!]")
_NUMERIC = re.compile(r"(?<![\d/.\-])(\d{1,2})\s*([/.\-])\s*(\d{1,2})\s*\2\s*(\d{4}|\d{2})(?![\d])|(?<![\d])(\d{4})\s*([/.\-])\s*(\d{1,2})\s*\6\s*(\d{1,2})(?![\d])")


def find_dates(text: str, strict: bool = False) -> list[Found]:
    """Every date in a text, with its place: numeric (22/01/2006, 2006-01-22) and with a month name (1 decembre 2005, December 1st 2005, 1-Dec-05).
    `strict`: digits must be digits (an OCR slip such as 2OO5 or l décembre is not a date): used where the date's FORMAT is what is being judged."""
    src = (text or "").translate(_ARABIC_DIGITS)
    out: list[Found] = []
    for m in _NUMERIC.finditer(src):
        if m.group(1) is not None:
            a, b, y = int(m.group(1)), int(m.group(3)), _year(int(m.group(4)), len(m.group(4)))
            d = _mk(a, b, y)                                                                           # d/m/y ...
            if d is None and b > 12 >= a:
                d = _mk(b, a, y)                                                                       # ... m/d/y when the "month" is above 12
        else:
            d = _mk(int(m.group(8)), int(m.group(7)), int(m.group(5)))
        if d:
            out.append(Found(d, src[m.start():m.end()], m.start(), m.end(), "numeric"))
    toks = list(_TOKEN.finditer(src))
    for i, tk in enumerate(toks):
        mo = month_of(tk.group(0))
        if mo is None:
            continue
        # left side: a day, maybe followed by a suffix (1er, 22nd) and fillers (of / de / du), written before the month
        start = tk.start()
        j = i - 1
        while j >= 0 and _fold(toks[j].group(0)) in (_FILLER - {"l"}) and not toks[j].group(0).isdigit():       # ("l" may be a "1" the OCR misread)
            j -= 1
        day = None
        if j >= 0:
            day = _day(toks[j].group(0), strict)
            start = toks[j].start()
            if day is None and _fold(toks[j].group(0)) in _SUFFIX and j >= 1:                                   # "1 er" / "22 nd": split by the OCR
                day, start, j = _day(toks[j - 1].group(0), strict), toks[j - 1].start(), j - 1
        # right side: a 4-digit or 2-digit year (month-first forms: "December 1st, 2005": the day comes first, then the year)
        k = i + 1
        while k < len(toks) and _fold(toks[k].group(0)) in _FILLER | {"l", "an", "year", "annee"} and not toks[k].group(0).isdigit():
            k += 1
        year = end = None
        if k < len(toks):
            t = toks[k].group(0)
            v4 = _num(t, 1900, 2100, (4,), strict)
            if v4 is not None:
                year, end = v4, toks[k].end()
            elif day is not None and (v2 := _num(t, 0, 99, (2,), strict)) is not None:
                year, end = _year(v2, 2), toks[k].end()
            elif day is None and (d2 := _day(t, strict)) is not None:                                                      # month first: December 1st, 2005
                day, start, k2 = d2, tk.start(), k + 1
                if k2 < len(toks) and _fold(toks[k2].group(0)) in _SUFFIX:
                    k2 += 1
                while k2 < len(toks) and _fold(toks[k2].group(0)) in _FILLER and not toks[k2].group(0).isdigit():
                    k2 += 1
                if k2 < len(toks) and (v := _num(toks[k2].group(0), 1900, 2100, (4,), strict)) is not None:
                    year, end = v, toks[k2].end()
                elif k2 < len(toks) and (v := _num(toks[k2].group(0), 0, 99, (2,), strict)) is not None:
                    year, end = _year(v, 2), toks[k2].end()
        if day is not None and year is not None and (d := _mk(day, mo, year)):
            start = min(start, tk.start())
            out.append(Found(d, src[start:end], start, end, "named"))
    out.sort(key=lambda f: (f.start, -(f.end - f.start)))
    clean: list[Found] = []
    for f in out:                                                                                                   # keep the longest of overlapping finds
        if clean and f.start < clean[-1].end:
            continue
        clean.append(f)
    return clean


_LOOSE = re.compile(r"^\D*(\d{1,2})[\s,.\-/]+(\d{1,2})[\s,.\-/]+(\d{4}|\d{2})\D*$")


def parse_date(text: str, strict: bool = False) -> date | None:
    """A date field read by the OCR -> a date, or None. The text may be the whole box ('le 1 decembre 2005') or hold stray characters.
    `strict` = judge the format as read: no OCR-slip repair (2OO5), no loose '22 01 2006' (see find_dates)."""
    s = (text or "").strip()
    if not s:
        return None
    if f := find_dates(s, strict):
        return max(f, key=lambda x: (x.kind == "named", x.end - x.start)).date
    if strict:
        return None
    t = s.translate(_ARABIC_DIGITS)
    fixed = "".join(c.translate(_LOOK) if not c.isdigit() and re.search(r"\d", t) and not c.isspace() and c not in "/.-," else c for c in t)   # 2O05 -> 2005 (only when digits are there)
    for cand in (t, fixed):
        if f := find_dates(cand):
            return f[0].date
        if m := _LOOSE.match(cand):                                                                                  # 22 01 2006 / 22,01,2006
            a, b, y = int(m.group(1)), int(m.group(2)), _year(int(m.group(3)), len(m.group(3)))
            if d := _mk(a, b, y) or (_mk(b, a, y) if b > 12 >= a else None):
                return d
    digits = re.sub(r"\D", "", fixed)
    if re.fullmatch(r"\d{8}", digits):                                                                               # 22012006
        return _mk(int(digits[:2]), int(digits[2:4]), int(digits[4:]))
    return None
