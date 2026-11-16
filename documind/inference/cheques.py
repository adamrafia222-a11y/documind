"""Read a cheque picture: the YOLO detector finds the bank name, cheque number, amount in words, payee, amount in figures, place, date,
signature, account number (RIB / IBAN) and MICR line; Tesseract reads each box; the values are parsed and CHECKED against each other.

    python -m documind.inference.cheques cheque.jpg            # prints the fields, the checks and the one-line CSV row
    python -m documind.inference.cheques cheque.jpg --json     # the full result (boxes included)

Checks (what a bank teller would look at):
  * amount in figures = amount in words (the words are converted back to a number: "mille deux cent euros et cinquante centimes" = 1200.50)
  * the cheque number printed at the top = the one in the MICR line
  * the date is a real date, the account number has the expected length (24 digits RIB, 23 IBAN), a signature was found
A failed check becomes a warning in the result (shown on the web page); it never silently corrects a value.

Honest limits: Tesseract reads printed text and clean handwriting-like fonts well, real cursive handwriting badly. The functions
`read_text` / `parse_*` are small and separate: to read real handwriting, replace `read_text` by a handwriting model (e.g. TrOCR).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from datetime import date as _date
from pathlib import Path

from PIL import Image

from ..config import MODELS_DIR
from .contract_text import english_words_to_number
from .receipts import DATE_RE, _open_receipt_image, ocr_box

MODEL_PATH = MODELS_DIR / "cheques_yolo.pt"
_model = None

FIELD_LABELS = {"bank_name": "BANK", "cheque_number": "CHEQUE_NO", "amount_words": "AMOUNT_WORDS", "payee": "PAYEE", "amount_digits": "AMOUNT", "place": "PLACE",
                "date": "DATE", "signature": "SIGNATURE", "account_number": "ACCOUNT_NO", "micr_line": "MICR"}
CHEQUE_COLUMNS = ["file", "BANK", "CHEQUE_NO", "PAYEE", "AMOUNT", "CURRENCY", "AMOUNT_WORDS", "PLACE", "DATE", "ACCOUNT_NO", "MICR", "SIGNATURE", "AMOUNT_CHECK"]
ESSENTIAL = ("amount_digits", "amount_words", "payee", "date", "signature")        # a cheque without one of these cannot be processed
# no whitelist on "date": a written date ("1 decembre 2005") has letters, and a digit whitelist also drops a "1" the OCR sees as "l" (see dates.py)
WHITELIST = {"cheque_number": "0123456789", "amount_digits": "0123456789,.- ", "account_number": "0123456789 ", "micr_line": "0123456789<:. "}
# handwriting is big (40-60 px tall): Tesseract reads it best at its own size, NOT enlarged (tested: payee 23/24 at x1.0, 18/24 at x2; amount 24/24 vs 23/24)
OCR_PARAMS = {"payee": dict(pad=6, scale=1.0), "amount_words": dict(pad=6, scale=1.0), "place": dict(pad=6, scale=1.0), "amount_digits": dict(pad=6, scale=1.0),
              "date": dict(pad=6, scale=1.0)}
EXTRA_PASSES = [dict(pad=6, scale=0.7), dict(pad=6, scale=1.4)]         # tried on the amount in words only when the first reading does not agree with the figures


# ------------------------------------------------------------------------------------------------ French number words -> number
_UNITS = {"zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
          "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16, "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60}
_CURRENCY = {"dirham": "MAD", "dirhams": "MAD", "euro": "EUR", "euros": "EUR"}
_CENTIME = {"centime", "centimes"}
_VOCAB = list(_UNITS) + ["cent", "cents", "mille", "million", "millions", "et", "de"] + list(_CURRENCY) + list(_CENTIME)


def _norm(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z]+", " ", t).split()


def _snap(tok: str) -> str | None:
    """A word of the vocabulary, or the closest one (the OCR reads 'qvarante' for 'quarante'); None if nothing is close."""
    if tok in _VOCAB:
        return tok
    m = difflib.get_close_matches(tok, _VOCAB, n=1, cutoff=0.72)
    return m[0] if m else None


def _words_to_int(tokens: list[str]) -> int | None:
    total = cur = 0
    prev = ""
    for tok in tokens:
        if tok in ("et", "de"):                                  # "et" (soixante et onze) and "de" (un million de dirhams) carry no value
            continue
        if tok in _UNITS:
            if tok in ("vingt", "vingts") and prev == "quatre":
                cur += 76                                        # quatre-vingt: the 4 already added + 76 = 80
            else:
                cur += _UNITS[tok]
        elif tok in ("cent", "cents"):
            cur = (cur or 1) * 100
        elif tok == "mille":
            total += (cur or 1) * 1000
            cur = 0
        elif tok in ("million", "millions"):
            total += (cur or 1) * 1_000_000
            cur = 0
        else:
            return None
        prev = tok
    return total + cur


def words_to_number(text: str) -> tuple[int, int, str] | None:
    """French amount in words (up to 999 999 999) -> (units, cents, currency 'MAD' / 'EUR' / '') or None when it cannot be understood.
    'Mille deux cent trente dirhams et cinquante centimes' -> (1230, 50, 'MAD')."""
    toks = []
    for w in _norm(text or ""):
        s = _snap(w)
        if s is None:
            return None
        toks.append(s)
    if not toks:
        return None
    currency, units_t, cents_t, i = "", [], [], 0
    cut = next((k for k, t in enumerate(toks) if t in _CURRENCY), None)
    if cut is not None:
        currency, units_t, rest = _CURRENCY[toks[cut]], toks[:cut], toks[cut + 1:]
    else:
        k = next((k for k, t in enumerate(toks) if t in _CENTIME), None)
        units_t, rest = (toks, []) if k is None else (toks[:0], toks)             # "cinquante centimes" alone: no units
    cents_t = [t for t in rest if t not in _CENTIME]
    units = _words_to_int(units_t) if units_t else 0
    cents = _words_to_int(cents_t) if cents_t else 0
    if units is None or cents is None or cents > 99:
        return None
    return units, cents, currency


# ------------------------------------------------------------------------------------------------ parsing the other fields
def _today():
    from datetime import date as _d
    return _d.today()


def _parse_iso(dmy: str):
    """'27/05/2024' -> date, None if it is not dd/mm/yyyy."""
    from datetime import datetime
    try:
        return datetime.strptime(dmy, "%d/%m/%Y").date()
    except ValueError:
        return None


def parse_amount(text: str) -> float | None:
    """'12 345,50' / '1200.5' / '450' -> 12345.5 / 1200.5 / 450.0; None if there is no number."""
    t = (text or "").replace("\u00a0", " ").strip()
    t = re.sub(r"(\d)\s*[-–—_]+\s*(\d{2})(?!\d)", r"\1.\2", t)               # 100-00 / 100—00 = 100.00 (pounds-pence with a dash)
    t = re.sub(r"(?<=\d)\s(\d{2})(?!\d)(?=\s*(?:MAD|DHS?|EUR|GBP|USD|[£$€])?\s*$)", r".\1", t)      # "150 00" / "1 250 00": a last group of TWO digits is the cents, not thousands
    m = re.search(r"\d[\d ]*(?:[.,]\d{1,2})?", t)
    if not m:
        return None
    num = m.group(0).replace(" ", "").replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def parse_date(text: str) -> tuple[str, bool]:
    """-> (date text as read, is it a real calendar date?)"""
    m = DATE_RE.search(text or "")
    if not m:                                                     # "1 decembre 2005", "December 1st, 2005", "1er déc. 05" ...
        from .dates import parse_date
        d = parse_date(text or "")
        return ((d.strftime("%d/%m/%Y"), 1990 <= d.year <= 2100) if d else ((text or "").strip(), False))
    s = m.group(0).replace(" ", "")
    parts = re.split(r"[-/.]", s)
    try:
        d, mo, y = (int(p) for p in parts) if len(parts[0]) <= 2 else (int(parts[2]), int(parts[1]), int(parts[0]))
        if y < 100:
            y += 2000
        _date(y, mo, d)
        return s, 1990 <= y <= 2100
    except ValueError:
        return s, False


def parse_digits(text: str, minimum: int = 1) -> str:
    return "".join(c for c in (text or "") if c.isdigit()) if len("".join(c for c in (text or "") if c.isdigit())) >= minimum else ""


def micr_cheque_number(micr: str) -> str:
    """The cheque number printed between the first two '<' of the MICR line ('<1234567<  ...')."""
    m = re.search(r"<\s*(\d{5,})\s*<", micr or "")
    return m.group(1) if m else ""


# ------------------------------------------------------------------------------------------------ detector + reading
def model_available(path: Path = MODEL_PATH) -> bool:
    return Path(path).exists()


def load_model(path: Path = MODEL_PATH):
    global _model
    if _model is None:
        if not Path(path).exists():
            raise SystemExit(f"{path} not found. Train it first:  python -m documind.data.prepare_cheques  then  python -m documind.training.train_cheques")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("Ultralytics is not installed in this environment. Run:  pip install -r requirements-receipts.txt")
        _model = YOLO(str(path))
    return _model


def read_text(img: Image.Image, box, field: str, lang: str | None = None) -> str:
    """Text of one detected box. The only place that calls an OCR engine: replace it to use a handwriting model."""
    return ocr_box(img, box, psm=7, lang=lang, whitelist=WHITELIST.get(field), field=field, **OCR_PARAMS.get(field, {}))


def fields_from_boxes(img: Image.Image, dets: list[tuple], lang: str | None = None) -> dict:
    """dets: [(class_name, [x0, y0, x1, y1], confidence)] -> {class: {"value", "raw_text", "box_confidence", "bbox"}} (the most confident box of each class)."""
    best: dict = {}
    for name, xyxy, conf in dets:
        if name in FIELD_LABELS and (name not in best or conf > best[name][2]):
            best[name] = (name, xyxy, conf)
    out = {}
    for name, xyxy, conf in best.values():
        readings = None
        if name == "date":                                         # several readings of the box vote for the date (handwriting model, Tesseract at 3 sizes)
            from .date_reader import read_date_box
            res_d = read_date_box(img, xyxy, lang)
            raw, readings = res_d["text"], res_d["readings"]
        else:
            raw = "" if name == "signature" else read_text(img, xyxy, name, lang)
        out[name] = {"value": "signed" if name == "signature" else raw, "raw_text": raw, "box_confidence": round(float(conf), 2), "bbox": [round(float(v), 1) for v in xyxy]}
        if readings is not None:
            out[name]["readings"] = readings
    if "amount_words" in out:                                    # the long handwritten line is the hardest to read: a second opinion only when needed
        amount = parse_amount(out["amount_digits"]["raw_text"]) if "amount_digits" in out else None
        agrees = lambda t: (w := (words_to_number(t) or english_words_to_number(t))) is not None and (amount is None or abs(w[0] + w[1] / 100 - amount) < 0.005)
        if not agrees(out["amount_words"]["raw_text"]):
            box = out["amount_words"]["bbox"]
            alts = [ocr_box(img, box, psm=7, lang=lang, field="amount_words", **kw) for kw in EXTRA_PASSES]
            out["amount_words"]["alternatives"] = alts
            best_alt = next((t for t in alts if agrees(t)), None)
            if best_alt is not None:
                out["amount_words"]["raw_text"] = out["amount_words"]["value"] = best_alt
    return out


def interpret(got: dict) -> dict:
    """Parse the raw texts and run the checks -> {"values": {...}, "checks": {...}, "warnings": [...]}."""
    v, checks, warnings = {}, {}, []
    g = lambda k: (got.get(k) or {}).get("raw_text", "")
    amount = parse_amount(g("amount_digits")) if "amount_digits" in got else None
    zero_amount = amount is not None and amount <= 0
    if zero_amount:                                                # "0.00": the handwriting was not read, do not present it as an amount
        amount = None
        warnings.append({"code": "amount_unreadable", "message": "The amount in figures was read as zero, which no cheque has: the handwriting could not be read."})
    words = (words_to_number(g("amount_words")) or english_words_to_number(g("amount_words"))) if "amount_words" in got else None
    v["amount"] = f"{amount:.2f}" if amount is not None else ("" if zero_amount else g("amount_digits"))
    v["amount_words"] = g("amount_words")
    sign = re.search(r"[£$€]|\b(MAD|DHS?|EUR|GBP|USD)\b", g("amount_digits"))
    v["currency"] = (words[2] if words else "") or ({"£": "GBP", "$": "USD", "€": "EUR", "DH": "MAD", "DHS": "MAD"}.get(sign.group(0), sign.group(0)) if sign else "")
    if amount is not None and words is not None:
        wv = words[0] + words[1] / 100
        checks["amount_match"] = abs(wv - amount) < 0.005
        if not checks["amount_match"]:
            warnings.append({"code": "amount_mismatch", "message": f"The amount in figures ({amount:,.2f}) and the amount in words ({wv:,.2f}) are different. Check the cheque."})
    else:
        checks["amount_match"] = None
        if "amount_digits" in got and "amount_words" in got:
            warnings.append({"code": "amount_not_checked", "message": "The amounts in figures and in words could not both be read as numbers, so they were not compared."})
    v["amount_check"] = {True: "ok", False: "mismatch", None: "not checked"}[checks["amount_match"]]
    v["cheque_no"] = parse_digits(g("cheque_number"), 5) or g("cheque_number")
    v["micr"] = g("micr_line")
    mn = micr_cheque_number(g("micr_line"))
    checks["micr_number_match"] = (mn == v["cheque_no"]) if (mn and v["cheque_no"]) else None
    if checks["micr_number_match"] is False:
        warnings.append({"code": "micr_mismatch", "message": f"The cheque number ({v['cheque_no']}) differs from the one in the MICR line ({mn})."})
    if "date" in got:
        v["date"], checks["date_valid"] = parse_date(g("date"))
        if not checks["date_valid"]:
            warnings.append({"code": "date_invalid", "message": f"The date '{v['date']}' is not a valid date."})
        elif (d := _parse_iso(v["date"])) is not None and (d - _today()).days > 366:
            warnings.append({"code": "date_in_future", "message": f"The date {v['date']} is more than a year ahead: a handwritten digit may have been misread (check the year)."})
    acc = parse_digits(g("account_number"), 1)
    v["account_no"] = acc or g("account_number")
    if "account_number" in got:
        checks["account_length_ok"] = len(acc) in (23, 24)
        if not checks["account_length_ok"]:
            warnings.append({"code": "account_length", "message": f"The account number has {len(acc)} digits (a RIB has 24, an IBAN 23): check it."})
    v["bank"], v["payee"], v["place"] = g("bank_name"), g("payee"), g("place")
    checks["signature_present"] = "signature" in got
    if not checks["signature_present"]:
        warnings.append({"code": "no_signature", "message": "No signature was detected."})
    missing = [k for k in ESSENTIAL if k not in got and k != "signature"]
    if missing:
        warnings.append({"code": "missing_field", "message": "Not found on the picture: " + ", ".join(missing) + ". Is this a cheque, and is the picture sharp?"})
    return {"values": v, "checks": checks, "warnings": warnings}


LAYOUT_FIELDS = ("date", "payee", "amount_digits", "amount_words")


def payee_ok(raw: str) -> bool:
    """A payee is a name. A date, number words, digits or symbols read in that box are another field's text (or noise)."""
    t = (raw or "").strip()
    letters, digits = sum(c.isalpha() for c in t), sum(c.isdigit() for c in t)
    if letters < 2 or digits > 3 or digits * 2 >= letters:                           # "3M Ltd" is a name, "12 05 2025" is not
        return False
    if _is_amount_words(t) or parse_date(t)[1]:
        return False
    return sum(1 for c in t if not (c.isalnum() or c in " .,'’&-")) <= 0.1 * len(t)


def _readable(name: str, raw: str) -> bool:
    """Is what the detector read usable for this field? (if not, the layout reader tries)"""
    if name == "date":
        return parse_date(raw)[1]
    if name == "amount_digits":
        return (parse_amount(raw) or 0) > 0                                                  # 0.00 is what garbage looks like, never a cheque amount
    if name == "amount_words":
        return bool(words_to_number(raw) or english_words_to_number(raw))
    return payee_ok(raw)


def _is_amount_words(raw: str) -> bool:
    """'One hundred and fifty', 'Mille deux cents dirhams': a sentence of number words (two at least)."""
    return len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", raw or "")) >= 2 and bool(words_to_number(raw) or english_words_to_number(raw))


def fix_swapped(got: dict) -> dict:
    """The detector sometimes labels the two handwritten lines the wrong way round (payee <-> amount in words). The TEXT tells which is which: number words are the
    amount in words, whatever the box was called."""
    pw, aw = got.get("payee"), got.get("amount_words")
    if pw and _is_amount_words(pw["raw_text"]) and not (aw and _is_amount_words(aw["raw_text"])):
        out = dict(got)
        if aw:
            out["payee"], out["amount_words"] = {**aw, "swapped": True}, {**pw, "swapped": True}
        else:
            out["amount_words"] = {**pw, "swapped": True}
            del out["payee"]
        return out
    return got


def _value_of_words(raw: str) -> float | None:
    w = words_to_number(raw) or english_words_to_number(raw)
    return (w[0] + w[1] / 100) if w else None


def reconcile_amounts(img, got: dict) -> dict:
    """The amount in figures and the amount in words disagree (48.50 / "Quarante euros et cinquante centimes": the model dropped "huit"). Ask the recognizers for their
    next-best readings of the WORDS and keep the first one that agrees with the figures; failing that, the next-best readings of the FIGURES that agree with the words.
    Nothing is invented: a reading is only replaced by another reading of the same box, and still disagreeing amounts keep their warning."""
    aw, ad = got.get("amount_words"), got.get("amount_digits")
    if not aw or not ad or not aw.get("bbox") or not ad.get("bbox"):
        return got
    wv, dv = _value_of_words(aw["raw_text"]), parse_amount(ad["raw_text"])
    if wv is None or dv is None or abs(wv - dv) < 0.005:
        return got
    from ..ocr import handwriting, trocr

    def crop(box):
        x0, y0, x1, y1 = box
        return img.crop((max(0, int(x0) - 3), max(0, int(y0) - 3), min(img.width, int(x1) + 4), min(img.height, int(y1) + 4)))

    engines = [(n, e) for n, e in (("handwriting", handwriting.get_engine()), ("trocr", trocr.get_engine())) if e is not None]
    out = dict(got)
    for name, e in engines:
        for t in handwriting.safe_nbest(e, crop(aw["bbox"]), 5, name):
            if (v := _value_of_words(t)) is not None and abs(v - dv) < 0.005:
                out["amount_words"] = {**aw, "value": t, "raw_text": t, "reconciled": True, "alternatives": [aw["raw_text"]]}
                return out
    for name, e in engines:
        for t in handwriting.safe_nbest(e, crop(ad["bbox"]), 5, name):
            t2 = re.sub(r"[^\d.,\- £$€]", "", t.replace("–", "-").replace("—", "-"))
            if t2 and (v := parse_amount(t2)) is not None and abs(v - wv) < 0.005:
                out["amount_digits"] = {**ad, "value": t2, "raw_text": t2, "reconciled": True, "alternatives": [ad["raw_text"]]}
                return out
    return got


def merge_layout(got: dict, layout: dict, need: set) -> dict:
    """Detector fields stay; each field in `need` comes from the layout reading when there is one. The detector often puts "amount in words" on the PAYEE line:
    when the layout found the real amount in words, that text is the payee (unless the layout found a payee itself)."""
    out = dict(got)
    old_aw = got.get("amount_words")
    if old_aw and "amount_words" in need and "amount_words" in layout and "payee" not in layout and "payee" in need \
            and not re.search(r"\d", old_aw["raw_text"]) and len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", old_aw["raw_text"])) >= 1:
        from .cheque_text import _is_number_word
        toks = old_aw["raw_text"].split()
        if sum(_is_number_word(t) for t in toks) / max(1, len(toks)) < 0.4:               # mostly NOT number words: it is a name, not a badly read amount
            out["payee"] = {**old_aw}
    for name, f in layout.items():
        if name in need:
            out[name] = f
    return out


def extract_cheque_bytes(data: bytes, filename: str = "cheque", conf: float = 0.25, lang: str | None = None, model=None) -> dict:
    """Picture / PDF bytes -> the same kind of dict as the other extractors (fields, items, row, columns, warnings, doc_type) plus `checks`."""
    img, n_pages = _open_receipt_image(data, filename)
    dets, notes = [], []
    try:
        model = model or load_model()
        r = model.predict(img, imgsz=int(model.overrides.get("imgsz", 1280)), conf=conf, verbose=False)[0]
        dets = [(model.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes]
    except SystemExit:                                            # no trained detector / no ultralytics: the layout reader below still works
        notes.append({"code": "detector_missing", "message": "The cheque detector is not installed: the cheque was read from its layout only."})
    got = fix_swapped(fields_from_boxes(img, dets, lang))
    need = {k for k in LAYOUT_FIELDS if k not in got or not _readable(k, got[k]["raw_text"])}
    if need:                                                      # date / payee / amount missed or unreadable: locate them on the page and read them
        try:
            from .cheque_text import read_cheque_layout
            aw = got.get("amount_words")
            wv = (words_to_number(aw["raw_text"]) or english_words_to_number(aw["raw_text"])) if aw else None
            hints = {k: got[k]["bbox"] for k in ("amount_words", "amount_digits", "signature", "date", "payee") if k in got and got[k].get("bbox") and k not in need}     # where the detector already found things
            got = merge_layout(got, read_cheque_layout(img, lang, need, hint_amount=(wv[0] + wv[1] / 100) if wv else None, hints=hints), need)
        except Exception as exc:                                  # never lose the detector result because the second reading failed ...
            import logging
            logging.getLogger("documind").exception("cheque layout reading failed")
            notes.append({"code": "layout_failed", "message": f"The second reading of the cheque (date, payee, amounts found from the page layout) failed: {type(exc).__name__}: {str(exc)[:200]}"})   # ... but say it
    got = reconcile_amounts(img, got)
    res = interpret(got)
    from ..ocr import handwriting
    if w := handwriting.unavailable_warning():                    # say it in the result: otherwise "the handwritten date is not detected" has no visible reason
        notes.append(w)
    for k, x in got.items():
        if x.get("unread"):
            notes.append({"code": "handwriting_unread", "message": f"The {k.replace('_', ' ')} was found on the cheque but its handwriting could not be read. "
                                                                    + ("Load the handwriting recognizer (see the other warning) or type it in." if handwriting.unavailable_warning() else "Type it in, or try a sharper picture.")})
    res["warnings"] = notes + res["warnings"]
    v = res["values"]
    shown = {"amount_digits": v["amount"] if v["amount"] else None}
    fields = [{"label": FIELD_LABELS[k], "text": (shown.get(k) or x["value"] or x["raw_text"]), "confidence": x["box_confidence"], "bbox": x["bbox"], "page": 1, "source": x.get("source", "detector"), **({"readings": x["readings"]} if x.get("readings") else {})}
              for k, x in got.items() if x["value"] or x["raw_text"]]
    fields.sort(key=lambda f: (f["bbox"][1], f["bbox"][0]))
    warnings = list(res["warnings"])
    if n_pages > 1:
        warnings.append({"code": "first_page_only", "message": f"The PDF has {n_pages} pages: only the first page was read."})
    row = {"file": filename, "BANK": v["bank"], "CHEQUE_NO": v["cheque_no"], "PAYEE": v["payee"], "AMOUNT": v["amount"], "CURRENCY": v["currency"], "AMOUNT_WORDS": v["amount_words"],
           "PLACE": v["place"], "DATE": v.get("date", ""), "ACCOUNT_NO": v["account_no"], "MICR": v["micr"],
           "SIGNATURE": "yes" if res["checks"]["signature_present"] else "no", "AMOUNT_CHECK": v["amount_check"]}
    return {"file": filename, "doc_type": "cheque", "warnings": warnings, "n_pages": n_pages,
            "pages": [{"page": 1, "width": img.width, "height": img.height, "size": [img.width, img.height]}],
            "fields": fields, "items": [], "items_check": {}, "checks": res["checks"], "row": row, "columns": CHEQUE_COLUMNS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lang", default=None, help="Tesseract language(s), e.g. fra+eng")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true", help="show what each engine read in the date box")
    a = ap.parse_args()
    if not Path(a.picture).is_file():
        raise SystemExit(f"File not found: {a.picture}\nGive the path of YOUR cheque picture, e.g.  python -m documind.inference.cheques \"C:\\Users\\aa\\Pictures\\cheque1.jpg\" --debug")
    data = Path(a.picture).read_bytes()
    res = extract_cheque_bytes(data, Path(a.picture).name, a.conf, a.lang)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    for f in res["fields"]:
        print(f"{f['label']:13s} {f['text']}   (box confidence {f['confidence']:.2f})")
    if a.debug:
        for f in res["fields"]:
            if f.get("readings"):
                print(f"\n{f['label']} readings (engine: text) - a handwritten value that is not found: tell which engine read what:")
                for k, v in f["readings"].items():
                    print(f"   {k:18s} {v!r}")
    print("\nchecks:", res["checks"])
    for w in res["warnings"]:
        print("WARNING:", w["message"])
    print("\n" + ",".join(CHEQUE_COLUMNS) + "\n" + ",".join(f'"{res["row"].get(c, "")}"' for c in CHEQUE_COLUMNS))


if __name__ == "__main__":
    main()
