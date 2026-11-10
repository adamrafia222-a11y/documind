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
from .receipts import DATE_RE, _open_receipt_image, ocr_box

MODEL_PATH = MODELS_DIR / "cheques_yolo.pt"
_model = None

FIELD_LABELS = {"bank_name": "BANK", "cheque_number": "CHEQUE_NO", "amount_words": "AMOUNT_WORDS", "payee": "PAYEE", "amount_digits": "AMOUNT", "place": "PLACE",
                "date": "DATE", "signature": "SIGNATURE", "account_number": "ACCOUNT_NO", "micr_line": "MICR"}
CHEQUE_COLUMNS = ["file", "BANK", "CHEQUE_NO", "PAYEE", "AMOUNT", "CURRENCY", "AMOUNT_WORDS", "PLACE", "DATE", "ACCOUNT_NO", "MICR", "SIGNATURE", "AMOUNT_CHECK"]
ESSENTIAL = ("amount_digits", "amount_words", "payee", "date", "signature")        # a cheque without one of these cannot be processed
WHITELIST = {"cheque_number": "0123456789", "amount_digits": "0123456789,. ", "account_number": "0123456789 ", "date": "0123456789/-. ", "micr_line": "0123456789<:. "}
# handwriting is big (40-60 px tall): Tesseract reads it best at its own size, NOT enlarged (tested: payee 23/24 at x1.0, 18/24 at x2; amount 24/24 vs 23/24)
OCR_PARAMS = {"payee": dict(pad=6, scale=1.0), "amount_words": dict(pad=6, scale=1.0), "place": dict(pad=6, scale=1.0), "amount_digits": dict(pad=6, scale=1.0),
              "date": dict(pad=6, scale=1.0)}
EXTRA_PASSES = [dict(pad=6, scale=0.7), dict(pad=6, scale=1.4)]         # tried on the amount in words only when the first reading does not agree with the figures


# ------------------------------------------------------------------------------------------------ French number words -> number
_UNITS = {"zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
          "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16, "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60}
_CURRENCY = {"dirham": "MAD", "dirhams": "MAD", "euro": "EUR", "euros": "EUR"}
_CENTIME = {"centime", "centimes"}
_VOCAB = list(_UNITS) + ["cent", "cents", "mille", "et"] + list(_CURRENCY) + list(_CENTIME)


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
        if tok == "et":
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
        else:
            return None
        prev = tok
    return total + cur


def words_to_number(text: str) -> tuple[int, int, str] | None:
    """French amount in words -> (units, cents, currency 'MAD' / 'EUR' / '') or None when it cannot be understood.
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
def parse_amount(text: str) -> float | None:
    """'12 345,50' / '1200.5' / '450' -> 12345.5 / 1200.5 / 450.0; None if there is no number."""
    t = (text or "").replace("\u00a0", " ").strip()
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
    if not m:
        return (text or "").strip(), False
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
    return ocr_box(img, box, psm=7, lang=lang, whitelist=WHITELIST.get(field), **OCR_PARAMS.get(field, {}))


def fields_from_boxes(img: Image.Image, dets: list[tuple], lang: str | None = None) -> dict:
    """dets: [(class_name, [x0, y0, x1, y1], confidence)] -> {class: {"value", "raw_text", "box_confidence", "bbox"}} (the most confident box of each class)."""
    best: dict = {}
    for name, xyxy, conf in dets:
        if name in FIELD_LABELS and (name not in best or conf > best[name][2]):
            best[name] = (name, xyxy, conf)
    out = {}
    for name, xyxy, conf in best.values():
        raw = "" if name == "signature" else read_text(img, xyxy, name, lang)
        out[name] = {"value": "signed" if name == "signature" else raw, "raw_text": raw, "box_confidence": round(float(conf), 2), "bbox": [round(float(v), 1) for v in xyxy]}
    if "amount_words" in out:                                    # the long handwritten line is the hardest to read: a second opinion only when needed
        amount = parse_amount(out["amount_digits"]["raw_text"]) if "amount_digits" in out else None
        agrees = lambda t: (w := words_to_number(t)) is not None and (amount is None or abs(w[0] + w[1] / 100 - amount) < 0.005)
        if not agrees(out["amount_words"]["raw_text"]):
            box = out["amount_words"]["bbox"]
            alts = [ocr_box(img, box, psm=7, lang=lang, **kw) for kw in EXTRA_PASSES]
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
    words = words_to_number(g("amount_words")) if "amount_words" in got else None
    v["amount"] = f"{amount:.2f}" if amount is not None else g("amount_digits")
    v["amount_words"] = g("amount_words")
    v["currency"] = words[2] if words else ""
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


def extract_cheque_bytes(data: bytes, filename: str = "cheque", conf: float = 0.25, lang: str | None = None, model=None) -> dict:
    """Picture / PDF bytes -> the same kind of dict as the other extractors (fields, items, row, columns, warnings, doc_type) plus `checks`."""
    img, n_pages = _open_receipt_image(data, filename)
    model = model or load_model()
    r = model.predict(img, imgsz=int(model.overrides.get("imgsz", 1280)), conf=conf, verbose=False)[0]
    dets = [(model.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes]
    got = fields_from_boxes(img, dets, lang)
    res = interpret(got)
    v = res["values"]
    shown = {"amount_digits": v["amount"] if v["amount"] else None}
    fields = [{"label": FIELD_LABELS[k], "text": (shown.get(k) or x["value"] or x["raw_text"]), "confidence": x["box_confidence"], "bbox": x["bbox"], "page": 1, "source": "detector"}
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
    a = ap.parse_args()
    data = Path(a.picture).read_bytes()
    res = extract_cheque_bytes(data, Path(a.picture).name, a.conf, a.lang)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    for f in res["fields"]:
        print(f"{f['label']:13s} {f['text']}   (box confidence {f['confidence']:.2f})")
    print("\nchecks:", res["checks"])
    for w in res["warnings"]:
        print("WARNING:", w["message"])
    print("\n" + ",".join(CHEQUE_COLUMNS) + "\n" + ",".join(f'"{res["row"].get(c, "")}"' for c in CHEQUE_COLUMNS))


if __name__ == "__main__":
    main()
