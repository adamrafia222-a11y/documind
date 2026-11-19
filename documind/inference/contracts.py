"""Read a loan contract (personal loan, car loan, mortgage): the YOLO detector finds the key terms, the OCR reads them, the values are parsed and CHECKED.

    python -m documind.inference.contracts contract.jpg            # fields, checks, the CSV row
    python -m documind.inference.contracts contract.jpg --json     # the full result (boxes included)

A contract is running text: the amount, the rate or the IBAN sit inside a sentence, and a value that does not fit at the end of a line wraps onto the next
one (the detector then gives one box per line). The boxes of one field are put back in reading order (the page tilt is measured and removed first) and
their texts joined. The lender's name is printed twice (letterhead + first paragraph): the most confident reading is kept.

Checks (a failed check is a warning, a value is never silently corrected):
  * amount in figures = amount in words (the French words are converted back to a number, millions included)
  * monthly payment = the loan formula for the amount, the rate and the duration (within 1 %: insurance and fees make real contracts differ a little)
  * the dates are real, the first payment is after the contract date, the signing date is not before it
  * the IBAN has the length of its country (MA 28, FR 27), both signatures were found
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import date as _date
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR
from ..ocr.lines import reading_order, skew_angle
from .cheques import parse_amount, words_to_number
from .contract_text import english_words_to_number, read_contract_text
from .receipts import _open_receipt_image, ocr_box

MODEL_PATH = MODELS_DIR / "contracts_yolo.pt"
_model = None

FIELD_LABELS = {"title": "TITLE", "contract_number": "CONTRACT_NO", "contract_date": "CONTRACT_DATE", "lender_name": "LENDER", "borrower_name": "BORROWER",
                "borrower_id": "BORROWER_ID", "borrower_address": "BORROWER_ADDRESS", "amount": "AMOUNT", "amount_words": "AMOUNT_WORDS", "interest_rate": "RATE",
                "duration": "DURATION", "monthly_payment": "MONTHLY_PAYMENT", "first_payment_date": "FIRST_PAYMENT_DATE", "iban": "IBAN", "place": "PLACE",
                "signing_date": "SIGNING_DATE", "signature_lender": "SIGNATURE_LENDER", "signature_borrower": "SIGNATURE_BORROWER", "stamp": "STAMP"}
NO_TEXT = ("signature_lender", "signature_borrower", "stamp")                      # found or not found, nothing to read
CONTRACT_COLUMNS = ["file", "CONTRACT_TYPE", "TITLE", "CONTRACT_NO", "CONTRACT_DATE", "LENDER", "BORROWER", "BORROWER_ID", "BORROWER_ADDRESS", "AMOUNT", "CURRENCY",
                    "AMOUNT_WORDS", "RATE", "DURATION_MONTHS", "MONTHLY_PAYMENT", "FIRST_PAYMENT_DATE", "IBAN", "PLACE", "SIGNING_DATE", "SIGNED_LENDER",
                    "SIGNED_BORROWER", "STAMP", "AMOUNT_CHECK", "PAYMENT_CHECK"]
ESSENTIAL = ("amount", "amount_words", "interest_rate", "duration", "monthly_payment", "borrower_name", "lender_name", "iban")
IBAN_LENGTH = {"MA": 28, "FR": 27}
MONTHS = {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
          "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "fev": 2, "mar": 3, "apr": 4, "avr": 4, "jun": 6, "jul": 7, "juil": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}


def _fold(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (t or "").lower()) if not unicodedata.combining(c))


# ------------------------------------------------------------------------------------------------ parsing
def contract_type(title: str) -> str:
    t = _fold(title)
    return "mortgage" if re.search(r"immobilier|mortgage|home loan", t) else "car" if re.search(r"automobile|car loan|vehicle", t) else "personal" if re.search(r"personnel|personal", t) else ""


def parse_date_fr(text: str) -> _date | None:
    """'12/05/2024', '12 mars 2022', '1 decembre 2005', 'December 1st, 2005', '1er déc. 05' (accents optional, OCR slips tolerated) -> a date, or None. See dates.py."""
    from .dates import parse_date
    return parse_date(text, strict=True)                    # strict: used to judge the FORMAT of a reading (OCR slips such as 1O are not repaired here)


def parse_duration(text: str) -> int | None:
    """'36 mois' -> 36, '5 ans' -> 60."""
    m = re.search(r"(\d+)\s*(months?|years?|yrs?|mois|ans?|annees?)?", _fold(text))
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if (m.group(2) or "").startswith(("an", "annee", "year", "yr")) else n


def parse_rate(text: str) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", text or "")
    return float(m.group(0).replace(",", ".")) if m else None


def currency_of(text: str) -> str:
    t = _fold(text)
    return "MAD" if re.search(r"\b(mad|dh|dhs|dirhams?)\b", t) else "EUR" if ("€" in (text or "") or re.search(r"\b(eur|euros?)\b", t)) else ""


def annuity(amount: float, annual_rate: float, months: int) -> float:
    """Constant monthly payment of a loan: amount * r / (1 - (1 + r)^-n), r = monthly rate."""
    r = annual_rate / 1200.0
    return amount / months if r == 0 else amount * r / (1 - (1 + r) ** -months)


def clean_iban(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


# ------------------------------------------------------------------------------------------------ detector + reading
def model_available(path: Path = MODEL_PATH) -> bool:
    return Path(path).exists()


def load_model(path: Path = MODEL_PATH):
    global _model
    if _model is None:
        if not Path(path).exists():
            raise SystemExit(f"{path} not found. Train it first:  python -m documind.data.prepare_contracts  then  python -m documind.training.train_contracts")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("Ultralytics is not installed in this environment. Run:  pip install -r requirements.txt")
        _model = YOLO(str(path))
    return _model


def clean_edges(text: str) -> str:
    """The comma or full stop printed right after a value in a sentence is often inside the crop: strip punctuation from both ends."""
    return (text or "").strip(" ,.;:()«»\"'")


def _union(boxes: list[list[float]]) -> list[float]:
    return [round(float(v), 1) for v in (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))]


def fields_from_boxes(img, dets: list[tuple], angle: float | None = None, lang: str | None = None) -> dict:
    """dets: [(class_name, [x0, y0, x1, y1], confidence)] -> {class: {"value", "raw_text", "box_confidence", "bbox", "n_boxes"}}.
    One line per box, joined in reading order (`angle` = page tilt, measured on the picture when not given)."""
    by: dict[str, list[tuple]] = {}
    for name, xyxy, conf in dets:
        if name in FIELD_LABELS:
            by.setdefault(name, []).append((xyxy, float(conf)))
    if angle is None:
        angle = 0.0
        if any(len(v) > 1 and k not in ("lender_name",) + NO_TEXT for k, v in by.items()):
            import cv2
            angle = skew_angle(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY))
    out = {}
    for name, items in by.items():
        boxes = [b for b, _ in items]
        if name in NO_TEXT:
            out[name] = {"value": "signed" if name != "stamp" else "stamp", "raw_text": "", "box_confidence": round(max(c for _, c in items), 2), "bbox": _union(boxes), "n_boxes": len(items)}
            continue
        if name == "lender_name":                                   # letterhead + first paragraph: the same name twice, keep the more confident reading
            xyxy, conf = max(items, key=lambda t: t[1])
            text = clean_edges(ocr_box(img, xyxy, psm=7, lang=lang, field=name))
            out[name] = {"value": text, "raw_text": text, "box_confidence": round(conf, 2), "bbox": [round(float(v), 1) for v in xyxy], "n_boxes": len(items)}
            continue
        if name in ("contract_date", "first_payment_date", "signing_date"):          # a (handwritten) date: several readings of the box vote for it
            from .date_reader import read_date_box
            xyxy, conf = max(items, key=lambda t: t[1])
            vote = read_date_box(img, xyxy, lang)
            text = clean_edges(vote["text"])
            out[name] = {"value": text, "raw_text": text, "box_confidence": round(conf, 2), "bbox": [round(float(v), 1) for v in xyxy], "n_boxes": len(items), "readings": vote["readings"]}
            continue
        order = reading_order(boxes, angle, img.size) if len(boxes) > 1 else [0]
        text = clean_edges(" ".join(t for t in (ocr_box(img, boxes[i], psm=7, lang=lang, field=name) for i in order) if t))
        out[name] = {"value": text, "raw_text": text, "box_confidence": round(min(c for _, c in items), 2), "bbox": _union(boxes), "n_boxes": len(items),
                     "boxes": [[round(float(v), 1) for v in boxes[i]] for i in order]}              # one box per printed line, in reading order
    return out


def _readable(name: str, raw: str) -> bool:
    """Is the text the detector read usable for this field? (an unreadable one is replaced by the text reading)"""
    if name == "amount":
        return parse_amount(raw) is not None
    if name == "amount_words":
        return bool(words_to_number(raw) or english_words_to_number(raw))
    if name in ("contract_date", "first_payment_date", "signing_date"):
        return parse_date_fr(raw) is not None
    if name == "duration":
        return parse_duration(raw) is not None
    if name == "interest_rate":
        return parse_rate(raw) is not None
    if name in ("lender_name", "borrower_name"):
        return len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", raw)) >= 2
    return bool(raw.strip())


def merge_readings(got: dict, text_fields: dict) -> dict:
    """Detector fields stay; a field the detector missed, or read as something unusable, comes from the text reading."""
    out = dict(got)
    for name, f in text_fields.items():
        have = out.get(name)
        if have is None or (name not in NO_TEXT and not _readable(name, have.get("raw_text", ""))):
            out[name] = f
    return out


def interpret(got: dict, not_applicable: set | frozenset = frozenset()) -> dict:
    """Parse the texts and run the checks -> {"values": {...}, "checks": {...}, "warnings": [...]}."""
    g = lambda k: (got.get(k) or {}).get("raw_text", "")
    v, checks, warnings = {}, {}, []
    amount = parse_amount(g("amount")) if "amount" in got else None
    words = (words_to_number(g("amount_words")) or english_words_to_number(g("amount_words"))) if "amount_words" in got else None
    v["amount"] = f"{amount:.2f}" if amount is not None else g("amount")
    v["currency"] = currency_of(g("amount")) or (words[2] if words else "")
    v["amount_words"] = g("amount_words")
    if amount is not None and words is not None:
        wv = words[0] + words[1] / 100
        checks["amount_match"] = abs(wv - amount) < 0.005
        if not checks["amount_match"]:
            warnings.append({"code": "amount_mismatch", "message": f"The amount in figures ({amount:,.2f}) and in words ({wv:,.2f}) are different."})
    else:
        checks["amount_match"] = None
        if "amount" in got and "amount_words" in got:
            warnings.append({"code": "amount_not_checked", "message": "The amount in figures and in words could not both be read as numbers, so they were not compared."})
    v["amount_check"] = {True: "ok", False: "mismatch", None: "not checked"}[checks["amount_match"]]

    rate, months = (parse_rate(g("interest_rate")) if "interest_rate" in got else None), (parse_duration(g("duration")) if "duration" in got else None)
    pay = parse_amount(g("monthly_payment")) if "monthly_payment" in got else None
    v["rate"] = f"{rate:.2f}" if rate is not None else g("interest_rate")
    v["duration_months"] = str(months) if months is not None else g("duration")
    v["monthly_payment"] = f"{pay:.2f}" if pay is not None else g("monthly_payment")
    if None not in (amount, rate, months, pay) and months:
        expected = annuity(amount, rate, months)
        checks["payment_match"] = abs(expected - pay) <= max(0.02, 0.01 * expected)
        if not checks["payment_match"]:
            warnings.append({"code": "payment_mismatch", "message": f"The monthly payment ({pay:,.2f}) differs from the loan formula for {amount:,.0f} at {rate} % over {months} months ({expected:,.2f})."})
    else:
        checks["payment_match"] = None
    v["payment_check"] = {True: "ok", False: "differs", None: "not checked"}[checks["payment_match"]]

    dates = {}
    for k in ("contract_date", "first_payment_date", "signing_date"):
        if k in got:
            dates[k] = parse_date_fr(g(k))
            v[k] = dates[k].isoformat() if dates[k] else g(k)
            if dates[k] is None:
                warnings.append({"code": "date_invalid", "message": f"The {k.replace('_', ' ')} '{g(k)}' is not a valid date."})
        else:
            v[k] = ""
    checks["dates_valid"] = all(dates.values()) if dates else None
    if dates.get("contract_date") and dates.get("first_payment_date") and dates["first_payment_date"] <= dates["contract_date"]:
        warnings.append({"code": "first_payment_early", "message": "The first payment is not after the contract date."})
        checks["first_payment_after_contract"] = False
    elif dates.get("contract_date") and dates.get("first_payment_date"):
        checks["first_payment_after_contract"] = True
    if dates.get("contract_date") and dates.get("signing_date") and dates["signing_date"] < dates["contract_date"]:
        warnings.append({"code": "signed_before_contract", "message": "The signing date is before the contract date."})

    iban = clean_iban(g("iban"))
    v["iban"] = iban
    if "iban" in got:
        want = IBAN_LENGTH.get(iban[:2])
        checks["iban_length_ok"] = (len(iban) == want) if want else None
        if checks["iban_length_ok"] is False:
            warnings.append({"code": "iban_length", "message": f"The IBAN has {len(iban)} characters ({iban[:2]} IBANs have {want}): check it."})

    v["title"], v["type"] = g("title"), contract_type(g("title"))
    for k in ("contract_number", "lender_name", "borrower_name", "borrower_id", "borrower_address", "place"):
        v[k] = g(k)
    checks["signature_lender"], checks["signature_borrower"], checks["stamp"] = ("signature_lender" in got), ("signature_borrower" in got), ("stamp" in got)
    for who in ("lender", "borrower"):
        if not checks[f"signature_{who}"]:
            warnings.append({"code": f"no_signature_{who}", "message": f"No signature of the {who} was detected."})
    missing = [k for k in ESSENTIAL if k not in got and k not in not_applicable]       # e.g. no monthly payment / IBAN in a single-repayment loan
    if missing:
        warnings.append({"code": "missing_field", "message": "Not found on the picture: " + ", ".join(missing) + ". Is this a loan contract, and is the picture sharp?"})
    return {"values": v, "checks": checks, "warnings": warnings}


def extract_contract_bytes(data: bytes, filename: str = "contract", conf: float = 0.25, lang: str | None = None, model=None) -> dict:
    """Picture / PDF bytes -> the same kind of dict as the other extractors (fields, row, columns, warnings, doc_type) plus `checks`."""
    img, n_pages = _open_receipt_image(data, filename)
    orig_w, orig_h = img.width, img.height                                        # the ORIGINAL picture: the screen draws the boxes on it
    canon = None
    from ..ocr import reconstruct
    if reconstruct.pages_active():
        from ..preprocessing import canonicalize_pil
        img, canon = canonicalize_pil(img, "contract")                                   # Stage A: the page cut out, upright, evenly lit
    dets, notes = [], []
    try:
        model = model or load_model()
        r = model.predict(img, imgsz=int(model.overrides.get("imgsz", 1280)), conf=conf, verbose=False)[0]
        dets = [(model.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes]
    except SystemExit:                                            # no trained detector / no ultralytics: the text reader below still works
        notes.append({"code": "detector_missing", "message": "The contract detector is not installed: the contract was read from its text only."})
    got = fields_from_boxes(img, dets, None, lang)
    try:
        text = read_contract_text(img, lang)                      # English or French sentences + signatures: fills what the detector missed
    except Exception as exc:                                      # never lose the detector result because the second reading failed ... but say it
        import logging
        logging.getLogger("documind").exception("contract text reading failed")
        text = {"fields": {}, "not_applicable": set()}
        notes.append({"code": "text_reading_failed", "message": f"The second reading of the contract (from its text) failed: {type(exc).__name__}: {str(exc)[:200]}"})
    got = merge_readings(got, text["fields"])
    res = interpret(got, text["not_applicable"])
    res["warnings"] = notes + res["warnings"]
    v = res["values"]
    shown = {"amount": v["amount"] or None, "monthly_payment": v["monthly_payment"] or None, "interest_rate": v["rate"] or None}
    fields = [{"label": FIELD_LABELS[k], "text": (shown.get(k) or x["value"] or x["raw_text"]), "confidence": x["box_confidence"], "bbox": x["bbox"], "page": 1, "source": x.get("source", "detector"),
               **({"boxes": x["boxes"]} if len(x.get("boxes", [])) > 1 else {})}                     # a value wrapped over several lines: draw each line, not one big rectangle
              for k, x in got.items() if x["value"] or x["raw_text"]]
    fields.sort(key=lambda f: (f["bbox"][1], f["bbox"][0]))
    if canon is not None:
        canon.map_back(fields)
    warnings = list(res["warnings"])
    if n_pages > 1:
        warnings.append({"code": "first_page_only", "message": f"The PDF has {n_pages} pages: only the first page was read."})
    c = res["checks"]
    row = {"file": filename, "CONTRACT_TYPE": v["type"], "TITLE": v["title"], "CONTRACT_NO": v["contract_number"], "CONTRACT_DATE": v["contract_date"], "LENDER": v["lender_name"],
           "BORROWER": v["borrower_name"], "BORROWER_ID": v["borrower_id"], "BORROWER_ADDRESS": v["borrower_address"], "AMOUNT": v["amount"], "CURRENCY": v["currency"],
           "AMOUNT_WORDS": v["amount_words"], "RATE": v["rate"], "DURATION_MONTHS": v["duration_months"], "MONTHLY_PAYMENT": v["monthly_payment"],
           "FIRST_PAYMENT_DATE": v["first_payment_date"], "IBAN": v["iban"], "PLACE": v["place"], "SIGNING_DATE": v["signing_date"],
           "SIGNED_LENDER": "yes" if c["signature_lender"] else "no", "SIGNED_BORROWER": "yes" if c["signature_borrower"] else "no", "STAMP": "yes" if c["stamp"] else "no",
           "AMOUNT_CHECK": v["amount_check"], "PAYMENT_CHECK": v["payment_check"]}
    return {"file": filename, "doc_type": "contract", "warnings": warnings, "n_pages": n_pages,
            "pages": [{"page": 1, "width": orig_w, "height": orig_h, "size": [orig_w, orig_h], **({"canonical_steps": canon.steps} if canon is not None else {}), "reconstructed_words": text.get("reconstructed_words", 0)}],
            "fields": fields, "items": [], "items_check": {}, "checks": c, "row": row, "columns": CONTRACT_COLUMNS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lang", default=None, help="Tesseract language(s), e.g. fra+eng")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = extract_contract_bytes(Path(a.picture).read_bytes(), Path(a.picture).name, a.conf, a.lang)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    for f in res["fields"]:
        print(f"{f['label']:20s} {f['text']}   ({f['confidence']:.2f})")
    print("\nchecks:", res["checks"])
    for w in res["warnings"]:
        print("WARNING:", w["message"])
    print("\n" + ",".join(CONTRACT_COLUMNS) + "\n" + ",".join(f'"{res["row"].get(c, "")}"' for c in CONTRACT_COLUMNS))


if __name__ == "__main__":
    main()
