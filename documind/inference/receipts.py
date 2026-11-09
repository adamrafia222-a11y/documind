"""Read a receipt picture: the YOLO detector finds the store, address, phone, date, time, receipt number, item rows (name / quantity / price),
subtotal, tax, total and payment method; Tesseract reads each box. (A model trained on the older 5 classes store / date / total / receipt_no /
item still works: the class names are mapped by ALIASES.)

    python -m documind.inference.receipts ticket.jpg                       # prints JSON
    python -m documind.inference.receipts a.jpg b.jpg --lang eng+ara --conf 0.3

In Python:
    from documind.inference.receipts import read_receipt
    read_receipt("ticket.jpg")      # {"date": {"value": "30/09/2022", ...}, "total": {...}, "items": [...]}

Uses models/receipts_yolo.pt (STEP 2) and the Tesseract settings of .env (TESSERACT_CMD). The OCR language is RECEIPT_OCR_LANG
(.env), else OCR_LANG: for Arabic receipts install the Tesseract data `ara` and set RECEIPT_OCR_LANG=eng+ara.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
from pathlib import Path

from PIL import Image, ImageOps

from ..config import MODELS_DIR, OCR_LANG, PDF_DPI, TESSERACT_CMD

MODEL_PATH = Path(os.getenv("RECEIPT_MODEL_PATH", MODELS_DIR / "receipts_yolo.pt"))
LANG = os.getenv("RECEIPT_OCR_LANG", OCR_LANG)
DATE_RE = re.compile(r"\d{1,2}\s?[-/.]\s?\d{1,2}\s?[-/.]\s?\d{2,4}|\d{4}\s?[-/.]\s?\d{1,2}\s?[-/.]\s?\d{1,2}")
AMOUNT_RE = re.compile(r"\d[\d ]*[.,]\d{1,2}")
ID_RE = re.compile(r"\d{4,}")
TIME_RE = re.compile(r"\d{1,2}\s?[:hH]\s?\d{2}(?:\s?:\s?\d{2})?")
PHONE_RE = re.compile(r"\+?\d[\d .\-]{6,}\d")
QTY_RE = re.compile(r"\d+")
_model = None

# class name in the model -> the name used here (generate_receipts.py names on the left, the older 5-class names too)
ALIASES = {"store_name": "store", "store": "store", "address": "address", "phone": "phone", "date": "date", "time": "time", "receipt_number": "receipt_no",
           "receipt_no": "receipt_no", "subtotal": "subtotal", "tax": "tax", "total": "total", "payment_method": "payment_method",
           "item_name": "item_name", "item_qty": "item_qty", "item_price": "item_price", "item": "item"}
SINGLE_FIELDS = ("store", "phone", "date", "time", "receipt_no", "subtotal", "tax", "total", "payment_method")      # one box per receipt: the most confident



WHITELIST = {"date": "0123456789/-. ", "time": "0123456789:hH ", "receipt_no": "0123456789", "phone": "0123456789 +.-", "item_qty": "0123456789",
             "total": "0123456789,. ", "subtotal": "0123456789,. ", "tax": "0123456789,. ", "item_price": "0123456789,. "}   # characters each field can contain


def ocr_box(img: Image.Image, xyxy, psm: int = 7, pad: int = 2, scale: float = 3.0, lang: str | None = None, whitelist: str | None = None) -> str:
    """Tesseract on one detected box: padded, grey, enlarged, with a white margin (Tesseract misses characters that touch the edge of the crop).
    `whitelist` limits the characters the box can contain (digits only for a quantity, ...). Small pad (2 px): the boxes of neighbouring values
    (quantity, " x ", name) are only a few pixels apart, a wider crop reads them too (tested: item quantity + price right 90 % with pad 2, 63 % with pad 6)."""
    import pytesseract
    from PIL import ImageOps
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    x0, y0, x1, y1 = xyxy
    crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))).convert("L")
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    crop = ImageOps.expand(ImageOps.autocontrast(crop, cutoff=1), border=16, fill=255)
    cfg = f"--psm {psm}" + (f" -c tessedit_char_whitelist={whitelist}" if whitelist else "")
    return " ".join(pytesseract.image_to_string(crop, lang=lang or LANG, config=cfg).split())


def _union(boxes) -> list[float]:
    return [round(float(v), 1) for v in (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))]


def _parse(name: str, txt: str) -> str | None:
    if name == "date":
        m = DATE_RE.search(txt)
        return m.group(0).replace(" ", "") if m else None
    if name == "time":
        m = TIME_RE.search(txt)
        return m.group(0).replace(" ", "").replace("H", "h") if m else None
    if name in ("total", "subtotal", "tax"):
        m = AMOUNT_RE.findall(txt)
        return m[-1].replace(" ", "") if m else None                              # the amount is printed last on the line
    if name == "receipt_no":
        m = ID_RE.findall(txt)
        return m[0] if m else None
    if name == "phone":
        m = PHONE_RE.search(txt)
        return " ".join(m.group(0).split()) if m else None
    return txt or None


def _cluster_rows(parts: list[tuple], theta: float, gap: float) -> list[list[tuple]]:
    """Group boxes into printed lines after removing a tilt of `theta` radians (y' = y cos - x sin): a gap above `gap` pixels starts a new line."""
    c, sn = math.cos(theta), math.sin(theta)
    keyed = sorted((((b[1] + b[3]) / 2) * c - ((b[0] + b[2]) / 2) * sn, kind, b, conf) for kind, b, conf in parts)
    rows: list[list[tuple]] = []
    last = None
    for y, kind, b, conf in keyed:
        if rows and last is not None and y - last <= gap:
            rows[-1].append((kind, b, conf))
        else:
            rows.append([(kind, b, conf)])
        last = y
    return rows


def group_item_rows(parts: list[tuple[str, list[float], float]]) -> list[dict[str, tuple]]:
    """parts: [(kind in item_name / item_qty / item_price, xyxy, conf)] -> rows (top to bottom), each {kind: (xyxy, conf)}.
    A photographed receipt is tilted, so the price on the right sits higher or lower than the name on the left: the tilt (-12 .. +12 degrees) that
    gives the FEWEST lines is used (the right angle puts name, quantity and price of one item on one line), then boxes less than 0.6 of a typical
    box height apart share a line."""
    if not parts:
        return []
    hs = sorted(b[3] - b[1] for _, b, _ in parts)
    gap = 0.6 * hs[len(hs) // 2]
    def cost(a: int) -> tuple:
        rows = _cluster_rows(parts, math.radians(a / 2), gap)
        dup = sum(len(r) - len({k for k, _, _ in r}) for r in rows)          # two names / two prices on one line = lines were merged: wrong angle
        return (len(rows) + 5 * dup, abs(a), a)
    best = min(cost(a) for a in range(-24, 25))
    rows = []
    for row in _cluster_rows(parts, math.radians(best[2] / 2), gap):
        d: dict[str, tuple] = {}
        for kind, b, c in row:
            if kind not in d or c > d[kind][1]:
                d[kind] = (b, c)
        rows.append(d)
    return rows


def fields_from_boxes(img: Image.Image, dets: list[tuple], lang: str | None = None) -> dict:
    """dets: [(class_name, [x0, y0, x1, y1], confidence)] -> {field: {"value", "raw_text", "box_confidence", "bbox"}} and "items" (top to bottom).
    Item rows: from item_name / item_qty / item_price boxes grouped by line (generate_receipts.py classes), or one `item` box per row (older labelling)."""
    best, legacy_items, parts, address = {}, [], [], []
    for cname, xyxy, conf in dets:
        name = ALIASES.get(cname, cname)
        if name == "item":
            legacy_items.append((xyxy[1], xyxy, conf))
        elif name in ("item_name", "item_qty", "item_price"):
            parts.append((name, xyxy, conf))
        elif name == "address":
            address.append((xyxy, conf))                                             # usually two lines: keep them all
        elif name not in best or conf > best[name][2]:
            best[name] = (name, xyxy, conf)
    out: dict = {}
    for name, xyxy, conf in best.values():
        txt = ocr_box(img, xyxy, psm=6 if name == "store" else 7, lang=lang, whitelist=WHITELIST.get(name))
        out[name] = {"value": _parse(name, txt), "raw_text": txt, "box_confidence": round(float(conf), 2), "bbox": [round(float(v), 1) for v in xyxy]}
    if address:
        lines = [(b, c, ocr_box(img, b, psm=7, lang=lang)) for b, c in sorted(address, key=lambda t: t[0][1])]
        txt = ", ".join(t for _, _, t in lines if t)
        out["address"] = {"value": txt or None, "raw_text": txt, "box_confidence": round(float(max(c for _, c, _ in lines)), 2), "bbox": _union([b for b, _, _ in lines])}

    items = []
    for _, b, c in sorted(legacy_items, key=lambda t: t[0]):
        raw = ocr_box(img, b, psm=7, lang=lang)
        m = AMOUNT_RE.findall(raw)
        items.append({"raw_text": raw, "description": raw, "qty": "", "amount": m[-1].replace(" ", "") if m else "",
                      "box_confidence": round(float(c), 2), "bbox": [round(float(v), 1) for v in b]})
    for row in group_item_rows(parts):
        desc = ocr_box(img, row["item_name"][0], psm=7, lang=lang) if "item_name" in row else ""
        qty_raw = ocr_box(img, row["item_qty"][0], psm=7, scale=2.0, lang=lang, whitelist=WHITELIST.get("item_qty")) if "item_qty" in row else ""
        price_raw = ocr_box(img, row["item_price"][0], psm=7, scale=2.0, lang=lang, whitelist=WHITELIST.get("item_price")) if "item_price" in row else ""
        q, a = QTY_RE.search(qty_raw), AMOUNT_RE.findall(price_raw)
        qty, amount = (q.group(0) if q else ""), (a[-1].replace(" ", "") if a else price_raw.replace(" ", ""))
        items.append({"raw_text": " ".join(x for x in (qty_raw, desc, price_raw) if x), "description": desc, "qty": qty, "amount": amount,
                      "box_confidence": round(float(min(c for _, c in row.values())), 2), "bbox": _union([b for b, _ in row.values()])})
    out["items"] = sorted(items, key=lambda it: it["bbox"][1])
    return out


def load_model(path: Path = MODEL_PATH):
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("Ultralytics is not installed. Run:  pip install -r requirements-receipts.txt")
        if not Path(path).exists():
            raise SystemExit(f"{path} not found. Train it first:  python -m documind.training.train_receipts")
        _model = YOLO(str(path))
    return _model


def read_receipt(path, conf: float = 0.25, lang: str | None = None, model=None) -> dict:
    model = model or load_model()
    img = Image.open(path).convert("RGB")
    r = model.predict(str(path), imgsz=int(model.overrides.get("imgsz", 960)), conf=conf, verbose=False)[0]
    dets = [(model.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes]
    return fields_from_boxes(img, dets, lang)


# ------------------------------------------------------------------------------------------------ server side (same result shape as invoices)
FIELD_LABELS = {"store": "STORE", "address": "ADDRESS", "phone": "PHONE", "date": "DATE", "time": "TIME", "receipt_no": "RECEIPT_NO",
                "subtotal": "SUBTOTAL", "tax": "TAX", "total": "TOTAL", "payment_method": "PAYMENT"}
RECEIPT_COLUMNS = ["file", "STORE", "ADDRESS", "PHONE", "DATE", "TIME", "RECEIPT_NO", "SUBTOTAL", "TAX", "TOTAL", "PAYMENT", "N_ITEMS", "ITEMS"]   # receipts.csv (never mixed with invoices)


def model_available(path: Path = MODEL_PATH) -> bool:
    return Path(path).exists()


def _open_receipt_image(data: bytes, filename: str) -> tuple[Image.Image, int]:
    """Bytes of a picture or PDF -> (RGB picture with the camera rotation applied, number of pages)."""
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        from ..preprocessing import pdf_to_images
        pages = pdf_to_images(data, PDF_DPI)
        if not pages:
            raise ValueError("Empty PDF.")
        return Image.fromarray(pages[0][:, :, ::-1]), len(pages)                      # BGR -> RGB, first page only
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"), 1
    except Exception:
        raise ValueError("Unreadable image (unsupported or corrupted file).")


def extract_receipt_bytes(data: bytes, filename: str = "receipt", conf: float = 0.25, lang: str | None = None, model=None) -> dict:
    """Picture / PDF bytes -> the same kind of dict as FieldExtractor.extract_bytes (fields, items, row), plus doc_type and columns."""
    img, n_pages = _open_receipt_image(data, filename)
    model = model or load_model()
    r = model.predict(img, imgsz=int(model.overrides.get("imgsz", 960)), conf=conf, verbose=False)[0]
    dets = [(model.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes]
    got = fields_from_boxes(img, dets, lang)
    fields = [{"label": FIELD_LABELS.get(k, k.upper()), "text": v["value"] or v["raw_text"], "confidence": v["box_confidence"], "bbox": v["bbox"], "page": 1,
               "source": "detector"} for k, v in got.items() if k != "items" and (v["value"] or v["raw_text"])]
    fields.sort(key=lambda f: (f["bbox"][1], f["bbox"][0]))
    items = [{"description": it["description"], "qty": it["qty"], "amount": it["amount"], "confidence": it["box_confidence"], "bbox": it["bbox"], "page": 1}
             for it in got["items"] if it["description"] or it["amount"]]
    warnings = []
    if not fields:
        warnings.append({"code": "nothing_found", "message": "No store, date or total was detected. Is this really a receipt? Try the invoice type, or a sharper photo."})
    if n_pages > 1:
        warnings.append({"code": "first_page_only", "message": f"The PDF has {n_pages} pages: only the first page was read as a receipt."})
    row = {"file": filename}
    for f in fields:
        row[f["label"]] = f["text"] if f["label"] not in row else f"{row[f['label']]} | {f['text']}"
    row["N_ITEMS"] = str(len(items))
    row["ITEMS"] = " | ".join((f"{it['qty']} x " if it["qty"] else "") + it["description"] + (f" = {it['amount']}" if it["amount"] else "") for it in items)
    return {"file": filename, "doc_type": "receipt", "warnings": warnings, "n_pages": n_pages,
            "pages": [{"page": 1, "width": img.width, "height": img.height, "size": [img.width, img.height]}],
            "fields": fields, "items": items, "items_check": {}, "row": row, "columns": RECEIPT_COLUMNS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pictures", nargs="+")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    ap.add_argument("--lang", default=None, help="Tesseract language(s), e.g. eng+ara")
    a = ap.parse_args()
    for p in a.pictures:
        print(p)
        print(json.dumps(read_receipt(p, a.conf, a.lang), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
