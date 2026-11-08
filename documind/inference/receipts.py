"""Read a receipt picture: the YOLO detector finds the store, date, total, receipt number and item rows, Tesseract reads each box.

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
import json
import os
import re
from pathlib import Path

from PIL import Image

from ..config import MODELS_DIR, OCR_LANG, TESSERACT_CMD

MODEL_PATH = Path(os.getenv("RECEIPT_MODEL_PATH", MODELS_DIR / "receipts_yolo.pt"))
LANG = os.getenv("RECEIPT_OCR_LANG", OCR_LANG)
DATE_RE = re.compile(r"\d{1,2}\s?[-/.]\s?\d{1,2}\s?[-/.]\s?\d{2,4}|\d{4}\s?[-/.]\s?\d{1,2}\s?[-/.]\s?\d{1,2}")
AMOUNT_RE = re.compile(r"\d[\d ]*[.,]\d{1,2}")
ID_RE = re.compile(r"\d{4,}")
_model = None


def ocr_box(img: Image.Image, xyxy, psm: int = 7, pad: int = 4, scale: float = 2.0, lang: str | None = None) -> str:
    """Tesseract on one detected box (padded, grey, enlarged)."""
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    x0, y0, x1, y1 = xyxy
    crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))).convert("L")
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    return " ".join(pytesseract.image_to_string(crop, lang=lang or LANG, config=f"--psm {psm}").split())


def fields_from_boxes(img: Image.Image, dets: list[tuple], lang: str | None = None) -> dict:
    """dets: [(class_name, [x0, y0, x1, y1], confidence)] -> the value of each field and the item rows (top to bottom)."""
    best, items = {}, []
    for name, xyxy, conf in dets:
        if name == "item":
            items.append((xyxy[1], xyxy, conf))
        elif name not in best or conf > best[name][2]:
            best[name] = (name, xyxy, conf)                                          # one box per single-value field: the most confident
    out: dict = {}
    for name, xyxy, conf in best.values():
        txt = ocr_box(img, xyxy, psm=6 if name == "store" else 7, lang=lang)
        if name == "date":
            m = DATE_RE.search(txt)
            val = m.group(0).replace(" ", "") if m else None
        elif name == "total":
            m = AMOUNT_RE.findall(txt)
            val = m[-1].replace(" ", "") if m else None                              # the amount is printed last on the line
        elif name == "receipt_no":
            m = ID_RE.findall(txt)
            val = m[0] if m else None
        else:
            val = txt or None
        out[name] = {"value": val, "raw_text": txt, "box_confidence": round(float(conf), 2)}
    out["items"] = [{"raw_text": ocr_box(img, b, psm=7, lang=lang), "box_confidence": round(float(c), 2)} for _, b, c in sorted(items, key=lambda t: t[0])]
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
