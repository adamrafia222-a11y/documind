"""Invoice fields with a YOLO detector (like the receipts, cheques and contracts): one box on the VALUE of every field, then each box is read.

    python -m documind.inference.invoice_yolo invoice.jpg                  # every field found, with its box and reading

Model: models/invoices_yolo.pt (train it: python -m documind.data.prepare_invoices --fatura data/raw/FATURA, then python -m documind.training.train_invoices).
The extractor (documind.inference.extractor.FieldExtractor) uses it as soon as the file exists (INVOICE_MODEL_PATH to change it). The line items come from the rules reader.

Reading a box: the same engines as the other types (documind.inference.receipts.ocr_box: Tesseract + the OCR model + the handwriting model, the better
reading is kept). A field that is a text BLOCK (address, note, bill to ...) is read as a block with Tesseract; a field that wraps over several boxes is joined
top to bottom. A field that is printed once keeps its most confident box.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import MODELS_DIR, OCR_LANG

MODEL_PATH = Path(os.getenv("INVOICE_MODEL_PATH", MODELS_DIR / "invoices_yolo.pt"))
CONF = float(os.getenv("INVOICE_YOLO_CONF", "0.25"))
BLOCKS = ("ADDRESS", "NOTE", "BUYER", "SEND_TO", "WORDS", "TERMS", "DESCRIPTION")     # text blocks: read whole, joined when split over several boxes
_model = None


def kind_of(label: str) -> str | None:
    """The kind of value of a field for the format check of the reading (documind.ocr.ensemble.KIND), or None for free text."""
    L = label.upper()
    if L.endswith("DATE"):
        return "date"
    if L in ("TOTAL", "AMOUNT_DUE", "BALANCE_DUE"):
        return "total"
    if L in ("SUB_TOTAL", "SUBTOTAL"):
        return "subtotal"
    if L in ("TAX", "GST", "VAT", "DISCOUNT", "SHIPPING"):
        return "tax"
    if "PHONE" in L or L.endswith("TEL"):
        return "phone"
    return None


def is_block(label: str) -> bool:
    return any(k in label.upper() for k in BLOCKS)


def model_available(path: Path | None = None) -> bool:
    return Path(path or MODEL_PATH).exists()


def load_model(path: Path | None = None):
    global _model
    path = Path(path or MODEL_PATH)
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("Ultralytics is not installed. Run:  pip install -r requirements.txt")
        if not path.exists():
            raise SystemExit(f"{path} not found. Train it first:  python -m documind.data.prepare_invoices --fatura data/raw/FATURA  then  python -m documind.training.train_invoices")
        _model = YOLO(str(path))
    return _model


def detect(model, img_bgr: np.ndarray, conf: float = CONF) -> list[tuple[str, list[float], float]]:
    """[(class_name, [x0, y0, x1, y1], confidence)] on a BGR picture (the coordinates are those of the picture)."""
    r = model.predict(img_bgr, imgsz=int(model.overrides.get("imgsz", 1280)), conf=conf, verbose=False)[0]
    return [(model.names[int(b.cls)], [float(v) for v in b.xyxy[0].tolist()], float(b.conf)) for b in r.boxes]


def _read(img: Image.Image, xyxy, label: str, lang: str | None) -> str:
    from .receipts import _tesseract_box, ocr_box
    if is_block(label):
        return _tesseract_box(img, xyxy, 6, 2, 2.0, lang, None)
    return ocr_box(img, xyxy, psm=7, lang=lang, field=kind_of(label))


def fields_from_dets(img: Image.Image, dets: list[tuple], page: int = 1, lang: str | None = None) -> list[dict]:
    """Detections -> fields {label, text, confidence, bbox, page} (the shape the invoice pipeline and the screen already use), top to bottom."""
    by: dict[str, list[tuple]] = {}
    for name, xyxy, conf in dets:
        by.setdefault(name.upper(), []).append((xyxy, conf))
    out = []
    for label, boxes in by.items():
        if is_block(label):
            boxes = sorted(boxes, key=lambda t: (round(t[0][1] / 10), t[0][0]))              # top to bottom, then left to right
        else:
            boxes = [max(boxes, key=lambda t: t[1])]                                          # printed once: the most confident box
        texts = [(_read(img, b, label, lang), b, c) for b, c in boxes]
        texts = [t for t in texts if t[0]]
        if not texts:
            continue
        xs0, ys0 = min(b[0] for _, b, _ in texts), min(b[1] for _, b, _ in texts)
        xs1, ys1 = max(b[2] for _, b, _ in texts), max(b[3] for _, b, _ in texts)
        out.append({"label": label, "text": " ".join(t for t, _, _ in texts), "confidence": round(float(np.mean([c for _, _, c in texts])), 3),
                    "bbox": [round(xs0, 1), round(ys0, 1), round(xs1, 1), round(ys1, 1)], "page": page})
    return sorted(out, key=lambda f: (f["bbox"][1], f["bbox"][0]))


def main() -> None:
    import argparse
    from ..preprocessing import read_image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--lang", default=OCR_LANG)
    a = ap.parse_args()
    img = read_image(a.picture)
    pil = Image.fromarray(img[:, :, ::-1].copy())
    for f in fields_from_dets(pil, detect(load_model(), img, a.conf), 1, a.lang):
        print(f"{f['label']:<18} {f['confidence']:.2f}  {f['text']}")


if __name__ == "__main__":
    main()
