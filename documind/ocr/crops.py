"""Text-line crops with their transcription, taken from the three datasets: the data a recognition model (OCR) is trained on.

  receipts  (generate_receipts.py)  every annotated box + the text of ground_truth.jsonl (store, address, phone, date, time, number, item name / qty / price,
            subtotal, tax, total, payment)
  cheques   (generate_cheques.py)   bank, cheque number, amount in words and figures, payee, place, date, account number, MICR line (signature has no text)
  contracts (generate_contracts.py) the key terms of a loan contract (amount, rate, duration, IBAN, names, dates ...); a value that wraps over two lines
            is cut between its boxes by comparing the width of the boxes with the length of the words (left out when that is not clear)
  fatura    (FATURA invoices)       every annotated field that has a one-line transcription

The boxes are those of the detectors (tight on the value), so the crops look like what the detector hands to the OCR at reading time.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

SPLITS = ("train", "val", "test")
MAX_CHARS = 100


@dataclass
class Crop:
    image: Path
    box: tuple[float, float, float, float]      # x0, y0, x1, y1 in pixels of `image`
    text: str
    source: str                                 # receipts / cheques / fatura
    kind: str                                   # the class of the box (store_name, amount_words, TOTAL ...)
    split: str                                  # train / val / test


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t or "")).strip()


def usable(text: str, box) -> bool:
    x0, y0, x1, y1 = box
    return bool(text) and len(text) <= MAX_CHARS and (x1 - x0) >= 6 and (y1 - y0) >= 6 and "\n" not in text


# ------------------------------------------------------------------------------------------------ generated datasets (COCO annotations + ground_truth.jsonl)
def _load_generated(root: Path, split: str):
    ann = root / "annotations" / f"{split}.json"
    if not ann.is_file():
        return None
    coco = json.loads(ann.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:                       # annotation order = the order the generator drew them
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append((cats[a["category_id"]], (x, y, x + w, y + h)))
    return {i["id"]: i for i in coco["images"]}, boxes


def _ground_truth(root: Path) -> dict[str, dict]:
    gt = {}
    p = root / "ground_truth.jsonl"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                gt[Path(r["file_name"]).name] = r["ground_truth"]
    return gt


def receipt_texts(truth: dict, boxes: list[tuple[str, tuple]]) -> list[tuple[str, tuple, str]]:
    """[(class, box, text)] for one receipt. Item boxes come in drawing order: k-th item_name / item_qty / item_price = k-th item.
    The address is drawn on two lines (street, then postcode + city)."""
    items, k = truth.get("items", []), {"item_name": 0, "item_qty": 0, "item_price": 0}
    addr = [p.strip() for p in truth.get("address", "").split(",", 1)]
    n_addr, out = 0, []
    for cls, box in boxes:
        text = None
        if cls == "address":
            text = addr[n_addr] if n_addr < len(addr) else None
            n_addr += 1
        elif cls in k:
            i = k[cls]
            k[cls] += 1
            if i < len(items):
                text = {"item_name": items[i]["name"], "item_qty": str(items[i]["qty"]), "item_price": items[i]["price"]}[cls]
        elif cls in truth:
            text = truth[cls]
        if text is not None:
            out.append((cls, box, str(text)))
    return out


def cheque_texts(truth: dict, boxes: list[tuple[str, tuple]]) -> list[tuple[str, tuple, str]]:
    """The cheque number is printed "N° 1234567" (the truth file keeps the digits only) and the account number in groups of 4 digits."""
    out = []
    for cls, box in boxes:
        if cls == "signature" or not isinstance(truth.get(cls), str):
            continue
        text = truth[cls]
        if cls == "cheque_number":
            text = f"N° {text}"
        elif cls == "account_number":
            text = " ".join(text[i:i + 4] for i in range(0, len(text), 4))
        out.append((cls, box, text))
    return out


CONTRACT_NO_TEXT = ("signature_lender", "signature_borrower", "stamp")


class PageContext:
    """What the mappers may need to know about the picture: its size, and (only when asked) its tilt."""

    def __init__(self, path: Path, size: tuple[int, int]):
        self.path, self.size, self._angle = path, size, None

    def angle(self) -> float:
        if self._angle is None:
            import cv2
            from .lines import skew_angle
            gray = cv2.imdecode(__import__("numpy").fromfile(str(self.path), dtype="uint8"), cv2.IMREAD_GRAYSCALE)
            self._angle = skew_angle(gray) if gray is not None else 0.0
        return self._angle


def contract_texts(truth: dict, boxes: list[tuple[str, tuple]], ctx: PageContext | None = None) -> list[tuple[str, tuple, str]]:
    """[(class, box, text)] for one contract page. One box per class is the normal case. The lender's name is printed twice (letterhead + first paragraph);
    a value that wrapped over 2-3 lines has one box per line and is cut by the width of its boxes (None of them when the cut is not clear)."""
    from .lines import reading_order, split_value
    groups: dict[str, list[tuple]] = {}
    for cls, box in boxes:
        if cls not in CONTRACT_NO_TEXT:
            groups.setdefault(cls, []).append(box)
    out = []
    for cls, bxs in groups.items():
        text = truth.get(cls)
        if not isinstance(text, str) or not text.strip():
            continue
        if cls == "lender_name":
            if len(bxs) == 2:
                out += [(cls, b, text) for b in bxs]
        elif len(bxs) == 1:
            out.append((cls, bxs[0], text))
        elif 2 <= len(bxs) <= 3 and ctx is not None:
            order = reading_order(bxs, ctx.angle(), ctx.size)
            parts = split_value(text, [bxs[i][2] - bxs[i][0] for i in order])
            if parts:
                out += [(cls, bxs[i], part) for i, part in zip(order, parts)]
    return out


def generated_crops(root: Path, source: str, store_is_upper=None) -> Iterator[Crop]:
    """Crops of a dataset made by generate_receipts.py / generate_cheques.py (root = the folder with images/, annotations/, ground_truth.jsonl).
    `store_is_upper(crop_image) -> bool` tells whether a drawn store name is in capitals (the generator writes the original name in the truth file
    but draws it in capitals half of the time)."""
    root = Path(root)
    gt = _ground_truth(root)
    mapper = {"receipts": receipt_texts, "cheques": cheque_texts, "contracts": contract_texts}[source]
    for split in SPLITS:
        loaded = _load_generated(root, split)
        if loaded is None:
            continue
        images, boxes = loaded
        for iid, info in sorted(images.items()):
            truth = gt.get(info["file_name"])
            path = root / "images" / split / info["file_name"]
            if truth is None or not path.is_file():
                continue
            page = boxes.get(iid, [])
            mapped = mapper(truth, page, PageContext(path, (info["width"], info["height"]))) if source == "contracts" else mapper(truth, page)
            for cls, box, text in mapped:
                text = clean_text(text)
                if cls == "store_name" and store_is_upper is not None:
                    text = text.upper() if store_is_upper(path, box) else text
                if usable(text, box):
                    yield Crop(path, box, text, source, cls, split)


# ------------------------------------------------------------------------------------------------ FATURA
def _split_of(stem: str, val: float = 0.1, test: float = 0.1) -> str:
    h = (sum(ord(c) * (i + 1) for i, c in enumerate(stem)) * 2654435761) % 1000 / 1000
    return "test" if h < test else "val" if h < test + val else "train"


def fatura_crops(root: Path, box_mode: str = "auto", include_colored: bool = True) -> Iterator[Crop]:
    """Crops of the FATURA invoices. Uses the project's own loader (documind.data.loader): every annotation box that carries a one-line text.
    box_mode: auto (the loader's format detection) or one of xyxy / xywh / xyxy_n1 / xywh_n1 / xyxy_n1000 / xywh_n1000."""
    from ..data.loader import find_samples, read_annotation, to_pixels
    for s in find_samples(root, include_colored=include_colored):
        with Image.open(s["image"]) as im:
            W, H = im.size
        ann = None
        for p in s["annotations"]:                       # first annotation file that has transcriptions
            a = read_annotation(p)
            if any(b.text for b in a.boxes):
                ann = a
                break
        if ann is None:
            continue
        top = max((max(b.vals) for b in ann.boxes), default=0.0)
        split = _split_of(s["stem"].lower().replace("_color", "").replace("-color", ""))
        for b in ann.boxes:
            text = clean_text(b.text if isinstance(b.text, str) else "")
            if box_mode == "auto":
                mode = "xywh" if b.hint == "xywh" else "xyxy"
                mode += "_n1" if top <= 1.5 else "_n1000" if (top <= 1000 and max(W, H) > 1100 and b.hint is None) else ""
            else:
                mode = box_mode
            x0, y0, x1, y1 = (float(v) for v in to_pixels([b.vals], mode, W, H)[0])
            box = (max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1))
            if usable(text, box) and (box[3] - box[1]) < 0.06 * H and (box[2] - box[0]) > 1.2 * (box[3] - box[1]):    # one text line, not a block
                yield Crop(s["image"], box, text, "fatura", normalize_label_name(b.label), split)


def normalize_label_name(label) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label or "text")).strip("_") or "text"
