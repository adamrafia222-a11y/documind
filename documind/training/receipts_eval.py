"""Evaluation helpers for the receipt detector that mAP does not give you.

mAP counts every predicted box that matches NO label as a mistake. When labels are incomplete (a date printed on the receipt but
not labelled, only some of the item rows labelled, two plausible totals but only one labelled) a model that does the right thing is
punished, and the score is capped well below 1. These helpers measure what you really want to know:

  found rate   store / date / total / receipt number ... (one box per receipt): is the MOST CONFIDENT predicted box of that class on a labelled box (IoU >= 0.5)?
               item rows, item name / qty / price, address: which share of the labelled boxes is found?
  suspects     high-confidence predictions that match no label (= probably a MISSING label) and labelled boxes the model
               does not see at all (= probably a WRONG label). See documind.training.audit_receipts.

All boxes are normalised (0..1): IoU does not change when the picture is scaled in x and y, so no picture size is needed.
"""
from __future__ import annotations

from pathlib import Path

SINGLE = ("store", "store_name", "phone", "date", "time", "receipt_no", "receipt_number", "subtotal", "tax", "total", "payment_method",
          "bank_name", "cheque_number", "amount_words", "payee", "amount_digits", "place", "signature", "account_number", "micr_line")   # one box per picture expected (receipts, cheques);
# address (2 lines), item, item_name, item_qty and item_price have many boxes: they are scored by the share of labelled boxes found
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_labels(path: Path) -> list[tuple[int, list[float]]]:
    """YOLO label file -> [(class, [x0, y0, x1, y1])] normalised."""
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) == 5:
            c, x, y, w, h = int(float(t[0])), *map(float, t[1:])
            out.append((c, [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_image(gt: list[tuple[int, list[float]]], pred: list[tuple[int, float, list[float]]], names: dict[int, str],
                iou_thr: float = 0.5, hi_conf: float = 0.6, min_conf: float = 0.25) -> dict[str, dict]:
    """gt: [(cls, xyxy)], pred: [(cls, conf, xyxy)]  ->  per class name:
         n_gt, found (single classes: True/False, None when the picture has no label of that class; item: number of labelled rows found),
         extra_hi  [(conf, xyxy, why)]  confident predictions that overlap no label ("no label of this class in the picture" / "label elsewhere"),
         missed_gt [xyxy]               labelled boxes with no prediction (conf >= min_conf) anywhere near."""
    res: dict[str, dict] = {}
    for c, name in names.items():
        g = [b for k, b in gt if k == c]
        p = sorted([(cf, b) for k, cf, b in pred if k == c], key=lambda t: -t[0])
        if name in SINGLE:
            found = None if not g else bool(p and any(iou(p[0][1], b) >= iou_thr for b in g))
        else:
            found = sum(any(cf >= min_conf and iou(pb, b) >= iou_thr for cf, pb in p) for b in g)
        extra = [(cf, b, "label elsewhere in the picture" if g else "no label of this class in the picture")
                 for cf, b in p if cf >= hi_conf and not any(iou(b, gb) >= 0.3 for gb in g)]
        missed = [b for b in g if not any(cf >= min_conf and iou(b, pb) >= 0.3 for cf, pb in p)]
        res[name] = {"n_gt": len(g), "found": found, "extra_hi": extra, "missed_gt": missed}
    return res


def summarize(per_image: list[dict[str, dict]]) -> dict[str, dict]:
    """found rate per class over many pictures."""
    out: dict[str, dict] = {}
    names = {n for r in per_image for n in r}
    for n in sorted(names):
        if n in SINGLE:
            vals = [r[n]["found"] for r in per_image if n in r and r[n]["found"] is not None]
            out[n] = {"pictures_with_label": len(vals), "found_rate": round(sum(vals) / len(vals), 4) if vals else None}
        else:
            tot = sum(r[n]["n_gt"] for r in per_image if n in r)
            hit = sum(r[n]["found"] for r in per_image if n in r)
            out[n] = {"labelled_boxes": tot, "found_rate": round(hit / tot, 4) if tot else None}
    return out


def split_dir(data_yaml: Path, split: str) -> Path | None:
    """folder of the pictures of a split, read from data.yaml (a simple 'key: value' file written by prepare_receipts)."""
    root, rel = None, None
    for line in Path(data_yaml).read_text(encoding="utf-8").splitlines():
        if line.startswith("path:"):
            root = line.split(":", 1)[1].strip()
        elif line.startswith(f"{split}:"):
            rel = line.split(":", 1)[1].strip()
    return Path(root) / rel if root and rel else None


def predict_split(model, images_dir: Path, imgsz: int, device=None, conf: float = 0.05, batch: int = 32):
    """yields (picture path, [(cls, conf, xyxy normalised)]) for every picture of a folder."""
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        for p, r in zip(chunk, model.predict(source=[str(f) for f in chunk], imgsz=imgsz, conf=conf, device=device, verbose=False)):
            b = r.boxes
            yield p, [(int(k), float(c), [float(v) for v in xy]) for k, c, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]


def found_rates(model, data_yaml: Path, split: str, imgsz: int, device=None) -> dict[str, dict]:
    d = split_dir(data_yaml, split)
    if d is None or not d.exists():
        return {}
    names = {int(k): v for k, v in model.names.items()}
    per = [match_image(read_labels(p.parent.parent / "labels" / f"{p.stem}.txt"), pr, names) for p, pr in predict_split(model, d, imgsz, device)]
    return summarize(per)
