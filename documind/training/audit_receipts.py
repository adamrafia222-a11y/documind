"""Find the labels that cap your score: python -m documind.training.audit_receipts --split val

Runs models/receipts_yolo.pt on a split and lists
  * MISSING label?  a confident prediction that matches no label (the date IS printed, nobody labelled it)
  * WRONG label?    a labelled box the model does not see at all
into reports/receipts_audit/<split>_suspects.csv, with the pictures (labels in white, predictions in colour) in
reports/receipts_audit/<split>/. Fix the labels in the files of data/processed/receipts_yolo/<split>/labels (the pictures drawn here come from there; keep a copy first),
then train again. Do NOT run STEP 1 again afterwards: it rebuilds data/processed from data/raw and overwrites your fixes.
That is the surest way to raise the score. Do it on val and test too: a score measured on incomplete labels is too low.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .receipts_eval import match_image, predict_split, read_labels, split_dir

DATA_YAML = PROCESSED_DIR / "receipts_yolo" / "data.yaml"
OUT_DIR = REPORTS_DIR / "receipts_audit"
PALETTE = [(255, 59, 48), (52, 199, 89), (10, 132, 255), (255, 159, 10), (191, 90, 242)]


def draw(img_path: Path, gt, preds, names, out_path: Path, why: str) -> None:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    t = max(2, round(max(W, H) / 350))
    for c, b in gt:
        d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(255, 255, 255), width=t + 2)
    for c, cf, b in preds:
        if cf >= 0.25:
            col = PALETTE[c % len(PALETTE)]
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=col, width=t)
            d.text((b[0] * W + 3, max(0, b[1] * H - 12)), f"{names.get(c, c)} {cf:.2f}", fill=col)
    d.text((6, 4), why, fill=(255, 255, 0))
    im.thumbnail((900, 1500))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--weights", default=str(MODELS_DIR / "receipts_yolo.pt"))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--hi-conf", type=float, default=0.6, help="a prediction above this confidence that matches no label is a suspect")
    ap.add_argument("--pictures", type=int, default=60, help="how many suspect pictures to draw")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics is not installed. Run:  pip install -r requirements-receipts.txt")
    model = YOLO(a.weights)
    names = {int(k): v for k, v in model.names.items()}
    d = split_dir(Path(a.data), a.split)
    rows, drawn, counts = [], 0, Counter()
    n_pics = 0
    for p, preds in predict_split(model, d, a.imgsz, a.device):
        n_pics += 1
        gt = read_labels(p.parent.parent / "labels" / f"{p.stem}.txt")
        res = match_image(gt, preds, names, hi_conf=a.hi_conf)
        found = False
        for cname, r in res.items():
            for cf, b, why in r["extra_hi"]:
                rows.append({"picture": p.name, "class": cname, "kind": "MISSING label?", "detail": why, "confidence": round(cf, 2), "box_xyxy_norm": [round(v, 3) for v in b]})
                counts[(cname, "MISSING label?", why)] += 1
                found = True
            for b in r["missed_gt"]:
                rows.append({"picture": p.name, "class": cname, "kind": "WRONG label?", "detail": "labelled box the model does not see", "confidence": "", "box_xyxy_norm": [round(v, 3) for v in b]})
                counts[(cname, "WRONG label?", "")] += 1
                found = True
        if found and drawn < a.pictures:
            draw(p, gt, preds, names, OUT_DIR / a.split / f"{p.stem}.png", "white = label, colour = prediction")
            drawn += 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.split}_suspects.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["picture", "class", "kind", "detail", "confidence", "box_xyxy_norm"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["kind"], -(r["confidence"] or 0))))
    print(f"{n_pics} pictures checked, {len(rows)} suspects -> {out}\n")
    for (c, k, why), n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"  {n:5d}  {c:11s} {k:15s} {why}")
    print(f"\nPictures to look at: {OUT_DIR / a.split}")


if __name__ == "__main__":
    main()
