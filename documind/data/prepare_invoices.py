"""STEP 1 (invoices, YOLO) - FATURA  ->  data/raw/invoices_yolo (YOLO layout)  ->  data/processed/invoices_yolo (cleaned)

    python -m documind.data.prepare_invoices --fatura data/raw/FATURA                 # all FATURA images, split by TEMPLATE (honest score)
    python -m documind.data.prepare_invoices --fatura data/raw/FATURA --split random  # split by invoice instead (every layout seen in training)
    python -m documind.data.prepare_invoices --fatura data/raw/FATURA --keep-table   # also keep the TABLE box as a class
    python -m documind.data.prepare_invoices --fatura data/raw/FATURA --photo-copies 2   # + 2 phone-photo-like copies of every training page (low confidence on photos)

The invoice detector works like the receipt, cheque and contract detectors: a YOLO model draws one box on the VALUE of every field, then each box is read.
This step does the part those types get from their generators: it turns the FATURA annotation files into YOLO label files
(documind.data.loader reads them: COCO first, then the other layouts) and splits the invoices into train / val / test.

  * Class names are the FATURA labels (documind.data.labels merges GST(5%) -> GST and BILL_TO -> BUYER).
    TABLE is dropped by default: the line items come from the rules reader (documind.inference.postprocess), not from the detector.
  * The split is by TEMPLATE by default (val and test are layouts never seen in training).
  * Then the shared checks run (documind.data.prepare_receipts): labels validated, duplicates removed, the same picture found in two splits,
    YOLO folders + data.yaml, reports/invoices_prepare_report.txt and reports/invoices_preview/*.png (pictures with their boxes drawn: LOOK AT THEM,
    they show at once whether the coordinates of your annotation files were read the right way).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import PROCESSED_DIR, RAW_DIR
from ..preprocessing.augment import augment_photo_like
from .labels import merge_label
from .loader import find_samples, normalize_label, read_annotation, to_pixels
from .prepare_receipts import main as _main

DEFAULT_DST_RAW = RAW_DIR / "invoices_yolo"
DEFAULT_DST = PROCESSED_DIR / "invoices_yolo"
DEFAULT_DROP = ("TABLE",)


def _pixel_boxes(ann_paths: list[Path], size: tuple[int, int]) -> tuple[list[tuple[str, tuple[float, float, float, float]]], str]:
    """[(label, (x0, y0, x1, y1) in pixels)] from the best annotation file of one invoice, and the convention used.
    COCO files say their own convention (x, y, w, h in the pixels of the image size they name): they are preferred."""
    W, H = size
    first = None
    for p in ann_paths:
        ann = read_annotation(p)
        if not ann.boxes:
            continue
        if all(b.hint == "xywh" for b in ann.boxes):
            mode = "xywh_ref" if ann.ref_size else "xywh"
            return [(b.label, tuple(float(v) for v in to_pixels([b.vals], mode, W, H, ann.ref_size)[0])) for b in ann.boxes], f"COCO ({mode})"
        first = first or ann
    if first is None:
        return [], "none"
    from .loader import parse_annotation
    p = next(p for p in ann_paths if read_annotation(p).boxes)
    return parse_annotation(p, size), "guessed (check reports/invoices_preview)"


def convert(src: Path, dst: Path, drop: tuple[str, ...] = DEFAULT_DROP, split: str = "template", seed: int = 42, include_colored: bool = False,
            max_images: int | None = None, photo_copies: int = 0) -> list[str]:
    """FATURA folder -> YOLO folder (images/<split>, labels/<split>, data.yaml). Returns the class names in id order."""
    from ..training.splits import split_docs
    samples = [s for s in find_samples(src, include_colored=include_colored) if s["annotations"]]
    if not samples:
        raise SystemExit(f"No image with an annotation file in {src}. Check the folder:  python -m documind.data.inspect_dataset --data {src}")
    if max_images and len(samples) > max_images:
        rng = np.random.default_rng(seed)
        samples = [samples[i] for i in rng.permutation(len(samples))[:max_images]]
    boxes_of, conventions = {}, {}
    for i, s in enumerate(samples):
        with Image.open(s["image"]) as im:
            size = im.size
        raw, conv = _pixel_boxes(s["annotations"], size)
        conventions[conv] = conventions.get(conv, 0) + 1
        boxes = []
        for label, (x0, y0, x1, y1) in raw:
            name = merge_label(normalize_label(label))
            x0, x1 = max(0.0, min(x0, x1)), min(float(size[0]), max(x0, x1))
            y0, y1 = max(0.0, min(y0, y1)), min(float(size[1]), max(y0, y1))
            if name and name not in drop and x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append((name, x0, y0, x1, y1))
        boxes_of[i] = (size, boxes)
    names = sorted({b[0] for _, bs in boxes_of.values() for b in bs})
    if not names:
        raise SystemExit("No usable box found in the annotations (all dropped?). Check --drop-labels and the dataset with documind.data.inspect_dataset.")
    train, val, test, how = split_docs([{"template": s["template"], "i": i} for i, s in enumerate(samples)], split, seed)
    print(f"{len(samples)} invoices, {len(names)} classes, split {how}: {len(train)} train / {len(val)} val / {len(test)} test; coordinates: {conventions}")
    rng_aug = np.random.default_rng(seed)
    if dst.exists():
        shutil.rmtree(dst)
    for part, docs in (("train", train), ("val", val), ("test", test)):
        (dst / "images" / part).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / part).mkdir(parents=True, exist_ok=True)
        for d in docs:
            s = samples[d["i"]]
            size, boxes = boxes_of[d["i"]]
            stem = Path(s["image"]).stem
            shutil.copy2(s["image"], dst / "images" / part / (stem + Path(s["image"]).suffix.lower()))
            lines = [f"{names.index(n)} {(x0 + x1) / 2 / size[0]:.6f} {(y0 + y1) / 2 / size[1]:.6f} {(x1 - x0) / size[0]:.6f} {(y1 - y0) / size[1]:.6f}"
                     for n, x0, y0, x1, y1 in boxes]
            (dst / "labels" / part / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            if part == "train" and photo_copies > 0:                       # phone-photo-like copies of the TRAINING pages only (same boxes: only the look changes)
                img = cv2.imdecode(np.fromfile(str(s["image"]), np.uint8), cv2.IMREAD_COLOR)
                for k in range(photo_copies):
                    ph = augment_photo_like(img, rng_aug, lowres=True)
                    cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(dst / "images" / part / f"{stem}_photo{k}.jpg"))
                    (dst / "labels" / part / f"{stem}_photo{k}.txt").write_text("\n".join(lines), encoding="utf-8")
    (dst / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fatura", default=str(RAW_DIR / "FATURA"), help="the FATURA folder (images + annotation files)")
    ap.add_argument("--raw-out", default=str(DEFAULT_DST_RAW), help="where the YOLO-layout copy is written")
    ap.add_argument("--split", choices=["template", "random"], default="template")
    ap.add_argument("--drop-labels", default=",".join(DEFAULT_DROP), help="labels not used as classes (default: TABLE, the line items come from the rules reader)")
    ap.add_argument("--keep-table", action="store_true", help="keep the TABLE box as a class")
    ap.add_argument("--include-colored", action="store_true", help="also use the coloured-background copy of every invoice")
    ap.add_argument("--max-images", type=int, default=None, help="use only this many invoices (quick test)")
    ap.add_argument("--photo-copies", type=int, default=0, help="add N phone-photo-like copies (uneven light, blur, noise, JPEG, lost resolution) of every TRAINING page, "
                    "same boxes: the detector then sees pages that look like your photos (recommended: 2)")
    ap.add_argument("--seed", type=int, default=42)
    own, rest = ap.parse_known_args()
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
        print("\nOwn options: --fatura --raw-out --split --drop-labels --keep-table --include-colored --max-images --seed\n"
              "Then the options of the shared cleaning step (documind.data.prepare_receipts): --out --names --max-side --quality --min-side --workers --preview")
        return
    drop = () if own.keep_table else tuple(x.strip().upper() for x in own.drop_labels.split(",") if x.strip())
    raw = Path(own.raw_out)
    names = convert(Path(own.fatura), raw, drop, own.split, own.seed, own.include_colored, own.max_images, own.photo_copies)
    sys.argv = [sys.argv[0], "--data", str(raw)] + rest
    # an invoice is a full A4 page (about 1650 x 2340) with 10-14 px text: nothing is shrunk before training
    _main(tag="invoices", default_src=raw, default_dst=DEFAULT_DST, default_names=names, default_max_side=2400, doc=__doc__)


if __name__ == "__main__":
    main()
