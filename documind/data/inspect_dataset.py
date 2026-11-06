"""Check that the dataset is found and its labels are understood:

    python -m documind.data.inspect_dataset --data data/raw/FATURA            # files, formats, labels
    python -m documind.data.inspect_dataset --data data/raw/FATURA --check    # + OCR 3 images, test every box
                                                                              #   convention, save overlay PNGs
The overlay images (reports/alignment_check_*.png) show the annotation boxes in green and the OCR words in
red: green boxes must sit on top of the red words.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ..config import REPORTS_DIR
from ..ocr import run_ocr
from ..preprocessing import preprocess, read_image
from .labeling import MODES, _fit_score, _word_boxes, align_labels
from .loader import find_samples, read_annotation, to_pixels


def _check(samples: list[dict], n: int, lang: str, psm: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    step = max(1, len(samples) // n)
    for k, s in enumerate(samples[::step][:n]):
        img = read_image(s["image"])
        h, w = img.shape[:2]
        prep = preprocess(img, deskew=False)
        words = run_ocr(prep.image, scale=prep.scale, lang=lang, psm=psm)
        anns = [read_annotation(p) for p in s["annotations"]]
        print(f"\n--- {s['image'].name}  ({w}x{h}), {len(words)} OCR words")
        wb = _word_boxes(words)
        for p, a in zip(s["annotations"], anns):
            print(f"  {Path(p).parent.name}/{Path(p).name}: {len(a.boxes)} boxes, file says image size {a.ref_size}")
            if not a.boxes:
                print("     (no box found: head of the file) ", Path(p).read_text(encoding='utf-8-sig')[:300].replace("\n", " "))
                continue
            vals = np.array([b.vals for b in a.boxes], dtype=float)
            print(f"     first raw boxes: {[(b.label, [round(v, 1) for v in b.vals]) for b in a.boxes[:3]]}")
            scores = {m: round(_fit_score(wb, to_pixels(vals, m, w, h, a.ref_size)), 2) for m in MODES
                      if not (m.endswith('_ref') and not a.ref_size)}
            print(f"     fit score per box convention: {scores}")
        info = align_labels(words, anns, (w, h))
        print(f"  => chosen: {info}")
        vis = img.copy()
        if info["source"] >= 0:
            a = anns[info["source"]]
            ab = to_pixels(np.array([b.vals for b in a.boxes], dtype=float), info["mode"], w, h, a.ref_size)
            for b, box in zip(a.boxes, ab):
                x0, y0, x1, y1 = [int(v) for v in box]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 170, 0), 2)
                cv2.putText(vis, b.label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)
        for wd in words:
            cv2.rectangle(vis, (int(wd["x0"]), int(wd["y0"])), (int(wd["x1"]), int(wd["y1"])), (0, 0, 255), 1)
        out = REPORTS_DIR / f"alignment_check_{k + 1}.png"
        cv2.imencode(".png", vis)[1].tofile(str(out))
        print(f"  overlay saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--check", action="store_true", help="OCR a few images and test the box conventions")
    ap.add_argument("--n", type=int, default=3, help="images used by --check")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--psm", type=int, default=None)
    args = ap.parse_args()

    samples = find_samples(args.data)
    all_pairs = find_samples(args.data, include_colored=True)
    print(f"{len(samples)} invoices with annotations ({len(all_pairs)} image files incl. coloured copies), "
          f"{len({s['template'] for s in samples})} templates")
    if not samples:
        print("Nothing found: the image and its JSON must share the same file name (e.g. Template1_Instance0).")
        return
    per_t = Counter(s["template"] for s in samples)
    print(f"Images per template: min {min(per_t.values())}, max {max(per_t.values())}")
    s = samples[0]
    img = read_image(s["image"])
    h, w = img.shape[:2]
    print(f"\nFirst sample: {s['image']}  ({w}x{h})")
    for p in s["annotations"]:
        try:
            a = read_annotation(p)
        except Exception as exc:
            print(f"  {p}: cannot parse ({exc})")
            continue
        print(f"  {p}: {len(a.boxes)} boxes")
        for b in a.boxes[:6]:
            print(f"      {b.label:<25} {[round(v, 1) for v in b.vals]}  hint={b.hint}")
        if not a.boxes:
            print("      head of file:", Path(p).read_text(encoding="utf-8-sig")[:300].replace("\n", " "))
    counts = Counter()
    for s in samples[:200]:
        for p in s["annotations"][:1]:
            try:
                counts.update(b.label for b in read_annotation(p).boxes)
            except Exception:
                pass
    print("\nLabels in the first 200 files:", dict(counts.most_common()))
    if args.check:
        from ..config import OCR_LANG, OCR_PSM
        _check(samples, args.n, args.lang or OCR_LANG, args.psm or OCR_PSM)


if __name__ == "__main__":
    main()
