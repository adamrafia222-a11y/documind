#!/usr/bin/env python3
"""prepare_receipts.py - preprocessing of the YOLO receipt dataset (train / val / test, each with images/ and labels/).

    python prepare_receipts.py --src data/raw/receipts --dst data/processed/receipts_yolo

What it does (raw data is never modified):
  1. pairs every picture with its label file and reports pictures without label / labels without picture
  2. checks every label line (class id, 4 numbers in 0..1, box inside the picture), fixes what is fixable
     (tiny overflow clipped, polygon -> box, duplicated lines removed) and drops what is not (reported)
  3. straightens phone pictures (EXIF rotation baked into the pixels), converts to RGB JPEG, shrinks pictures larger than --max-side
     (labels are normalised, so they stay valid) - a 4 MB phone photo becomes about 300 KB and training becomes much faster
  4. detects the same picture appearing in two splits (leakage: it would make the test score look better than it is)
  5. writes <dst>/{train,val,test}/{images,labels}, <dst>/data.yaml (what YOLO reads), dataset_stats.json,
     prepare_report.txt and <dst>/preview/*.png (pictures with their boxes drawn: LOOK AT THEM)

Class ids -> names (from your example picture: 0 logo/store name, 1 date, 2 total, 3 receipt number, 4 item rows).
Change them with --names if the other pictures say otherwise.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
DEFAULT_NAMES = ["store", "date", "total", "receipt_no", "item"]
PALETTE = [(255, 59, 48), (52, 199, 89), (10, 132, 255), (255, 159, 10), (191, 90, 242), (0, 199, 190), (255, 45, 85), (162, 132, 94)]


# ------------------------------------------------------------------------------------------------ labels
def parse_label(path: Path, min_side: float) -> tuple[list[tuple], Counter]:
    """-> (boxes [(cls, xc, yc, w, h)], issue counter). Accepts boxes (5 values) and polygons (class + x y pairs)."""
    boxes, issues, seen = [], Counter(), set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = raw.split()
        if not t:
            continue
        try:
            cls = int(float(t[0])); v = [float(x) for x in t[1:]]
        except ValueError:
            issues["unreadable line"] += 1; continue
        if len(v) == 4:
            xc, yc, w, h = v; x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        elif len(v) >= 6 and len(v) % 2 == 0:                                      # polygon -> its bounding box
            xs, ys = v[0::2], v[1::2]; x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys); issues["polygon converted to box"] += 1
        else:
            issues["wrong number of values"] += 1; continue
        if cls < 0 or not all(np.isfinite([x0, y0, x1, y1])):
            issues["invalid class or number"] += 1; continue
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            if max(-x0, -y0, x1 - 1, y1 - 1) > 0.05:
                issues["box far outside the picture (clipped)"] += 1
            else:
                issues["tiny overflow clipped"] += 1
        w, h = cx1 - cx0, cy1 - cy0
        if w < min_side or h < min_side:
            issues["box too small (dropped)"] += 1; continue
        box = (cls, round((cx0 + cx1) / 2, 6), round((cy0 + cy1) / 2, 6), round(w, 6), round(h, 6))
        if box in seen:
            issues["duplicate line removed"] += 1; continue
        seen.add(box); boxes.append(box)
    return boxes, issues


# ------------------------------------------------------------------------------------------------ one picture (runs in a worker)
def process_one(job: dict) -> dict:
    out = {"split": job["split"], "stem": job["stem"], "ok": False, "issues": Counter(), "boxes": [], "hash": None}
    try:
        im = Image.open(job["img"]); im.load()
        out["orig"] = im.size
        orient = im.getexif().get(274, 1)
        if orient not in (None, 1) and not job["ignore_exif"]:
            im = ImageOps.exif_transpose(im); out["issues"]["rotated by EXIF (baked into pixels)"] += 1
        im = im.convert("RGB")
        s = max(im.size)
        if s > job["max_side"]:
            f = job["max_side"] / s; im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS); out["issues"]["resized"] += 1
        out["size"] = im.size
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        out["hash"] = "".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten())     # dHash: same picture, any size
        boxes = []
        if job["label"]:
            boxes, iss = parse_label(Path(job["label"]), job["min_side"]); out["issues"].update(iss)
        out["boxes"] = boxes
        dst_img, dst_lbl = Path(job["dst_img"]), Path(job["dst_lbl"])
        im.save(dst_img, "JPEG", quality=job["quality"], optimize=True)
        dst_lbl.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8")
        out["ok"] = True
    except Exception as e:                                                          # corrupt picture etc.
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def draw_preview(img_path: Path, lbl_path: Path, names: dict[int, str], out_path: Path) -> None:
    im = Image.open(img_path).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im); t = max(2, round(max(W, H) / 400))
    for line in lbl_path.read_text().splitlines():
        c, x, y, w, h = line.split(); c = int(c); x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        col = PALETTE[c % len(PALETTE)]; d.rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), outline=col, width=t)
        d.text((x - w / 2 + 3, max(0, y - h / 2 - 11)), names.get(c, str(c)), fill=col)
    im.thumbnail((900, 1500)); out_path.parent.mkdir(parents=True, exist_ok=True); im.save(out_path)


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="raw dataset: <src>/{train,val,test}/{images,labels}")
    ap.add_argument("--dst", required=True, help="where the cleaned dataset is written")
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES), help="class names in id order, comma separated")
    ap.add_argument("--max-side", type=int, default=1600, help="longest side in pixels after resizing (1600 is plenty for imgsz 960-1280)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--min-side", type=float, default=0.002, help="drop boxes thinner than this fraction of the picture")
    ap.add_argument("--unlabeled", choices=["drop", "keep"], default="drop", help="pictures with no label file: drop them (default) or keep them as background")
    ap.add_argument("--ignore-exif", action="store_true", help="do not apply the EXIF rotation (use if the boxes were drawn on the raw, unrotated pixels)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--preview", type=int, default=6, help="pictures drawn with their boxes per split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    names = {i: n.strip() for i, n in enumerate(a.names.split(","))}
    report: list[str] = []
    jobs, missing_label, orphan_label = [], defaultdict(list), defaultdict(list)
    for split in SPLITS:
        idir, ldir = src / split / "images", src / split / "labels"
        if not idir.is_dir():
            report.append(f"[{split}] no images folder: {idir}  (split skipped)"); continue
        (dst / split / "images").mkdir(parents=True, exist_ok=True); (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        labels = {p.stem: p for p in ldir.glob("*.txt")} if ldir.is_dir() else {}
        used = set()
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lbl = labels.get(p.stem)
            if lbl is None:
                missing_label[split].append(p.name)
                if a.unlabeled == "drop":
                    continue
            else:
                used.add(p.stem)
            stem, k = p.stem, 1
            while (dst / split / "images" / f"{stem}.jpg").exists() or any(j["split"] == split and j["stem"] == stem for j in jobs[-3:]):
                stem, k = f"{p.stem}_{k}", k + 1                                    # same name with a different extension
            jobs.append({"split": split, "stem": stem, "img": str(p), "label": str(lbl) if lbl else None, "max_side": a.max_side, "quality": a.quality,
                         "min_side": a.min_side, "ignore_exif": a.ignore_exif, "dst_img": str(dst / split / "images" / f"{stem}.jpg"), "dst_lbl": str(dst / split / "labels" / f"{stem}.txt")})
        orphan_label[split] = sorted(set(labels) - used - {p.stem for p in idir.iterdir()})
    if not jobs:
        sys.exit(f"nothing to do: expected {src}/train/images etc.")

    t0 = time.time(); results = []
    print(f"processing {len(jobs)} pictures with {a.workers} worker(s) ...")
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(process_one, jobs, chunksize=16), 1):
                results.append(r)
                if i % 500 == 0:
                    print(f"  {i}/{len(jobs)}  ({i / (time.time() - t0):.0f} pictures/s)", flush=True)
    else:
        results = [process_one(j) for j in jobs]

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    for r in bad:
        report.append(f"[{r['split']}] unreadable picture {r['stem']}: {r.get('error')}")
    # --- issues summary
    issue_total: dict[str, Counter] = defaultdict(Counter); examples: dict[str, list] = defaultdict(list)
    for r in ok:
        for k, v in r["issues"].items():
            issue_total[r["split"]][k] += v
            if len(examples[k]) < 3 and "resized" not in k:
                examples[k].append(r["stem"])
    # --- leakage: the same picture in two splits
    groups = defaultdict(list)
    for r in ok:
        groups[r["hash"]].append((r["split"], r["stem"]))
    leaks = [g for g in groups.values() if len({sp for sp, _ in g}) > 1]
    # --- statistics
    stats = {"splits": {}, "names": names}
    for split in SPLITS:
        rs = [r for r in ok if r["split"] == split]
        if not rs:
            continue
        cls = Counter(b[0] for r in rs for b in r["boxes"])
        per_img = [len(r["boxes"]) for r in rs]
        stats["splits"][split] = {"images": len(rs), "boxes": int(sum(cls.values())), "boxes_per_image_mean": round(float(np.mean(per_img)), 2), "images_without_boxes": int(sum(1 for x in per_img if x == 0)),
                                  "per_class": {names.get(c, f"class_{c}"): n for c, n in sorted(cls.items())},
                                  "median_box_wh": {names.get(c, f"class_{c}"): [round(float(np.median([b[3] for r in rs for b in r["boxes"] if b[0] == c])), 4), round(float(np.median([b[4] for r in rs for b in r["boxes"] if b[0] == c])), 4)] for c in sorted(cls)}}
    found = sorted({b[0] for r in ok for b in r["boxes"]})
    for c in found:
        names.setdefault(c, f"class_{c}")
    train_cls = {b[0] for r in ok if r["split"] == "train" for b in r["boxes"]}
    for split in ("val", "test"):
        miss = sorted({b[0] for r in ok if r["split"] == split for b in r["boxes"]} - train_cls)
        if miss:
            report.append(f"WARNING [{split}] contains classes that never appear in train: {[names[c] for c in miss]}")
    for c in range(max(found) + 1 if found else 0):
        if c not in found:
            report.append(f"WARNING class id {c} ({names.get(c, f'class_{c}')}) has no box anywhere: check the numbering")
    # --- data.yaml
    top = max(max(found, default=-1), len(DEFAULT_NAMES) - 1)                       # YOLO needs the ids 0..max without gaps
    cfg = {"path": dst.resolve().as_posix(), "names": {c: names.get(c, f"class_{c}") for c in range(top + 1)}}
    for split in SPLITS:
        if split in stats["splits"]:
            cfg[split] = f"{split}/images"
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (dst / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # --- previews
    rng = random.Random(a.seed)
    for split in stats["splits"]:
        pick = rng.sample([r for r in ok if r["split"] == split], min(a.preview, stats["splits"][split]["images"]))
        for r in pick:
            draw_preview(dst / split / "images" / f"{r['stem']}.jpg", dst / split / "labels" / f"{r['stem']}.txt", names, dst / "preview" / f"{split}_{r['stem']}.png")
    # --- report
    lines = [f"prepared {len(ok)} pictures in {time.time() - t0:.0f}s -> {dst}", ""]
    for split, s in stats["splits"].items():
        lines.append(f"{split:5s} {s['images']:6d} pictures {s['boxes']:7d} boxes ({s['boxes_per_image_mean']} per picture)  {s['per_class']}")
    for split in SPLITS:
        if missing_label[split]:
            lines.append(f"[{split}] {len(missing_label[split])} pictures have no label file ({'dropped' if a.unlabeled == 'drop' else 'kept as background'}): e.g. {missing_label[split][:3]}")
        if orphan_label[split]:
            lines.append(f"[{split}] {len(orphan_label[split])} label files have no picture: e.g. {orphan_label[split][:3]}")
        for k, v in sorted(issue_total[split].items()):
            lines.append(f"[{split}] {v:6d} x {k}" + (f"   e.g. {examples[k]}" if k in examples and examples[k] else ""))
    if leaks:
        n_pics = sum(len(g) for g in leaks)
        lines.append(f"WARNING {len(leaks)} group(s) of identical pictures ({n_pics} files) appear in more than one split, e.g. "
                     + "; ".join(", ".join(f"{sp}/{st}" for sp, st in g[:4]) for g in leaks[:2])
                     + ". The val/test scores would be too good: keep each picture in ONE split.")
    lines += report
    text = "\n".join(lines)
    (dst / "prepare_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\nNext: look at {dst / 'preview'}, then train with data={dst / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
