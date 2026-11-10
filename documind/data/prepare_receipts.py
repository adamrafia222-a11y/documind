"""STEP 1 (receipts) - clean the YOLO receipt dataset: data/raw/receipts  ->  data/processed/receipts_yolo

Two raw layouts are understood (found automatically; labels are `class x_center y_center width height`, all 0..1):
    A) generate_receipts.py (scripts/generate_receipts.py):   <data>/images/{train,val,test}/*.jpg    <data>/labels/{train,val,test}/*.txt   + <data>/data.yaml
    B) one folder per split:                                  <data>/{train,val,test}/images/*.jpg    <data>/{train,val,test}/labels/*.txt

    python scripts/generate_receipts.py --n 10000 --out data/raw/receipts          # make the dataset (13 classes)
    python -m documind.data.prepare_receipts                                       # default folders from documind/config.py
    python -m documind.data.prepare_receipts --data receipt_dataset --workers 4

What it does (the raw data is never modified):
  1. pairs every picture with its label file and reports pictures without label / labels without picture
  2. checks every label line (class id, numbers in 0..1, box inside the picture), fixes what can be fixed
     (small overflow clipped, polygon -> box, duplicate lines removed) and drops what cannot (reported)
  3. straightens phone pictures (EXIF rotation baked into the pixels), converts to RGB JPEG and shrinks pictures larger than
     --max-side (labels are normalised, so they stay valid)
  4. detects the same picture in two splits (leakage: it makes the test score look better than it is)
  5. writes data/processed/receipts_yolo/{train,val,test}/{images,labels} + data.yaml (read by YOLO), and in reports/:
     receipts_prepare_report.txt, receipts_stats.json and receipts_preview/*.png (pictures with their boxes drawn: LOOK AT THEM)

Class names: --names if given, else the `names` of <data>/data.yaml (written by generate_receipts.py), else the 13 classes of
generate_receipts.py:  store_name, address, phone, date, time, receipt_number, item_name, item_qty, item_price, subtotal, tax, total, payment_method.
(The older 5-class data: --names store,date,total,receipt_no,item)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from ..config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
DEFAULT_NAMES = ["store_name", "address", "phone", "date", "time", "receipt_number", "item_name", "item_qty", "item_price", "subtotal", "tax", "total", "payment_method"]
LEGACY_NAMES = ["store", "date", "total", "receipt_no", "item"]               # the first 5-class labelling
PALETTE = [(255, 59, 48), (52, 199, 89), (10, 132, 255), (255, 159, 10), (191, 90, 242), (0, 199, 190), (255, 45, 85), (162, 132, 94)]
def has_pictures_in(p: Path) -> bool:
    """True when the folder looks like a raw dataset (either layout: <p>/images/... or <p>/train/images)."""
    return (Path(p) / "images").is_dir() or (Path(p) / "train" / "images").is_dir()


def find_raw_receipts(raw_dir: Path = RAW_DIR) -> Path:
    """First usual folder name that contains a dataset (either raw layout)."""
    for name in ("receipts", "receipt_dataset"):
        p = raw_dir / name
        if has_pictures_in(p):
            return p
    return raw_dir / "receipts"      # nothing found: the error message will name this path


DEFAULT_SRC = find_raw_receipts()
DEFAULT_DST = PROCESSED_DIR / "receipts_yolo"


# ------------------------------------------------------------------------------------------------ layout and names
def read_names_from_yaml(path: Path) -> list[str] | None:
    """Class names of a YOLO data.yaml ('names: [a, b]' or the block form '  0: a'), without PyYAML. None when not found."""
    if not Path(path).is_file():
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^names\s*:\s*(\[.*\])\s*$", text, re.M)
    if m:
        try:
            return [str(n) for n in ast.literal_eval(m.group(1))]
        except (ValueError, SyntaxError):
            return None
    block = re.search(r"^names\s*:\s*\n((?:[ \t]+\S.*\n?)+)", text, re.M)
    if block:
        pairs = re.findall(r"^[ \t]+(\d+)\s*:\s*(.+?)\s*$", block.group(1), re.M)
        if pairs:
            return [n.strip().strip("'\"") for _, n in sorted(pairs, key=lambda t: int(t[0]))]
    return None


def split_folders(src: Path, split: str) -> tuple[Path, Path]:
    """(pictures folder, labels folder) of a split, whichever layout the raw data uses (B: <split>/images first, then A: images/<split>)."""
    b = (src / split / "images", src / split / "labels")
    a = (src / "images" / split, src / "labels" / split)
    return b if b[0].is_dir() or not a[0].is_dir() else a


# ------------------------------------------------------------------------------------------------ labels
def parse_label(path: Path, min_side: float = 0.002) -> tuple[list[tuple], Counter]:
    """-> (boxes [(cls, xc, yc, w, h)], issue counter). Accepts boxes (5 values) and polygons (class + x y pairs)."""
    boxes, issues, seen = [], Counter(), set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = raw.split()
        if not t:
            continue
        try:
            cls = int(float(t[0])); v = [float(x) for x in t[1:]]
        except ValueError:
            issues["unreadable line"] += 1
            continue
        if len(v) == 4:
            xc, yc, w, h = v
            x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        elif len(v) >= 6 and len(v) % 2 == 0:                                      # polygon -> its bounding box
            xs, ys = v[0::2], v[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            issues["polygon converted to box"] += 1
        else:
            issues["wrong number of values"] += 1
            continue
        if cls < 0 or not all(np.isfinite([x0, y0, x1, y1])):
            issues["invalid class or number"] += 1
            continue
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            far = max(-x0, -y0, x1 - 1, y1 - 1) > 0.05
            issues["box far outside the picture (clipped)" if far else "small overflow clipped"] += 1
        w, h = cx1 - cx0, cy1 - cy0
        if w < min_side or h < min_side:
            issues["box too small (dropped)"] += 1
            continue
        box = (cls, round((cx0 + cx1) / 2, 6), round((cy0 + cy1) / 2, 6), round(w, 6), round(h, 6))
        if box in seen:
            issues["duplicate line removed"] += 1
            continue
        seen.add(box)
        boxes.append(box)
    return boxes, issues


# ------------------------------------------------------------------------------------------------ one picture (runs in a worker)
def process_one(job: dict) -> dict:
    out = {"split": job["split"], "stem": job["stem"], "ok": False, "issues": Counter(), "boxes": [], "hash": None}
    try:
        im = Image.open(job["img"])
        im.load()
        out["orig"] = im.size
        if im.getexif().get(274, 1) not in (None, 1) and not job["ignore_exif"]:
            im = ImageOps.exif_transpose(im)
            out["issues"]["rotated by EXIF (baked into pixels)"] += 1
        im = im.convert("RGB")
        if max(im.size) > job["max_side"]:
            f = job["max_side"] / max(im.size)
            im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
            out["issues"]["resized"] += 1
        out["size"] = im.size
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        out["hash"] = "".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten())     # dHash: same picture at any size
        boxes = []
        if job["label"]:
            boxes, iss = parse_label(Path(job["label"]), job["min_side"])
            out["issues"].update(iss)
        out["boxes"] = boxes
        im.save(job["dst_img"], "JPEG", quality=job["quality"], optimize=True)
        Path(job["dst_lbl"]).write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8")
        out["ok"] = True
    except Exception as e:                                                          # corrupt picture etc.
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def draw_preview(img_path: Path, lbl_path: Path, names: dict[int, str], out_path: Path) -> None:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    t = max(2, round(max(W, H) / 400))
    for line in lbl_path.read_text().splitlines():
        c, x, y, w, h = line.split()
        c = int(c)
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        col = PALETTE[c % len(PALETTE)]
        d.rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), outline=col, width=t)
        d.text((x - w / 2 + 3, max(0, y - h / 2 - 11)), names.get(c, str(c)), fill=col)
    im.thumbnail((900, 1500))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def write_data_yaml(path: Path, dst: Path, names: dict[int, str], splits: list[str]) -> None:
    """data.yaml written by hand (no PyYAML needed); YOLO needs the class ids 0..max without gaps."""
    lines = [f"path: {dst.resolve().as_posix()}"] + [f"{s}: {s}/images" for s in splits] + ["names:"]
    lines += [f"  {i}: {json.dumps(n, ensure_ascii=False)}" for i, n in sorted(names.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ main
def run(src: Path = DEFAULT_SRC, dst: Path = DEFAULT_DST, names_list: list[str] | None = None, max_side: int = 1100, quality: int = 92,
        min_side: float = 0.002, unlabeled: str = "drop", ignore_exif: bool = False, workers: int = 4, preview: int = 6, seed: int = 0,
        reports_dir: Path = REPORTS_DIR, tag: str = "receipts", default_names: list[str] | None = None) -> dict:
    """`tag` names the reports (<tag>_prepare_report.txt, <tag>_stats.json, <tag>_preview/) and the next command (train_<tag>);
    `default_names` are used when neither --names nor <src>/data.yaml gives the class names. Both default to receipts."""
    src, dst, reports_dir = Path(src), Path(dst), Path(reports_dir)
    from_yaml = read_names_from_yaml(src / "data.yaml") if not names_list else None
    names = {i: n.strip() for i, n in enumerate(names_list or from_yaml or default_names or DEFAULT_NAMES)}
    print(f"Class names ({'--names' if names_list else 'data.yaml of the raw data' if from_yaml else 'default'}): {list(names.values())}")
    report: list[str] = []
    jobs: list[dict] = []
    missing_label, orphan_label = defaultdict(list), defaultdict(list)
    for split in SPLITS:
        idir, ldir = split_folders(src, split)
        if not idir.is_dir():
            report.append(f"[{split}] no images folder: {idir}  (split skipped; expected <data>/images/<split> or <data>/<split>/images)")
            continue
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        labels = {p.stem: p for p in ldir.glob("*.txt")} if ldir.is_dir() else {}
        used, taken = set(), set()
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lbl = labels.get(p.stem)
            if lbl is None:
                missing_label[split].append(p.name)
                if unlabeled == "drop":
                    continue
            else:
                used.add(p.stem)
            stem, k = p.stem, 1
            while stem in taken:                                                    # same name with two extensions
                stem, k = f"{p.stem}_{k}", k + 1
            taken.add(stem)
            jobs.append({"split": split, "stem": stem, "img": str(p), "label": str(lbl) if lbl else None, "max_side": max_side, "quality": quality,
                         "min_side": min_side, "ignore_exif": ignore_exif, "dst_img": str(dst / split / "images" / f"{stem}.jpg"),
                         "dst_lbl": str(dst / split / "labels" / f"{stem}.txt")})
        orphan_label[split] = sorted(set(labels) - used - {p.stem for p in idir.iterdir()})
    if not jobs:
        raise SystemExit(f"Nothing to do: expected pictures in {src}/images/train or {src}/train/images ... (see the module docstring)")

    t0 = time.time()
    print(f"Processing {len(jobs)} pictures with {workers} worker(s) ...")
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(tqdm(ex.map(process_one, jobs, chunksize=16), total=len(jobs)))
    else:
        results = [process_one(j) for j in tqdm(jobs)]
    ok = [r for r in results if r["ok"]]
    for r in results:
        if not r["ok"]:
            report.append(f"[{r['split']}] unreadable picture {r['stem']}: {r.get('error')}")

    issue_total: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    for r in ok:
        for k, v in r["issues"].items():
            issue_total[r["split"]][k] += v
            if len(examples[k]) < 3 and k != "resized":
                examples[k].append(r["stem"])
    groups = defaultdict(list)
    for r in ok:
        groups[r["hash"]].append((r["split"], r["stem"]))
    leaks = [g for g in groups.values() if len({sp for sp, _ in g}) > 1]

    stats: dict = {"splits": {}, "names": names}
    for split in SPLITS:
        rs = [r for r in ok if r["split"] == split]
        if not rs:
            continue
        cls = Counter(b[0] for r in rs for b in r["boxes"])
        per_img = [len(r["boxes"]) for r in rs]
        stats["splits"][split] = {
            "images": len(rs), "boxes": int(sum(cls.values())), "boxes_per_image_mean": round(float(np.mean(per_img)), 2),
            "images_without_boxes": int(sum(1 for x in per_img if x == 0)), "per_class": {names.get(c, f"class_{c}"): n for c, n in sorted(cls.items())},
            "median_box_wh": {names.get(c, f"class_{c}"): [round(float(np.median([b[3] for r in rs for b in r["boxes"] if b[0] == c])), 4),
                                                           round(float(np.median([b[4] for r in rs for b in r["boxes"] if b[0] == c])), 4)] for c in sorted(cls)}}
    found = sorted({b[0] for r in ok for b in r["boxes"]})
    top = max(max(found, default=-1), len(names) - 1)
    for c in range(top + 1):
        names.setdefault(c, f"class_{c}")
    train_cls = {b[0] for r in ok if r["split"] == "train" for b in r["boxes"]}
    for split in ("val", "test"):
        miss = sorted({b[0] for r in ok if r["split"] == split for b in r["boxes"]} - train_cls)
        if miss:
            report.append(f"WARNING [{split}] contains classes that never appear in train: {[names[c] for c in miss]}")
    for c in range(top + 1):
        if c not in found:
            report.append(f"WARNING class id {c} ({names[c]}) has no box anywhere: check the numbering")

    write_data_yaml(dst / "data.yaml", dst, names, [s for s in SPLITS if s in stats["splits"]])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{tag}_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    rng = random.Random(seed)
    for split in stats["splits"]:
        for r in rng.sample([r for r in ok if r["split"] == split], min(preview, stats["splits"][split]["images"])):
            draw_preview(dst / split / "images" / f"{r['stem']}.jpg", dst / split / "labels" / f"{r['stem']}.txt", names,
                         reports_dir / f"{tag}_preview" / f"{split}_{r['stem']}.png")

    lines = [f"Prepared {len(ok)} pictures in {time.time() - t0:.0f}s -> {dst}", ""]
    for split, s in stats["splits"].items():
        lines.append(f"{split:5s} {s['images']:6d} pictures {s['boxes']:7d} boxes ({s['boxes_per_image_mean']} per picture)  {s['per_class']}")
    for split in SPLITS:
        if missing_label[split]:
            lines.append(f"[{split}] {len(missing_label[split])} pictures have no label file ({'dropped' if unlabeled == 'drop' else 'kept as background'}): e.g. {missing_label[split][:3]}")
        if orphan_label[split]:
            lines.append(f"[{split}] {len(orphan_label[split])} label files have no picture: e.g. {orphan_label[split][:3]}")
        for k, v in sorted(issue_total[split].items()):
            lines.append(f"[{split}] {v:6d} x {k}" + (f"   e.g. {examples[k]}" if examples.get(k) else ""))
    if leaks:
        lines.append(f"WARNING {len(leaks)} group(s) of identical pictures ({sum(len(g) for g in leaks)} files) appear in more than one split, e.g. "
                     + "; ".join(", ".join(f"{sp}/{st}" for sp, st in g[:4]) for g in leaks[:2])
                     + ". The val/test scores would be too good: keep each picture in ONE split.")
    lines += report
    text = "\n".join(lines)
    (reports_dir / f"{tag}_prepare_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\nNext: look at {reports_dir / (tag + '_preview')}, then: python -m documind.training.train_{tag}")
    return stats


def main(tag: str = "receipts", default_src: Path | None = None, default_dst: Path | None = None, default_names: list[str] | None = None,
         default_max_side: int = 1100, doc: str | None = None) -> None:
    default_src, default_dst = default_src or DEFAULT_SRC, default_dst or DEFAULT_DST
    ap = argparse.ArgumentParser(description=doc or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(default_src), help="raw dataset: <data>/images/<split> + <data>/labels/<split> (as written by the generators) or <data>/<split>/{images,labels}  (default: %(default)s)")
    ap.add_argument("--out", default=str(default_dst), help="cleaned dataset (default: %(default)s)")
    ap.add_argument("--names", default=None, help="class names in id order, comma separated (default: the names of <data>/data.yaml, else the generator's classes)")
    ap.add_argument("--max-side", type=int, default=default_max_side, help="longest side in pixels after resizing (the training --imgsz should be about this value / 1.15)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--min-side", type=float, default=0.002, help="drop boxes thinner than this fraction of the picture")
    ap.add_argument("--unlabeled", choices=["drop", "keep"], default="drop", help="pictures with no label file: drop them (default) or keep them as background")
    ap.add_argument("--ignore-exif", action="store_true", help="do not apply the EXIF rotation (use if the boxes were drawn on the raw, unrotated pixels)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--preview", type=int, default=6, help=f"pictures drawn with their boxes per split (saved in reports/{tag}_preview)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(Path(a.data), Path(a.out), a.names.split(",") if a.names else None, a.max_side, a.quality, a.min_side, a.unlabeled, a.ignore_exif, a.workers, a.preview, a.seed,
        tag=tag, default_names=default_names)


if __name__ == "__main__":
    main()
