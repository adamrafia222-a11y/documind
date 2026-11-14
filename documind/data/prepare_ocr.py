"""STEP 1 (OCR) - cut text-line crops + their transcription out of your datasets (receipts, cheques, contracts, FATURA):  ->  data/processed/ocr_crops

    python -m documind.data.prepare_ocr                                   # receipts + cheques + FATURA found in data/raw
    python -m documind.data.prepare_ocr --sources receipts,cheques,contracts   # only the generated ones
    python -m documind.data.prepare_ocr --fatura data/raw/FATURA --max-per-source 40000 --check 150

Output:  <out>/{train,val,test}/images/<source>_<n>.png   and   <out>/{train,val,test}/labels.tsv   (file <TAB> text <TAB> source <TAB> kind)
Splits: receipts, cheques and contracts keep the splits of their generator; FATURA is split by invoice (80 / 10 / 10), so the same invoice never lands
in two splits. Training crops are capped per source (--max-per-source, balanced between the field types so that "item_qty" does not drown
"amount_words"); val and test are capped by --max-eval-per-source.

--check N reads N random val crops per source with Tesseract and prints its error rate (CER = share of characters wrong): that is the
"before" number the trained model has to beat, and a sanity check of the labels (a printed field with a CER of 0.5 means the label is wrong).
reports/ocr_crops_preview.png shows 24 crops with their label: LOOK AT IT.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from ..ocr.crops import SPLITS, Crop, fatura_crops, generated_crops
from ..ocr.metrics import summarize
from .prepare_cheques import find_raw_cheques
from .prepare_contracts import find_raw_contracts
from .prepare_receipts import find_raw_receipts

DEFAULT_OUT = PROCESSED_DIR / "ocr_crops"
ALL_SOURCES = ("receipts", "cheques", "contracts", "fatura")


def store_case_detector():
    """Optional Tesseract-based check for receipt store names. This is disabled by default because running it on every crop can stall the prep job.
    Set DOCUMIND_STORE_CASE_CHECK=1 only when you explicitly want that behavior."""
    import os
    if os.environ.get("DOCUMIND_STORE_CASE_CHECK", "0").lower() not in {"1", "true", "yes", "on"}:
        def is_upper(path, box) -> bool:
            return False
        return is_upper
    try:
        from ..ocr.baseline import read_tesseract
    except Exception:
        def is_upper(path, box) -> bool:
            return False
        return is_upper

    def is_upper(path, box) -> bool:
        try:
            with Image.open(path) as im:
                x0, y0, x1, y1 = box
                crop = im.convert("RGB").crop((x0 - 3, y0 - 3, x1 + 3, y1 + 3))
            letters = [c for c in read_tesseract(crop) if c.isalpha()]
            return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= 0.6
        except Exception:
            return False
    return is_upper


def cap_balanced(crops: list[Crop], cap: int, rng: random.Random) -> list[Crop]:
    """At most `cap` crops, shared as evenly as possible between the kinds (a kind with fewer crops than its share keeps all of them)."""
    if cap <= 0 or len(crops) <= cap:
        return crops
    by = defaultdict(list)
    for c in crops:
        by[c.kind].append(c)
    for lst in by.values():
        rng.shuffle(lst)
    quota, left, kinds = {}, cap, sorted(by, key=lambda k: len(by[k]))
    for i, k in enumerate(kinds):
        share = left // (len(kinds) - i)
        quota[k] = min(len(by[k]), share)
        left -= quota[k]
    out = [c for k in kinds for c in by[k][:quota[k]]]
    rng.shuffle(out)
    return out


def save_crops(crops: list[Crop], out: Path, pad: int) -> list[dict]:
    """Cut and save the crops (every picture is opened once). Returns the label rows."""
    by_image = defaultdict(list)
    for n, c in enumerate(crops):
        by_image[c.image].append((n, c))
    rows, counter = [], Counter()
    for image, items in by_image.items():
        with Image.open(image) as im:
            im = im.convert("RGB")
            for _, c in items:
                x0, y0, x1, y1 = c.box
                crop = im.crop((max(0, int(x0) - pad), max(0, int(y0) - pad), min(im.width, int(x1) + pad + 1), min(im.height, int(y1) + pad + 1)))
                if crop.width < 4 or crop.height < 4:
                    continue
                counter[(c.split, c.source)] += 1
                name = f"{c.source}_{counter[(c.split, c.source)]:06d}.png"
                folder = out / c.split / "images"
                folder.mkdir(parents=True, exist_ok=True)
                crop.save(folder / name, optimize=False, compress_level=1)
                rows.append({"split": c.split, "file": f"images/{name}", "text": c.text, "source": c.source, "kind": c.kind})
    return rows


def preview(out: Path, rows: list[dict], path: Path, n: int = 24, seed: int = 0) -> None:
    rng = random.Random(seed)
    pick = rng.sample(rows, min(n, len(rows)))
    W, rowh = 900, 64
    sheet = Image.new("RGB", (W, rowh * len(pick) + 4), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(pick):
        with Image.open(out / r["split"] / r["file"]) as im:
            im = im.convert("RGB")
            k = min(1.0, (rowh - 14) / im.height, 440 / im.width)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))))
            sheet.paste(im, (4, i * rowh + 4))
        d.text((460, i * rowh + 10), f"[{r['source']}/{r['kind']}]", fill=(90, 90, 90))
        d.text((460, i * rowh + 28), r["text"][:72], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def baseline_check(out: Path, rows: list[dict], n: int, seed: int = 0) -> dict:
    """Tesseract on n random val crops per source -> CER / WER / exact per source and per kind."""
    from ..ocr.baseline import read_tesseract
    rng = random.Random(seed)
    res: dict = {}
    for source in sorted({r["source"] for r in rows}):
        pool = [r for r in rows if r["source"] == source and r["split"] == "val"] or [r for r in rows if r["source"] == source]
        pairs, kinds = [], defaultdict(list)
        for r in rng.sample(pool, min(n, len(pool))):
            with Image.open(out / r["split"] / r["file"]) as im:
                hyp = read_tesseract(im.convert("RGB"))
            pairs.append((r["text"], hyp))
            kinds[r["kind"]].append((r["text"], hyp))
        res[source] = {"all": summarize(pairs), "kinds": {k: summarize(v) for k, v in sorted(kinds.items())}}
    return res


def run(out: Path = DEFAULT_OUT, sources=ALL_SOURCES, receipts: Path | None = None, cheques: Path | None = None, fatura: Path | None = None,
        max_per_source: int = 60000, max_eval_per_source: int = 3000, pad: int = 3, seed: int = 0, check: int = 0, fatura_mode: str = "auto",
        fatura_colored: bool = True, reports_dir: Path = REPORTS_DIR, contracts: Path | None = None) -> dict:
    out, rng, t0 = Path(out), random.Random(seed), time.time()
    found: dict[str, list[Crop]] = {}
    paths = {"receipts": receipts or find_raw_receipts(), "cheques": cheques or find_raw_cheques(), "contracts": contracts or find_raw_contracts(), "fatura": fatura or RAW_DIR / "FATURA"}
    for s in sources:
        root = Path(paths[s])
        if not root.exists():
            print(f"[{s}] folder not found: {root}  (skipped; give it with --{s})")
            continue
        t0_source = time.time()
        if s == "fatura":
            found[s] = list(fatura_crops(root, fatura_mode, fatura_colored))
        else:
            found[s] = list(generated_crops(root, s, store_case_detector() if s == "receipts" else None))
        print(f"[{s}] {len(found[s])} crops found in {root} ({time.time() - t0_source:.1f}s)")
    if not any(found.values()):
        raise SystemExit("No crops found. Give the dataset folders with --receipts / --cheques / --fatura.")
    chosen: list[Crop] = []
    for s, crops in found.items():
        for split in SPLITS:
            part = [c for c in crops if c.split == split]
            chosen += cap_balanced(part, max_per_source if split == "train" else max_eval_per_source, rng)
    print(f"Saving {len(chosen)} crops to {out} ...")
    rows = save_crops(chosen, out, pad)
    print(f"Saved {len(rows)} crops across {len({r['split'] for r in rows})} splits")
    for split in SPLITS:
        lines = [f"{r['file']}\t{r['text'].replace(chr(9), ' ')}\t{r['source']}\t{r['kind']}" for r in rows if r["split"] == split]
        if lines:
            (out / split).mkdir(parents=True, exist_ok=True)
            (out / split / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chars = Counter(ch for r in rows for ch in r["text"])
    stats = {"counts": {f"{sp}/{s}": n for (sp, s), n in sorted(Counter((r["split"], r["source"]) for r in rows).items())},
             "n_chars": len(chars), "non_ascii": "".join(sorted(c for c in chars if ord(c) > 127)), "max_len": max(len(r["text"]) for r in rows),
             "seconds": round(time.time() - t0, 1)}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    preview(out, rows, reports_dir / "ocr_crops_preview.png")
    if check:
        stats["tesseract_baseline"] = baseline_check(out, rows, check, seed)
    (reports_dir / "ocr_crops_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sources", default=",".join(ALL_SOURCES), help="comma separated: receipts, cheques, contracts, fatura")
    ap.add_argument("--receipts", default=None, help="receipt dataset folder (default: data/raw/receipts or data/raw/receipt_dataset)")
    ap.add_argument("--cheques", default=None, help="cheque dataset folder (default: data/raw/cheques or data/raw/cheque_dataset)")
    ap.add_argument("--contracts", default=None, help="contract dataset folder (default: data/raw/contracts or data/raw/contract_dataset)")
    ap.add_argument("--fatura", default=None, help="FATURA folder (default: data/raw/FATURA)")
    ap.add_argument("--max-per-source", type=int, default=60000, help="training crops kept per source (balanced between field types)")
    ap.add_argument("--max-eval-per-source", type=int, default=3000, help="val / test crops kept per source and split")
    ap.add_argument("--pad", type=int, default=3, help="pixels of margin around every box")
    ap.add_argument("--fatura-mode", default="auto", help="FATURA box convention: auto, xyxy, xywh, xyxy_n1000 ... (use if the crops of reports/ocr_crops_preview.png are shifted)")
    ap.add_argument("--no-fatura-colored", action="store_true", help="only the white-background FATURA copies")
    ap.add_argument("--check", type=int, default=0, help="read N random val crops per source with Tesseract and print its error rate (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    st = run(Path(a.out), [s for s in a.sources.split(",") if s], a.receipts and Path(a.receipts), a.cheques and Path(a.cheques), a.fatura and Path(a.fatura),
             a.max_per_source, a.max_eval_per_source, a.pad, a.seed, a.check, a.fatura_mode, not a.no_fatura_colored, contracts=a.contracts and Path(a.contracts))
    print(f"\nWritten to {a.out} in {st['seconds']} s")
    for k, n in st["counts"].items():
        print(f"  {k:16s} {n:7d} crops")
    print(f"characters: {st['n_chars']} different (non-ASCII: {st['non_ascii'] or 'none'}), longest text {st['max_len']} characters")
    for s, b in st.get("tesseract_baseline", {}).items():
        print(f"\nTesseract on {s} (val crops)   CER {b['all']['cer']}   WER {b['all']['wer']}   exact {b['all']['exact']}   n={b['all']['n']}")
        for k, v in b["kinds"].items():
            print(f"    {k:16s} CER {v['cer']:.3f}  exact {v['exact']:.2f}  (n={v['n']})")
    print(f"\nLook at reports/ocr_crops_preview.png, then: python -m documind.training.train_ocr")


if __name__ == "__main__":
    main()
