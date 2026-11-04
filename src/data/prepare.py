"""STEP 1 - data preparation: raw images + labels  ->  data/processed/corpus.jsonl

For every selected image: pre-processing -> OCR -> alignment of OCR words with the labelled boxes.
The OCR result is cached in data/interim/ocr_cache (words + boxes only, NO labels), so a second run is fast
and a change in the label logic never needs a new OCR pass.

    python -m documind.data.prepare --data data/raw/FATURA --workers 4                 # ALL images
    python -m documind.data.prepare --data data/raw/FATURA --per-template 40 --workers 4   # 40 per template
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import BACKGROUND_LABEL, CORPUS_PATH, INTERIM_DIR, OCR_CACHE_DIR, OCR_LANG, OCR_PSM
from ..ocr import run_ocr
from ..preprocessing import augment_photo_like, preprocess, read_image
from .corpus import save_corpus
from .labeling import align_labels
from .loader import find_samples, read_annotation, select_samples

DEFAULT_DROP = "TABLE"      # line items are not extracted (see README) -> their words stay background


def _cache_path(sample: dict, lang: str, psm: int, augment: bool) -> Path:
    key = hashlib.md5(f"{sample['image']}|{lang}|{psm}|{augment}".encode()).hexdigest()[:10]
    return OCR_CACHE_DIR / f"ocr_{lang}_{psm}_{'aug' if augment else 'clean'}" / f"{sample['stem']}_{key}.json"


def _ocr_words(sample: dict, lang: str, psm: int, augment: bool, seed: int) -> tuple[list[dict], int, int]:
    """OCR words (original-image coordinates) + image size. Cached; the cache never stores labels."""
    cache = _cache_path(sample, lang, psm, augment)
    if cache.exists():
        doc = json.loads(cache.read_text(encoding="utf-8"))
        words = [{k: v for k, v in w.items() if k != "label"} for w in doc["words"]]
        return words, int(doc["width"]), int(doc["height"])
    img = read_image(sample["image"])
    h, w = img.shape[:2]
    if augment:
        seed_i = int(hashlib.md5(f"{sample['stem']}|{seed}".encode()).hexdigest()[:8], 16)
        img = augment_photo_like(img, np.random.default_rng(seed_i))
    # no deskew in training: it would rotate the image and break the box alignment
    prep = preprocess(img, deskew=False)
    words = run_ocr(prep.image, scale=prep.scale, lang=lang, psm=psm)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stem": sample["stem"], "width": w, "height": h, "words": words}), encoding="utf-8")
    return words, w, h


def _process(args: tuple) -> dict | None:
    sample, lang, psm, augment, seed, drop = args
    try:
        words, w, h = _ocr_words(sample, lang, psm, augment, seed)
        annotations, errors = [], []
        for p in sample["annotations"]:
            try:
                annotations.append(read_annotation(p))
            except Exception as exc:                      # a broken annotation file must not hide the others
                errors.append(f"{Path(p).name}: {exc}")
        info = align_labels(words, annotations, (w, h), drop=set(drop))
        return {"stem": sample["stem"], "template": sample["template"], "width": w, "height": h,
                "words": words, "n_annotations": info["n_boxes"], "align": info["mode"],
                "align_source": Path(sample["annotations"][info["source"]]).parent.name if info["source"] >= 0 else None,
                "align_fit": info["fit"], "ann_errors": errors}
    except Exception as exc:  # keep going if one file is broken
        print(f"[skip] {sample['image']}: {exc}")
        return None


def build_corpus(samples: list[dict], *, lang: str = OCR_LANG, psm: int = OCR_PSM, workers: int = 4,
                 augment: bool = False, seed: int = 42, drop: tuple[str, ...] = (DEFAULT_DROP,)) -> list[dict]:
    jobs = [(s, lang, psm, augment and (i % 2 == 0), seed, tuple(drop)) for i, s in enumerate(samples)]
    docs: list[dict] = []
    if workers <= 1:
        results = (_process(j) for j in jobs)
        for r in tqdm(results, total=len(jobs), desc="OCR + label alignment"):
            if r:
                docs.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in tqdm(pool.map(_process, jobs, chunksize=4), total=len(jobs), desc="OCR + label alignment"):
                if r:
                    docs.append(r)
    return [d for d in docs if d["words"]]


def _write_debug(docs: list[dict], samples: list[dict]) -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / "alignment_debug.json"
    by_stem = {s["stem"]: s for s in samples}
    rows = []
    for d in docs[:3]:
        s = by_stem.get(d["stem"], {})
        files = []
        for p in s.get("annotations", []):
            try:
                ann = read_annotation(p)
                files.append({"file": str(p), "n_boxes": len(ann.boxes), "ref_size": ann.ref_size,
                              "first_boxes": [(b.label, b.vals, b.hint) for b in ann.boxes[:5]],
                              "head": Path(p).read_text(encoding="utf-8-sig")[:600]})
            except Exception as exc:
                files.append({"file": str(p), "error": str(exc)})
        rows.append({"image": str(s.get("image")), "size": [d["width"], d["height"]],
                     "first_ocr_words": [(w["text"], round(w["x0"]), round(w["y0"]), round(w["x1"]), round(w["y1"]))
                                         for w in d["words"][:8]], "annotation_files": files})
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset folder, e.g. data/raw/FATURA (images + annotation JSON)")
    ap.add_argument("--out", default=str(CORPUS_PATH), help="output corpus file")
    ap.add_argument("--per-template", type=int, default=0,
                    help="images used per template; 0 (default) = ALL images (FATURA: 200 per template)")
    ap.add_argument("--max-images", type=int, default=0, help="global cap; 0 (default) = no cap")
    ap.add_argument("--include-colored", action="store_true",
                    help="FATURA has a white and a coloured copy of every invoice: use both (default: one copy)")
    ap.add_argument("--drop-labels", default=DEFAULT_DROP,
                    help="comma-separated labels turned into background (default TABLE; '' = keep everything)")
    ap.add_argument("--workers", type=int, default=4, help="parallel OCR processes")
    ap.add_argument("--lang", default=OCR_LANG)
    ap.add_argument("--psm", type=int, default=OCR_PSM)
    ap.add_argument("--augment", action="store_true", help="make half of the images look like phone photos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_samples = find_samples(args.data, include_colored=args.include_colored)
    if not all_samples:
        raise SystemExit(f"No image+JSON pairs found under {args.data}. "
                         "Run: python -m documind.data.inspect_dataset --data " + args.data)
    samples = select_samples(all_samples, args.per_template, args.max_images, args.seed)
    n_t = len({s["template"] for s in samples})
    print(f"{len(all_samples)} images with annotations found -> {len(samples)} used, {n_t} templates"
          f"{'  (ALL images)' if len(samples) == len(all_samples) else ''}")
    drop = tuple(x.strip().upper() for x in args.drop_labels.split(",") if x.strip())

    docs = build_corpus(samples, lang=args.lang, psm=args.psm, workers=args.workers,
                        augment=args.augment, seed=args.seed, drop=drop)
    n_words = sum(len(d["words"]) for d in docs)
    n_fields = sum(w["label"] != BACKGROUND_LABEL for d in docs for w in d["words"])
    n_docs_lab = sum(any(w["label"] != BACKGROUND_LABEL for w in d["words"]) for d in docs)
    modes = Counter(d["align"] for d in docs)
    classes = Counter(w["label"] for d in docs for w in d["words"] if w["label"] != BACKGROUND_LABEL)
    errors = [e for d in docs for e in d.get("ann_errors", [])]
    print(f"{len(docs)} documents, {n_words} OCR words, {n_fields} of them belong to a field "
          f"({n_docs_lab}/{len(docs)} documents have at least one labelled word)")
    print(f"Coordinate convention detected: {dict(modes)}")
    print("Words per field:", dict(classes.most_common()))
    if errors:
        print(f"[warn] {len(errors)} annotation files could not be read, e.g. {errors[0]}")

    if n_fields == 0:
        dbg = _write_debug(docs, samples)
        raise SystemExit(
            "No OCR word was matched to a label, so the corpus was NOT written "
            f"(the existing {args.out} is unchanged: do not train on it).\n"
            f"Details of 3 documents: {dbg}\n"
            "Visual check: python -m documind.data.inspect_dataset --data " + args.data + " --check")
    if n_docs_lab < 0.8 * len(docs):
        print(f"[warn] only {n_docs_lab}/{len(docs)} documents got labels: check the data with "
              "python -m documind.data.inspect_dataset --data " + args.data + " --check")

    meta = {"created_at": datetime.now().isoformat(timespec="seconds"), "source": str(args.data),
            "n_documents": len(docs), "n_images_available": len(all_samples), "per_template": args.per_template,
            "n_words": n_words, "n_field_words": n_fields, "ocr_lang": args.lang, "ocr_psm": args.psm,
            "augment": args.augment, "seed": args.seed, "include_colored": args.include_colored,
            "dropped_labels": list(drop), "alignment_modes": dict(modes), "words_per_field": dict(classes),
            "templates": sorted({d["template"] for d in docs})}
    save_corpus(docs, meta, args.out)
    print(f"Corpus saved to {args.out}")


if __name__ == "__main__":
    main()
