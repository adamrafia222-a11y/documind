"""Test of the OCR reconstruction on YOUR labelled documents: the real upload pipeline three ways.

    off   nothing                          auto  the PICTURE is straightened (stage A)          on  picture + the WORDS re-read by the OCR model (A + B)

    python -m documind.training.compare_reconstruct --data data/raw/FATURA --n 60 --photo                       # invoices
    python -m documind.training.compare_reconstruct --type contract --data data/raw/contracts --n 60 --photo    # contracts (ground_truth.jsonl of generate_contracts.py)

For every image the annotation file gives the true text of each field (FATURA does). The extractor reads the picture twice (off / on) and each
field is scored: exact match (ignoring case and spaces) and character similarity (1.0 = identical). The pictures are the same in both runs.

    --photo     turn the clean images into phone-like photos first (on a table, tilted, in perspective, shadow, blur): this is what the reconstruction is for.
                Without it the images are the clean originals and the difference is small (they are already in training form).
    --n         number of images (default 60)        --seed   which images (default 42)
    (the invoice detector models/invoices_yolo.pt, or the contract detector models/contracts_yolo.pt + the text reader, is the one that is installed)
    --type      invoice (default) or contract

Writes reports/compare_reconstruct.csv (one line per field and image) and prints the totals and the 15 fields that changed most.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..config import REPORTS_DIR
from ..data.loader import find_samples, normalize_label, read_annotation, select_samples
from ..preprocessing import read_image

DROP = {"TABLE"}


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", (s or "").lower()) if not unicodedata.combining(c))     # CRÉDIT == credit: the OCR often drops accents
    return re.sub(r"\s+", "", s)


def similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def to_photo(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A clean page -> a phone-like photo: dark table around it, tilt and perspective, uneven light, blur, JPEG noise."""
    h, w = img.shape[:2]
    m = int(0.12 * max(h, w))
    bg = np.full((h + 2 * m, w + 2 * m, 3), rng.integers(35, 90), np.uint8)
    bg[:] = bg + rng.integers(-8, 8, size=bg.shape, dtype=np.int16).clip(-8, 8).astype(np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = lambda: rng.uniform(-0.03, 0.03) * max(h, w)
    dst = np.float32([[m + jit(), m + jit()], [m + w + jit(), m + jit()], [m + w + jit(), m + h + jit()], [m + jit(), m + h + jit()]])
    ang = np.deg2rad(rng.uniform(-8, 8))
    c = np.float32([(w + 2 * m) / 2, (h + 2 * m) / 2])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = (dst - c) @ R.T + c
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    size = (w + 2 * m, h + 2 * m)
    page = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size)
    out = np.where(mask[:, :, None] > 0, page, bg).astype(np.float32)
    gradient = np.linspace(rng.uniform(0.55, 0.9), rng.uniform(0.95, 1.1), size[0], dtype=np.float32)
    out = out * gradient[None, :, None] * rng.uniform(0.85, 1.1)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), rng.uniform(0.5, 1.3))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 85))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def truth(sample: dict) -> dict[str, list[str]]:
    """label -> the true texts of that field, from the annotation file (boxes without a transcription are ignored)."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in sample["annotations"]:                       # FATURA has several annotation formats per image: the first one that carries the text
        for b in read_annotation(p).boxes:
            lab = normalize_label(b.label)
            if lab and lab not in DROP and b.text and b.text.strip():
                out[lab].append(b.text.strip())
        if out:
            break
    return out


def score(pred_fields: list[dict], true: dict[str, list[str]]) -> dict[str, tuple[int, float]]:
    """label -> (exact 0/1, similarity) of the best predicted text of that label against the true text."""
    got: dict[str, list[str]] = defaultdict(list)
    for f in pred_fields:
        got[f["label"]].append(f["text"])
    res = {}
    for lab, texts in true.items():
        truth_text = " ".join(texts)
        cands = got.get(lab, []) + ([" ".join(got[lab])] if len(got.get(lab, [])) > 1 else [])
        best = max((similarity(c, truth_text) for c in cands), default=0.0)
        res[lab] = (int(best == 1.0), best)
    return res



# ------------------------------------------------------------------------------------------------ contracts
CONTRACT_TEXT = {"title": "TITLE", "contract_number": "CONTRACT_NO", "lender_name": "LENDER", "borrower_name": "BORROWER", "borrower_id": "BORROWER_ID",
                 "borrower_address": "BORROWER_ADDRESS", "amount_words": "AMOUNT_WORDS", "iban": "IBAN", "place": "PLACE"}
CONTRACT_DATES = {"contract_date": "CONTRACT_DATE", "first_payment_date": "FIRST_PAYMENT_DATE", "signing_date": "SIGNING_DATE"}
CONTRACT_NUMBERS = {"amount_value": "AMOUNT", "rate_value": "RATE_PCT", "months": "DURATION_MONTHS", "monthly_value": "MONTHLY_PAYMENT"}
_ROW = {"RATE_PCT": "RATE"}


def _digits(s: str) -> str:
    """26/07/2025 and 2025-07-26 -> 26072025."""
    s = (s or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    return (m.group(3) + m.group(2) + m.group(1)) if m else re.sub(r"\D", "", s)


def contract_samples(root: str) -> list[dict]:
    out = []
    for gt in Path(root).rglob("ground_truth.jsonl"):
        for line in gt.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            img = gt.parent / r["file_name"]
            if img.is_file():
                out.append({"image": img, "truth": r["ground_truth"], "stem": img.stem, "split": r.get("split")})
    return out


def score_contract(row: dict, gt: dict) -> dict[str, tuple[int, float]]:
    res = {}
    for k, col in CONTRACT_TEXT.items():
        if gt.get(k):
            sim = similarity(str(row.get(col, "")), gt[k])
            res[col] = (int(sim == 1.0), sim)
    for k, col in CONTRACT_DATES.items():
        if gt.get(k):
            ok = _digits(str(row.get(col, ""))) == _digits(gt[k])
            res[col] = (int(ok), 1.0 if ok else similarity(_digits(str(row.get(col, ""))), _digits(gt[k])))
    for k, col in CONTRACT_NUMBERS.items():
        if gt.get(k) is not None:
            try:
                got = float(str(row.get(_ROW.get(col, col), "")).replace(",", "."))
            except ValueError:
                got = None
            ok = got is not None and abs(got - float(gt[k])) < 0.006
            res[col] = (int(ok), 1.0 if ok else similarity(str(row.get(_ROW.get(col, col), "")), str(gt[k])))
    return res


def run_mode_contract(mode: str, images: list[tuple[str, np.ndarray]], trues: list[dict]) -> list[dict[str, tuple[int, float]]]:
    os.environ["DOCUMIND_RECONSTRUCT"] = mode
    from ..inference.contracts import extract_contract_bytes
    out = []
    for i, ((name, img), gt) in enumerate(zip(images, trues), 1):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        res = extract_contract_bytes(buf.tobytes(), name)
        out.append(score_contract(res["row"], gt))
        print(f"  [{mode}] {i}/{len(images)} {name}", end="\r")
    print()
    return out


def run_mode(mode: str, images: list[tuple[str, np.ndarray]], trues: list[dict]) -> list[dict[str, tuple[int, float]]]:
    os.environ["DOCUMIND_RECONSTRUCT"] = mode
    from ..inference.extractor import FieldExtractor           # built AFTER the variable is set (it decides there whether the word step runs)
    ex = FieldExtractor()
    out = []
    for i, ((name, img), true) in enumerate(zip(images, trues), 1):
        res = ex.extract_image(img, 1)
        out.append(score(res.get("fields", []), true))
        print(f"  [{mode}] {i}/{len(images)} {name}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--photo", action="store_true")
    ap.add_argument("--type", default="invoice", choices=["invoice", "contract"])
    a = ap.parse_args()

    if a.type == "contract":
        samples = contract_samples(a.data)
        if not samples:
            raise SystemExit("No ground_truth.jsonl with images found in " + a.data)
        samples = select_samples(samples, None, a.n, a.seed)
    else:
        samples = [s for s in find_samples(a.data) if truth(s)]
        samples = select_samples(samples, None, a.n, a.seed)
        if not samples:
            raise SystemExit("No image with a transcribed annotation found in " + a.data)
    rng = np.random.default_rng(a.seed)
    images, trues = [], []
    for s in samples:
        img = read_image(str(s["image"]))
        images.append((Path(s["image"]).name, to_photo(img, rng) if a.photo else img))
        trues.append(s["truth"] if a.type == "contract" else truth(s))
    print(f"{len(images)} {a.type}s, {'phone-like photos' if a.photo else 'clean originals'}" + ("" if a.type == "contract" else "; invoice detector"))

    MODES = ("off", "auto", "on")
    if a.type == "contract":
        trues_cols = [score_contract({}, t).keys() for t in trues]
        results = {m: run_mode_contract(m, images, trues) for m in MODES}
        trues = [{c: 1 for c in cols} for cols in trues_cols]
    else:
        results = {m: run_mode(m, images, trues) for m in MODES}

    rows, per_label = [], defaultdict(lambda: {m: [] for m in MODES})
    for i, (name, _) in enumerate(images):
        for lab in trues[i]:
            row = {"image": name, "field": lab}
            for m in MODES:
                e, sim = results[m][i][lab]
                row[f"exact_{m}"], row[f"sim_{m}"] = e, round(sim, 3)
                per_label[lab][m].append((e, sim))
            rows.append(row)
    out = REPORTS_DIR / "compare_reconstruct.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def tot(m: str, k: str) -> float:
        idx = 0 if k == "exact" else 1
        return float(np.mean([t[idx] for v in per_label.values() for t in v[m]]))

    names = {"off": "off  (nothing)", "auto": "auto (picture only)", "on": "on   (picture + words)"}
    print(f"\n{len(rows)} true fields over {len(images)} images")
    print(f"  {'':<24}{'exact match':>12}{'similarity':>12}")
    for m in MODES:
        print(f"  {names[m]:<24}{tot(m, 'exact'):>12.1%}{tot(m, 'sim'):>12.3f}")
    print("\nfields where the three settings differ most (exact match off / auto / on):")
    spread = sorted(per_label.items(), key=lambda kv: -(max(np.mean([t[0] for t in kv[1][m]]) for m in MODES) - min(np.mean([t[0] for t in kv[1][m]]) for m in MODES)))[:15]
    for lab, v in spread:
        print(f"  {lab:<22} " + "  ".join(f"{np.mean([t[0] for t in v[m]]):6.1%}" for m in MODES) + f"   (n={len(v['off'])})")
    best = max(MODES, key=lambda m: (round(tot(m, "exact"), 4), round(tot(m, "sim"), 4), m == "off", m == "auto"))
    print(f"\nBest on these files: DOCUMIND_RECONSTRUCT={best}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
