"""STEP 3 (OCR) - Tesseract against the trained model, on crops neither of them was tuned on (the `test` crops of prepare_ocr).

    python -m documind.training.evaluate_ocr                       # 800 test crops per source, both engines
    python -m documind.training.evaluate_ocr --n 3000 --beams 4    # more crops (Tesseract reads about 8 crops per second)
    python -m documind.training.evaluate_ocr --engines trocr       # only the trained model

CER = character error rate: edits needed to turn the reading into the truth / characters of the truth (0 = perfect, 0.05 = 5 characters in 100 wrong).
exact = share of crops read without a single mistake. The table is per source and per field type, so you see WHERE the new model helps (accents,
capitals, handwriting) and where it does not. Everything is written to reports/ocr_report.json.

Careful: the crops come from the same generators as the training crops (and FATURA is one dataset of 50 templates). A very good score says the model
learned these documents, not that it reads every real document. Try real pictures (python -m documind.inference.receipts photo.jpg) before trusting it.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ..config import REPORTS_DIR
from ..data.prepare_ocr import DEFAULT_OUT
from ..ocr.metrics import cer, summarize
from .train_ocr import read_labels

REPORT = REPORTS_DIR / "ocr_report.json"


def sample_rows(root: Path, split: str, n: int, seed: int = 0) -> list[dict]:
    rows = read_labels(root, split)
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    out = []
    for lst in by.values():
        rng.shuffle(lst)
        out += lst[:n]
    return out


def read_all(engine: str, rows: list[dict], model_dir: Path, beams: int) -> list[str]:
    if engine == "tesseract":
        from ..ocr.baseline import read_tesseract
        hyps = []
        for i, r in enumerate(rows, 1):
            hyps.append(read_tesseract(Image.open(r["path"]).convert("RGB")))
            if i % 200 == 0:
                print(f"  tesseract {i}/{len(rows)}", flush=True)
        return hyps
    from ..ocr.trocr import TrOCRRecognizer
    rec = TrOCRRecognizer(model_dir, num_beams=beams)
    hyps = []
    for i in range(0, len(rows), 64):
        hyps += rec.read([Image.open(r["path"]).convert("RGB") for r in rows[i:i + 64]], batch_size=64)
        print(f"  trocr {min(i + 64, len(rows))}/{len(rows)}", flush=True)
    return hyps


def table(rows: list[dict], hyps: dict[str, list[str]]) -> dict:
    res: dict = {}
    for eng, hs in hyps.items():
        res[eng] = {"all": summarize([(r["text"], h) for r, h in zip(rows, hs)]), "sources": {}}
        for s in sorted({r["source"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["source"] == s]
            res[eng]["sources"][s] = {"all": summarize([(rows[i]["text"], hs[i]) for i in idx]),
                                      "kinds": {k: summarize([(rows[i]["text"], hs[i]) for i in idx if rows[i]["kind"] == k]) for k in sorted({rows[i]["kind"] for i in idx})}}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=None, help="trained model folder (default: models/ocr_trocr)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=800, help="crops per source")
    ap.add_argument("--engines", default="tesseract,trocr")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--worst", type=int, default=12, help="how many of the worst readings of the trained model to print")
    a = ap.parse_args()
    from ..ocr.trocr import MODEL_DIR
    model_dir = Path(a.model) if a.model else MODEL_DIR
    rows = sample_rows(Path(a.data), a.split, a.n)
    if not rows:
        raise SystemExit(f"No {a.split} crops in {a.data}. Run first: python -m documind.data.prepare_ocr")
    engines = [e for e in a.engines.split(",") if e]
    hyps = {}
    for e in engines:
        t0 = time.time()
        print(f"reading {len(rows)} crops with {e} ...", flush=True)
        hyps[e] = read_all(e, rows, model_dir, a.beams)
        print(f"  done in {time.time() - t0:.0f} s")
    res = table(rows, hyps)
    sources = sorted({r["source"] for r in rows})
    print(f"\n{'':26s}" + "".join(f"{e:>24s}" for e in engines) + "      (CER / exact)")
    for s in sources:
        print(f"{s.upper():26s}" + "".join(f"{res[e]['sources'][s]['all']['cer']:>14.4f} / {res[e]['sources'][s]['all']['exact']:.2f}" for e in engines))
        for k in sorted({r["kind"] for r in rows if r["source"] == s}):
            print(f"   {k:23s}" + "".join(f"{res[e]['sources'][s]['kinds'][k]['cer']:>14.4f} / {res[e]['sources'][s]['kinds'][k]['exact']:.2f}" for e in engines))
    print(f"{'ALL':26s}" + "".join(f"{res[e]['all']['cer']:>14.4f} / {res[e]['all']['exact']:.2f}" for e in engines))
    if "trocr" in hyps:
        bad = sorted(((cer(r["text"], h), r, h) for r, h in zip(rows, hyps["trocr"])), key=lambda t: -t[0])[:a.worst]
        print(f"\nworst readings of the trained model:")
        for c, r, h in bad:
            print(f"  CER {c:.2f}  [{r['source']}/{r['kind']}]  truth: {r['text'][:60]!r}   read: {h[:60]!r}")
        res["worst"] = [{"cer": round(c, 3), "source": r["source"], "kind": r["kind"], "truth": r["text"], "read": h} for c, r, h in bad]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
