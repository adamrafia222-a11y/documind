"""STEP 2 (receipts) - train the YOLO receipt detector on data/processed/receipts_yolo  ->  models/receipts_yolo.pt

    python -m documind.training.train_receipts                                  # yolo11s, 50 epochs, imgsz 960
    python -m documind.training.train_receipts --model yolo11n.pt --epochs 40   # smaller / faster
    python -m documind.training.train_receipts --eval-only                      # re-score models/receipts_yolo.pt on val and test

Needs the extra packages of requirements-receipts.txt (Ultralytics + PyTorch). Use a GPU (Colab / Kaggle, or your own);
on a CPU it works but takes hours. Results: reports/receipts_runs/<name>/ (curves, confusion matrix), reports/receipts_report.json
(precision / recall / mAP per class on val AND test), models/receipts_yolo.pt (the best weights).

Speed: the validation score of the first real run stopped improving after about 45 epochs, so 50 epochs with an early stop after 10
epochs without progress is enough. If an epoch is slow, open Task Manager > Performance > GPU while it runs: below about 70 %
the GPU waits for the data loader: raise --workers, and run STEP 1 with --max-side 1100 (smaller pictures decode faster).
--fraction 0.3 trains on 30 % of the pictures: use it to compare settings (imgsz, model) in minutes before a full run.

The augmentation is chosen for receipts: no left-right flip (mirrored text is not a receipt), small rotation / perspective
(phone pictures), brightness changes, mosaic switched off for the last epochs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from ..config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .receipts_eval import found_rates

DATA_YAML = PROCESSED_DIR / "receipts_yolo" / "data.yaml"
RUNS_DIR = REPORTS_DIR / "receipts_runs"
MODEL_OUT = MODELS_DIR / "receipts_yolo.pt"
REPORT_PATH = REPORTS_DIR / "receipts_report.json"


def _yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics is not installed in this environment. Run:  pip install -r requirements-receipts.txt")
    return YOLO


def default_workers() -> int:
    return max(2, min(8, os.cpu_count() or 2))


def train(data: Path = DATA_YAML, model: str = "yolo11s.pt", epochs: int = 50, imgsz: int = 960, batch: int = 8, patience: int = 10,
          name: str = "receipts", device=None, workers: int | None = None, cache=False, fraction: float = 1.0, time: float | None = None) -> Path:
    """Train and return the path of the best weights."""
    YOLO = _yolo()
    if not Path(data).exists():
        raise SystemExit(f"{data} not found. Run STEP 1 first:  python -m documind.data.prepare_receipts")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(model)
    yolo.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, patience=patience, device=device,
        workers=workers if workers is not None else default_workers(), cache=cache, fraction=fraction, time=time,
        project=str(RUNS_DIR), name=name, exist_ok=True, seed=0, cos_lr=True,
        fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=15,
        degrees=4.0, translate=0.08, scale=0.35, shear=2.0, perspective=0.0005,
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
    )
    best = RUNS_DIR / name / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODEL_OUT)
    return MODEL_OUT


def evaluate(weights: Path = MODEL_OUT, data: Path = DATA_YAML, imgsz: int = 960, device=None) -> list[dict]:
    """Score on val and test. `test` was never used for training or for choosing the best epoch: it is the honest number."""
    YOLO = _yolo()
    yolo = YOLO(str(weights))
    wanted = {s for s in ("val", "test") if f"\n{s}:" in "\n" + Path(data).read_text(encoding="utf-8")}
    rows = []
    for split in ("val", "test"):
        if split not in wanted:
            continue
        m = yolo.val(data=str(data), split=split, imgsz=imgsz, device=device, plots=False, verbose=False)
        found = found_rates(yolo, Path(data), split, imgsz, device)             # "did it find the field?" (mAP also punishes missing labels)
        rates = [v["found_rate"] for v in found.values() if v.get("found_rate") is not None]
        rows.append({"split": split, "class": "ALL", "precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                     "found_rate": round(sum(rates) / len(rates), 4) if rates else None})
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            rows.append({"split": split, "class": yolo.names[int(c)], "precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50-95": float(ap),
                         "found_rate": found.get(yolo.names[int(c)], {}).get("found_rate")})
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"weights": str(weights), "imgsz": imgsz, "rows": rows}, indent=2), encoding="utf-8")
    return rows


def print_rows(rows: list[dict]) -> None:
    print(f"{'split':6s}{'class':14s}{'precision':>10s}{'recall':>8s}{'mAP50':>8s}{'mAP50-95':>10s}{'found_rate':>12s}")
    for r in rows:
        fr = r.get("found_rate")
        print(f"{r['split']:6s}{r['class']:14s}{r['precision']:10.3f}{r['recall']:8.3f}{r['mAP50']:8.3f}{r['mAP50-95']:10.3f}{(f'{fr:12.3f}' if fr is not None else '           -')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt (fastest) / yolo11s.pt / yolo11m.pt (most accurate). Newer families work the same way.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960, help="receipts are tall and the boxes are thin: 960 or 1280 beats the default 640")
    ap.add_argument("--batch", type=int, default=8, help="lower it on 'CUDA out of memory'; -1 lets Ultralytics choose")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--name", default="receipts")
    ap.add_argument("--device", default=None, help="0 (first GPU), cpu ... default: automatic")
    ap.add_argument("--workers", type=int, default=None, help=f"data-loader processes (default: {default_workers()} here)")
    ap.add_argument("--cache", default="False", choices=["False", "ram", "disk"], help="keep decoded pictures in RAM / on disk: faster, needs 8-10 GB")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training pictures to use (0.3 = quick comparison run)")
    ap.add_argument("--time", type=float, default=None, help="training budget in HOURS (overrides --epochs)")
    ap.add_argument("--eval-only", action="store_true", help="skip training, score models/receipts_yolo.pt")
    a = ap.parse_args()
    if not a.eval_only:
        w = train(Path(a.data), a.model, a.epochs, a.imgsz, a.batch, a.patience, a.name, a.device, a.workers,
                  False if a.cache == "False" else a.cache, a.fraction, a.time)
        print(f"\nBest weights saved: {w}")
    print_rows(evaluate(MODEL_OUT, Path(a.data), a.imgsz, a.device))
    print(f"\nReport: {REPORT_PATH}   curves: {RUNS_DIR / a.name}")


if __name__ == "__main__":
    main()
