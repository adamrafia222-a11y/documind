"""STEP 2 (cheques) - train the YOLO cheque detector on data/processed/cheques_yolo  ->  models/cheques_yolo.pt

    python -m documind.training.train_cheques                                   # yolo11s, 60 epochs, imgsz 1280, target 0.90
    python -m documind.training.train_cheques --model yolo11m.pt --batch 2      # bigger / more precise (slower)
    python -m documind.training.train_cheques --fraction 0.2 --epochs 10        # quick test run
    python -m documind.training.train_cheques --eval-only                       # re-score models/cheques_yolo.pt on val and test

Same code as the receipts (documind.training.train_receipts): same score tables (precision / recall / mAP50 / found_rate per class on val AND test,
reports/cheques_report.json), same target check (--target 0.90, printed whether it is reached or not), same requirements (requirements-receipts.txt:
Ultralytics + PyTorch, a GPU is strongly advised). Curves and confusion matrix: reports/cheques_runs/<name>/.

What is different from the receipts:
  * imgsz 1280 with batch 4 (fits an 8 GB GPU): a cheque is wide (1600 x 700) and the handwriting / MICR line are small; the prepared pictures are at most 1600 px wide
  * 60 epochs with an early stop after 15 epochs without progress: the 10 classes are large boxes at fixed places and are learned quickly
    (use --epochs 100 --patience 100 to run all epochs)
  * lighter mosaic (0.5): a cheque is read as ONE page, the printed labels and the fields keep their relative places; small rotation only
    (scans), no left-right / up-down flip (a mirrored cheque is not a cheque)
"""
from __future__ import annotations

from ..config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .train_receipts import Profile, run_cli

DATA_YAML = PROCESSED_DIR / "cheques_yolo" / "data.yaml"
RUNS_DIR = REPORTS_DIR / "cheques_runs"
MODEL_OUT = MODELS_DIR / "cheques_yolo.pt"
REPORT_PATH = REPORTS_DIR / "cheques_report.json"

CHEQUE_HINTS = (
    "  1. look at the per-class rows above: signature and amount_words / payee (long handwriting) are usually the weakest",
    "  2. a bigger model:  --model yolo11m.pt --batch 2   (4 is the default batch: it fits 8 GB of GPU memory at imgsz 1280 with yolo11s)",
    "  3. all epochs:  --epochs 100 --patience 100, and more pictures:  python scripts/generate_cheques.py --n 20000 --seed 1 (labels are exact)",
    "  4. look at reports/cheques_runs/<name>/ (curves, confusion matrix) and the pictures of reports/cheques_preview")
CHEQUE_REACHED = ("Careful: val and test come from the same generator as train, so this proves the model learned the generated layouts. "
                  "Try real cheques (python -m documind.inference.cheques cheque.jpg) before trusting it.")


def profile() -> Profile:
    return Profile(kind="cheques", data_yaml=DATA_YAML, runs_dir=RUNS_DIR, model_out=MODEL_OUT, report_path=REPORT_PATH, epochs=60, imgsz=1280, batch=4, patience=15,
                   prepare_cmd="python -m documind.data.prepare_cheques", imgsz_help="cheques are wide (1600 px): 1280 keeps the handwriting readable",
                   aug=dict(fliplr=0.0, flipud=0.0, mosaic=0.5, close_mosaic=10, degrees=3.0, translate=0.06, scale=0.3, shear=1.0, perspective=0.0003,
                            hsv_h=0.01, hsv_s=0.4, hsv_v=0.4),
                   hints=CHEQUE_HINTS, reached_note=CHEQUE_REACHED, doc=__doc__)


def main() -> None:
    run_cli(profile())


if __name__ == "__main__":
    main()
