"""STEP 2 (contracts) - train the YOLO contract detector on data/processed/contracts_yolo  ->  models/contracts_yolo.pt

    python -m documind.training.train_contracts                                 # yolo11s, 60 epochs, imgsz 1280, target 0.90
    python -m documind.training.train_contracts --imgsz 1536 --batch 2          # more pixels for the small text (slower)
    python -m documind.training.train_contracts --model yolo11m.pt --batch 2    # bigger / more precise (slower)
    python -m documind.training.train_contracts --fraction 0.2 --epochs 10      # quick test run
    python -m documind.training.train_contracts --eval-only                     # re-score models/contracts_yolo.pt on val and test

Same code as the receipts and cheques (documind.training.train_receipts): same score tables (precision / recall / mAP50 / found_rate per class on val AND
test, reports/contracts_report.json), same target check (--target 0.90), same requirements (requirements-receipts.txt). Curves: reports/contracts_runs/<name>/.

What is different:
  * the page is a full A4 (up to 1800 px) with 20 px text and values that are only a few words long: imgsz 1280 (batch 4 fits an 8 GB GPU); if the small
    classes (borrower_id, interest_rate, duration) are weak, raise imgsz to 1536 and lower the batch to 2
  * no mosaic: it would shrink four pages into one and make the small text even smaller; no flips (a mirrored page is not a contract); a small rotation
    only (scans are tilted by a degree or two)
  * several boxes of one class per page are normal (the lender's name twice, a value wrapped over two lines): the "found rate" of such classes is the share
    of labelled boxes found, not "is the best box right"
"""
from __future__ import annotations

from ..config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .train_receipts import Profile, run_cli

DATA_YAML = PROCESSED_DIR / "contracts_yolo" / "data.yaml"
RUNS_DIR = REPORTS_DIR / "contracts_runs"
MODEL_OUT = MODELS_DIR / "contracts_yolo.pt"
REPORT_PATH = REPORTS_DIR / "contracts_report.json"

CONTRACT_HINTS = (
    "  1. look at the per-class rows above: the small values (borrower_id, interest_rate, duration) and the wrapped ones (iban, amount_words) are the usual weak spots",
    "  2. more pixels: --imgsz 1536 --batch 2   (and prepare_contracts --max-side 1800, the default, so nothing was shrunk before)",
    "  3. all epochs:  --epochs 100 --patience 100, a bigger model:  --model yolo11m.pt --batch 2, more pages:  python scripts/generate_contracts.py --n 20000 --seed 1",
    "  4. look at reports/contracts_runs/<name>/ (curves, confusion matrix) and the pictures of reports/contracts_preview")
CONTRACT_REACHED = ("Careful: val and test come from the same generator as train, so this proves the model learned the generated layouts. "
                    "Try real contracts (python -m documind.inference.contracts contract.jpg) before trusting it.")


def profile() -> Profile:
    return Profile(kind="contracts", data_yaml=DATA_YAML, runs_dir=RUNS_DIR, model_out=MODEL_OUT, report_path=REPORT_PATH, epochs=60, imgsz=1280, batch=4, patience=15,
                   prepare_cmd="python -m documind.data.prepare_contracts", imgsz_help="a contract page is 1800 px tall with 20 px text: 1280 (1536 for more detail, batch 2)",
                   aug=dict(fliplr=0.0, flipud=0.0, mosaic=0.0, close_mosaic=0, degrees=2.0, translate=0.04, scale=0.25, shear=0.5, perspective=0.0002,
                            hsv_h=0.01, hsv_s=0.3, hsv_v=0.4),
                   hints=CONTRACT_HINTS, reached_note=CONTRACT_REACHED, doc=__doc__)


def main() -> None:
    run_cli(profile())


if __name__ == "__main__":
    main()
