"""STEP 2 (invoices) - train the YOLO invoice detector on data/processed/invoices_yolo  ->  models/invoices_yolo.pt

    python -m documind.training.train_invoices                                  # yolo11s, 60 epochs, imgsz 1280, target 0.90
    python -m documind.training.train_invoices --imgsz 1536 --batch 2           # more pixels for the small text (slower)
    python -m documind.training.train_invoices --model yolo11m.pt --batch 2     # bigger / more precise (slower)
    python -m documind.training.train_invoices --fraction 0.2 --epochs 10       # quick test run
    python -m documind.training.train_invoices --eval-only                      # re-score models/invoices_yolo.pt on val and test

Same code as the receipts, cheques and contracts (documind.training.train_receipts): same score tables (precision / recall / mAP50 / found_rate per class on
val AND test, reports/invoices_report.json), same target check (--target 0.90), same requirements (requirements.txt). Curves: reports/invoices_runs/<name>/.

What is different:
  * the page is a full A4 (up to 2400 px) with 10-14 px text: imgsz 1280 (batch 4 fits an 8 GB GPU); if the small classes are weak, raise imgsz to 1536 and lower the batch to 2
  * no mosaic (it would shrink four pages into one), no flips (a mirrored invoice is not an invoice), a small rotation only
  * val and test are layouts NEVER seen in training when prepare_invoices used the default --split template: that score is the honest one for new invoices
"""
from __future__ import annotations

from ..config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .train_receipts import Profile, run_cli

DATA_YAML = PROCESSED_DIR / "invoices_yolo" / "data.yaml"
RUNS_DIR = REPORTS_DIR / "invoices_runs"
MODEL_OUT = MODELS_DIR / "invoices_yolo.pt"
REPORT_PATH = REPORTS_DIR / "invoices_report.json"

INVOICE_HINTS = (
    "  1. look at the per-class rows above: small values (NUMBER, DATE, TAX) and long blocks (SELLER_ADDRESS, BUYER, NOTE) are the usual weak spots",
    "  2. more pixels: --imgsz 1536 --batch 2   (and prepare_invoices --max-side 2400, the default, so nothing was shrunk before)",
    "  3. all epochs:  --epochs 100 --patience 100, a bigger model:  --model yolo11m.pt --batch 2",
    "  4. val/test are unseen LAYOUTS (--split template): a low score there with a high train score means the 40 or so training templates are too few: add real invoices",
    "  5. look at reports/invoices_runs/<name>/ (curves, confusion matrix) and the pictures of reports/invoices_preview")
INVOICE_REACHED = ("Careful: FATURA is generated from a few dozen templates. A high score on val/test proves the model learned these layouts "
                   "(unseen layouts if the split was by template). Try your own invoices (python -m documind.inference.invoice_yolo invoice.jpg) before trusting it.")


def profile() -> Profile:
    return Profile(kind="invoices", data_yaml=DATA_YAML, runs_dir=RUNS_DIR, model_out=MODEL_OUT, report_path=REPORT_PATH, epochs=60, imgsz=1280, batch=4, patience=15,
                   prepare_cmd="python -m documind.data.prepare_invoices --fatura data/raw/FATURA",
                   imgsz_help="an invoice page is up to 2400 px tall with 10-14 px text: 1280 (1536 for more detail, batch 2)",
                   aug=dict(fliplr=0.0, flipud=0.0, mosaic=0.0, close_mosaic=0, degrees=2.0, translate=0.04, scale=0.25, shear=0.5, perspective=0.0002,
                            hsv_h=0.01, hsv_s=0.3, hsv_v=0.4),
                   hints=INVOICE_HINTS, reached_note=INVOICE_REACHED, doc=__doc__)


def main() -> None:
    run_cli(profile())


if __name__ == "__main__":
    main()
