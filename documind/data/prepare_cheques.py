"""STEP 1 (cheques) - clean the YOLO cheque dataset: data/raw/cheques  ->  data/processed/cheques_yolo

    python scripts/generate_cheques.py --n 10000 --out data/raw/cheques             # make the dataset (10 classes)
    python -m documind.data.prepare_cheques                                          # default folders (data/raw/cheques, else data/raw/cheque_dataset)
    python -m documind.data.prepare_cheques --data cheque_dataset --workers 4

The checks are exactly those of the receipts (same code, documind.data.prepare_receipts): pictures paired with their label file, every label
line validated (class id, numbers in 0..1, box inside the picture), duplicate lines removed, EXIF rotation applied, the same picture found
in two splits (leakage), YOLO folders + data.yaml written, and reports/cheques_prepare_report.txt, reports/cheques_stats.json and
reports/cheques_preview/*.png (pictures with their boxes drawn: LOOK AT THEM).

Raw layouts understood: <data>/images/<split> + <data>/labels/<split> (what generate_cheques.py writes) or <data>/<split>/{images,labels}.
Class names: --names, else the `names` of <data>/data.yaml, else the 10 classes of generate_cheques.py:
    bank_name, cheque_number, amount_words, payee, amount_digits, place, date, signature, account_number, micr_line.
A cheque is wide (about 1600 x 700): --max-side defaults to 1600 (not 1100), because the handwriting and the thin MICR line need the pixels.
"""
from __future__ import annotations

from pathlib import Path

from ..config import PROCESSED_DIR, RAW_DIR
from .prepare_receipts import has_pictures_in, main as _main

CHEQUE_NAMES = ["bank_name", "cheque_number", "amount_words", "payee", "amount_digits", "place", "date", "signature", "account_number", "micr_line"]


def find_raw_cheques(raw_dir: Path = RAW_DIR) -> Path:
    """First usual folder name that contains a dataset (either raw layout)."""
    for name in ("cheques", "cheque_dataset"):
        if has_pictures_in(raw_dir / name):
            return raw_dir / name
    return raw_dir / "cheques"                                   # nothing found: the error message will name this path


DEFAULT_SRC = find_raw_cheques()
DEFAULT_DST = PROCESSED_DIR / "cheques_yolo"


def main() -> None:
    _main(tag="cheques", default_src=DEFAULT_SRC, default_dst=DEFAULT_DST, default_names=CHEQUE_NAMES, default_max_side=1600, doc=__doc__)


if __name__ == "__main__":
    main()
