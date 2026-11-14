"""STEP 1 (contracts) - clean the YOLO contract dataset: data/raw/contracts  ->  data/processed/contracts_yolo

    python scripts/generate_contracts.py --n 10000 --out data/raw/contracts             # make the dataset (19 classes, one A4 page per contract)
    python -m documind.data.prepare_contracts                                           # default folders (data/raw/contracts, else data/raw/contract_dataset)
    python -m documind.data.prepare_contracts --data contract_dataset --workers 4

Same checks as the receipts and cheques (the shared code of documind.data.prepare_receipts): pictures paired with their label file, every label line
validated, duplicates removed, EXIF rotation applied, the same picture found in two splits, YOLO folders + data.yaml, reports/contracts_prepare_report.txt,
reports/contracts_stats.json and reports/contracts_preview/*.png (pictures with their boxes drawn: LOOK AT THEM).

Raw layouts understood: <data>/images/<split> + <data>/labels/<split> (what generate_contracts.py writes) or <data>/<split>/{images,labels}.
Class names: --names, else the `names` of <data>/data.yaml, else the 19 classes of generate_contracts.py.
A contract is a full A4 page (1240 x 1754 before the scan effects) with 20 px text: --max-side defaults to 1800 so that nothing is shrunk.
Several boxes of the same class on one page are NORMAL here: the lender's name is printed twice and a value that wraps over two lines has one box per line.
"""
from __future__ import annotations

from pathlib import Path

from ..config import PROCESSED_DIR, RAW_DIR
from .prepare_receipts import has_pictures_in, main as _main

CONTRACT_NAMES = ["title", "contract_number", "contract_date", "lender_name", "borrower_name", "borrower_id", "borrower_address", "amount", "amount_words",
                  "interest_rate", "duration", "monthly_payment", "first_payment_date", "iban", "place", "signing_date", "signature_lender", "signature_borrower", "stamp"]


def find_raw_contracts(raw_dir: Path = RAW_DIR) -> Path:
    """First usual folder name that contains a dataset (either raw layout)."""
    for name in ("contracts", "contract_dataset"):
        if has_pictures_in(raw_dir / name):
            return raw_dir / name
    return raw_dir / "contracts"                                 # nothing found: the error message will name this path


DEFAULT_SRC = find_raw_contracts()
DEFAULT_DST = PROCESSED_DIR / "contracts_yolo"


def main() -> None:
    _main(tag="contracts", default_src=DEFAULT_SRC, default_dst=DEFAULT_DST, default_names=CONTRACT_NAMES, default_max_side=1800, doc=__doc__)


if __name__ == "__main__":
    main()
