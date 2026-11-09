#!/usr/bin/env python3
"""prepare_receipts.py - same as  python -m documind.data.prepare_receipts  (one implementation, no copy to keep in sync).

    python scripts/prepare_receipts.py --data data/raw/receipts --out data/processed/receipts_yolo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from documind.data.prepare_receipts import main                                                      # noqa: E402

if __name__ == "__main__":
    main()
