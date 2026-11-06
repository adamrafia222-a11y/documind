"""Command line:  python -m documind.inference.cli invoice1.jpg scan.pdf -o result.csv"""
from __future__ import annotations

import argparse
from pathlib import Path

from .extractor import FieldExtractor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", default="extraction.csv")
    args = ap.parse_args()
    ex = FieldExtractor()
    results = []
    for f in args.files:
        p = Path(f)
        print(f"-> {p.name}")
        results.append(ex.extract_bytes(p.read_bytes(), p.name))
    Path(args.output).write_bytes(ex.to_csv(results))
    print(f"CSV written: {args.output}")


if __name__ == "__main__":
    main()
