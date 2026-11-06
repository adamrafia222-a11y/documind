"""CSV helpers."""
from __future__ import annotations

import csv
import io

from ..config import BACKGROUND_LABEL


def fields_to_row(fields: list[dict]) -> dict[str, str]:
    """One document -> one row. Repeated fields (e.g. several addresses) are joined with ' | '."""
    row: dict[str, list[str]] = {}
    for f in sorted(fields, key=lambda f: (f.get("page", 1), f["bbox"][1], f["bbox"][0])):
        row.setdefault(f["label"], []).append(f["text"])
    return {k: " | ".join(v) for k, v in row.items()}


def rows_to_csv(rows: list[dict], columns: list[str]) -> bytes:
    """UTF-8 with BOM so Excel opens accents correctly."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue().encode("utf-8-sig")


def csv_columns(classes: list[str]) -> list[str]:
    return ["file"] + [c for c in classes if c != BACKGROUND_LABEL] + ["N_ITEMS", "ITEMS"]
