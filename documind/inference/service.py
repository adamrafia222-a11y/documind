"""One entry point for every document type: decide the type, run the right extractor, build ONE CSV PER TYPE.

    svc = DocumentService()
    result = svc.extract(data, "scan.jpg")                 # doc_type="auto" (router) | "invoice" | "receipt"
    svc.csv_files([result1, result2, ...])                 # {"invoices.csv": b"...", "receipts.csv": b"..."}  never mixed

To add a new document type: write an extractor that returns the same kind of dict (file, fields, items, row, columns, doc_type),
add it to `EXTRACTORS`, and give it a CSV name in `CSV_NAMES`. The API and the web page need no other change.
"""
from __future__ import annotations

from pathlib import Path

from ..config import MODEL_PATH
from .export import rows_to_csv
from .extractor import FieldExtractor, ModelNotFound
from .router import TYPES, detect_type

CSV_NAMES = {"invoice": "invoices.csv", "receipt": "receipts.csv"}
LABELS = {"invoice": "Invoice", "receipt": "Receipt"}


class DocumentService:
    def __init__(self) -> None:
        self._invoice: FieldExtractor | None = None

    # ------------------------------------------------------------------ what is installed
    def available(self) -> dict[str, bool]:
        from . import receipts
        return {"invoice": Path(MODEL_PATH).exists(), "receipt": receipts.model_available()}

    @property
    def invoice_loaded(self) -> bool:
        return self._invoice is not None

    def invoice(self) -> FieldExtractor:
        if self._invoice is None:
            self._invoice = FieldExtractor()                      # raises ModelNotFound with the command to train it
        return self._invoice

    def columns(self, doc_type: str) -> list[str]:
        if doc_type == "receipt":
            from .receipts import RECEIPT_COLUMNS
            return list(RECEIPT_COLUMNS)
        return self.invoice().columns

    # ------------------------------------------------------------------ extraction
    def extract(self, data: bytes, filename: str = "document", doc_type: str = "auto") -> dict:
        av = self.available()
        routing = None
        if doc_type == "auto":
            if not any(av.values()):
                raise ModelNotFound("No model is installed. Train the invoice model (python -m documind.training.train) and/or the receipt model "
                                    "(python -m documind.training.train_receipts).")
            routing = detect_type(data, filename, [t for t in TYPES if av[t]])
            doc_type = routing["doc_type"]
        if doc_type not in TYPES:
            raise ValueError(f"Unknown document type '{doc_type}' (use auto, invoice or receipt).")
        if not av[doc_type]:
            hint = "python -m documind.training.train" if doc_type == "invoice" else "python -m documind.training.train_receipts"
            raise ModelNotFound(f"The {doc_type} model is not installed. Train it first:  {hint}")
        if doc_type == "receipt":
            from . import receipts
            try:
                result = receipts.extract_receipt_bytes(data, filename)
            except SystemExit as exc:                              # load_model() reports a missing Ultralytics package this way
                raise ModelNotFound(str(exc))
        else:
            ex = self.invoice()
            result = ex.extract_bytes(data, filename)
            result["columns"] = ex.columns
        result["doc_type"] = doc_type
        result["doc_type_source"] = "auto" if routing else "manual"
        if routing:
            result["routing"] = routing
        return result

    # ------------------------------------------------------------------ CSV: one file per type
    def to_csv(self, doc_type: str, results: list[dict]) -> bytes:
        cols = results[0].get("columns") if results else None
        return rows_to_csv([r["row"] for r in results], cols or self.columns(doc_type))

    def csv_files(self, results: list[dict]) -> dict[str, bytes]:
        """Results of any mix of types -> {file name: csv bytes}, one file per type that is present."""
        groups: dict[str, list[dict]] = {}
        for r in results:
            groups.setdefault(r.get("doc_type", "invoice"), []).append(r)
        return {CSV_NAMES.get(t, f"{t}.csv"): self.to_csv(t, rs) for t, rs in groups.items()}
