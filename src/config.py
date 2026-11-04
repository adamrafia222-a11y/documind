"""Central configuration: folders and settings.

Every value can be overridden with an environment variable or a `.env` file
placed in the project root (see `.env.example`).
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader (KEY=VALUE per line), so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

# ------------------------------------------------------------------ folders
DATA_DIR = Path(os.getenv("DOCUMIND_DATA_DIR", BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"              # original datasets (FATURA...), never modified
INTERIM_DIR = DATA_DIR / "interim"      # OCR cache, can be deleted at any time
PROCESSED_DIR = DATA_DIR / "processed"  # corpus ready for training (OCR words + labels)
TOY_DIR = DATA_DIR / "toy"              # tiny fake dataset to test the pipeline
OCR_CACHE_DIR = INTERIM_DIR / "ocr_cache"
CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"

MODELS_DIR = Path(os.getenv("DOCUMIND_MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.getenv("DOCUMIND_REPORTS_DIR", BASE_DIR / "reports"))
MODEL_PATH = Path(os.getenv("DOCUMIND_MODEL_PATH", MODELS_DIR / "field_extractor.joblib"))

# ---------------------------------------------------------------------- OCR
# Windows example: C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
# FATURA is English -> "eng". For French documents use "fra" or "fra+eng"
# (the matching .traineddata files must be installed with Tesseract).
OCR_LANG = os.getenv("OCR_LANG", "eng")
# 6 = one uniform block of text, 11 = sparse text. Try both on your own documents.
OCR_PSM = int(os.getenv("OCR_PSM", "6"))

# ---------------------------------------------------------------- model / API
BACKGROUND_LABEL = "O"          # words that belong to no annotated field
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))  # below -> word is ignored
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
