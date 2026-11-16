"""Read a DATE box by voting between several readings of the same box.

A handwritten date is the field where one engine alone is least reliable: Tesseract reads print, not handwriting; the handwriting model can slip on a
digit; a different size of the same box gives a different Tesseract text. Each reading is turned into a real date (dates.parse_date: numeric or with a
month name, OCR slips repaired); the date that most readings agree on wins, the handwriting model counting most. Only when no reading is a date, the
best raw reading is returned (so the user still sees what was read).

    read_date_box(img, box, lang)  ->  {"text": the winning reading, "date": date | None, "readings": {engine: text}, "votes": {iso date: weight}}
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from PIL import Image

from ..ocr import handwriting, trocr
from .dates import parse_date
from .receipts import _tesseract_box

WEIGHTS = {"handwriting": 1.5, "handwriting_wide": 1.0, "trocr": 0.8, "page": 0.8, "tesseract_x1.0": 0.5, "tesseract_x0.7": 0.5, "tesseract_x1.4": 0.5}
FAMILY = {"tesseract_x1.0": "tesseract", "tesseract_x0.7": "tesseract", "tesseract_x1.4": "tesseract", "page": "tesseract", "handwriting": "handwriting", "handwriting_wide": "handwriting", "trocr": "trocr"}
CAP = {"tesseract": 1.0, "handwriting": 1.5, "trocr": 0.8}      # readings of one engine make the SAME mistake at 3 sizes ("27.05.2024" -> 27052074): together they are one vote, not three
PRIORITY = ["handwriting", "handwriting_wide", "trocr", "page", "tesseract_x1.0", "tesseract_x0.7", "tesseract_x1.4"]


def read_date_box(img: Image.Image, xyxy, lang: str | None = None, pad: int = 6, extra: dict | None = None) -> dict:
    """`extra` = readings made elsewhere (e.g. {"page": the text found on the whole page}) that vote too."""
    readings: dict[str, str] = {}
    for sc in (1.0, 0.7, 1.4):                                                  # handwriting is big: Tesseract reads it best at its own size, but sizes disagree
        readings[f"tesseract_x{sc}"] = _tesseract_box(img, xyxy, 7, pad, sc, lang, None)
    x0, y0, x1, y1 = xyxy
    crops = {"": img.crop((max(0, int(x0) - 3), max(0, int(y0) - 3), min(img.width, int(x1) + 4), min(img.height, int(y1) + 4)))}
    h = max(1.0, y1 - y0)
    crops["_wide"] = img.crop((max(0, int(x0 - 0.15 * h)), max(0, int(y0 - 0.25 * h)), min(img.width, int(x1 + 0.15 * h)), min(img.height, int(y1 + 0.25 * h))))
    engine = trocr.get_engine()
    if engine is not None:
        readings["trocr"] = handwriting.safe_read(engine, [crops[""]], "trocr")[0]
    hw = handwriting.get_engine()
    if hw is not None:
        out = handwriting.safe_read(hw, list(crops.values()))
        readings["handwriting"], readings["handwriting_wide"] = out[0], out[1]
    for k, v in (extra or {}).items():
        readings[k] = v
    votes: dict[date, float] = defaultdict(float)
    family: dict[tuple[date, str], float] = defaultdict(float)
    best_text: dict[date, tuple[int, str]] = {}
    for name, text in readings.items():
        d = parse_date(text or "")
        if d is None:
            continue
        fam = FAMILY.get(name, name)
        before = family[(d, fam)]
        family[(d, fam)] = min(CAP.get(fam, 1.0), before + WEIGHTS.get(name, 0.5))
        votes[d] += family[(d, fam)] - before                                                        # only what the family adds, up to its cap
        rank = PRIORITY.index(name) if name in PRIORITY else len(PRIORITY)
        if d not in best_text or rank < best_text[d][0]:
            best_text[d] = (rank, text.strip())
    for d in votes:                                                                                  # priors: a plausible year (small bonus, only decides between competing readings)
        votes[d] += (0.2 if 1990 <= d.year <= 2100 else 0.0) + (0.3 if abs(d.year - date.today().year) <= 3 else 0.0)
    if votes:
        win = max(votes, key=lambda d: (votes[d], -best_text[d][0]))
        return {"text": best_text[win][1], "date": win, "readings": readings, "votes": {d.isoformat(): round(w, 2) for d, w in votes.items()}}
    pick = next((readings[k] for k in PRIORITY if readings.get(k, "").strip()), "")
    return {"text": pick.strip(), "date": None, "readings": readings, "votes": {}}
