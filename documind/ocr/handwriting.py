"""Handwritten text. Tesseract reads PRINT; on handwriting (the amount and payee of a cheque, a filled-in line of a contract, a note on an invoice) it
returns garbage or nothing, and the models/ocr_trocr recognizer is trained on printed crops too. This module adds a recognizer made for handwriting
(Microsoft TrOCR "handwritten", pretrained: no training needed) and decides WHEN to use it:

  * a box (cheque / contract / receipt field)     -> receipts.ocr_box reads it with Tesseract, measures how sure Tesseract was, and asks this recognizer when
                                                     the field is handwritten by nature (cheque payee, amount in words, place) or Tesseract was unsure or its
                                                     text does not look like the field. documind.ocr.ensemble then keeps the better reading.
  * a full page (invoice / contract text)        -> `rescue(...)` finds the lines where Tesseract returned only low-confidence junk and re-reads those lines.

    DOCUMIND_HANDWRITING=auto   (default) use it when torch + transformers are installed and the model can be loaded (the first use downloads ~1.3 GB
                                from Hugging Face, then it is cached); otherwise a warning, and Tesseract alone
    DOCUMIND_HANDWRITING=on     the same, but an error when it cannot be loaded      off   never
    HANDWRITING_MODEL           microsoft/trocr-base-handwritten (default; best)  |  microsoft/trocr-small-handwritten (3x faster)  |  a folder you fine-tuned
                                (python -m documind.training.train_ocr --base microsoft/trocr-base-handwritten on your own cheque crops is the way to improve it)
    HANDWRITING_LOW_CONF        Tesseract confidence below which a box / a word counts as "not print" (default 60)
DOCUMIND_OCR=tesseract also switches it off (that setting means "Tesseract only").
"""
from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
from PIL import Image, ImageOps

from ..config import OCR_LANG, TESSERACT_CMD

MODEL = os.getenv("HANDWRITING_MODEL", "microsoft/trocr-base-handwritten")
LOW_CONF = float(os.getenv("HANDWRITING_LOW_CONF", "60"))
ALWAYS = {"payee", "amount_words", "place", "date", "amount_digits", "signing_date"}      # cheque / contract values that are handwritten by nature: always give the recognizer a try
_engine = None
_tried = False
_error = ""
_runtime_errors: dict[str, str] = {}                 # engine label -> last error while reading ("" once it works again)


def mode() -> str:
    m = os.getenv("DOCUMIND_HANDWRITING", "auto").lower()
    return "off" if m in ("off", "0", "false", "no") or os.getenv("DOCUMIND_OCR", "auto").lower() == "tesseract" else "on" if m in ("on", "1", "true", "yes") else "auto"


def get_engine():
    """The shared handwriting recognizer, or None (-> Tesseract only). Loaded once, on first use."""
    global _engine, _tried, _error
    m = mode()
    if m == "off":
        return None
    if _engine is None and (not _tried or m == "on"):
        _tried = True
        try:
            from .trocr import TrOCRRecognizer
            _engine = TrOCRRecognizer(MODEL, num_beams=int(os.getenv("HANDWRITING_BEAMS", "3")))
        except Exception as exc:                              # no torch / no transformers / offline and not downloaded yet
            _error = f"{type(exc).__name__}: {str(exc)[:140]}"
            if m == "on":
                raise SystemExit(f"DOCUMIND_HANDWRITING=on but {MODEL} could not be loaded ({type(exc).__name__}: {exc}). pip install -r requirements.txt, and run once with internet access.")
            warnings.warn(f"Handwriting recognizer '{MODEL}' could not be loaded ({type(exc).__name__}: {exc}): handwriting will be read by Tesseract only. {_fix_hint()}", stacklevel=2)
    return _engine


def safe_read(engine, images: list, label: str = "handwriting") -> list[str]:
    """engine.read(images), but a failure (GPU out of memory, a corrupt crop, a half precision error ...) gives empty readings instead of aborting the whole field:
    the other engines still vote. The error is kept for the warning of the result (see unavailable_warning)."""
    try:
        out = engine.read(images)
        _runtime_errors[label] = ""
        return out
    except Exception as exc:
        _runtime_errors[label] = f"{type(exc).__name__}: {str(exc)[:160]}"
        import logging
        logging.getLogger("documind").exception("recognizer '%s' failed while reading", label)
        return [""] * len(images)


def safe_nbest(engine, image, n: int = 5, label: str = "handwriting") -> list[str]:
    """engine.read_nbest(image, n), [] when the engine cannot (older recognizer) or fails."""
    fn = getattr(engine, "read_nbest", None)
    if fn is None:
        return []
    try:
        out = fn(image, n)
        _runtime_errors[label] = ""
        return out
    except Exception as exc:
        _runtime_errors[label] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return []


def _fix_hint() -> str:
    """The exact commands, for the Python that is running right now (the one Streamlit / uvicorn uses may not be the one pip installed into)."""
    py = sys.executable or "python"
    if "No module named" in _error:
        return f'Install it into THIS Python: "{py}" -m pip install -r requirements.txt   then, once, with internet access: "{py}" -m documind.ocr.handwriting'
    return f'Check it with: "{py}" -m documind.ocr.handwriting (first use downloads ~1.3 GB, so it needs internet access)'


def unavailable_warning() -> dict | None:
    """A warning for the result of a document with handwriting, when the recognizer cannot be used: otherwise the failure is only a line in a log and the
    handwritten date / amount / name just 'is not detected'."""
    if mode() == "off":
        return None
    if get_engine() is not None:
        failed = {k: v for k, v in _runtime_errors.items() if v}
        if failed:
            return {"code": "handwriting_failed", "message": "A text recognizer loaded but failed while reading, so Tesseract (which cannot read handwriting) was used for those values: "
                    + "; ".join(f"{k}: {v}" for k, v in failed.items())}
        return None
    why = f" ({_error})" if _error else ""
    return {"code": "handwriting_unavailable", "message": "The handwriting recognizer is not loaded" + why + ": handwritten dates, amounts and names are read by Tesseract, which cannot read "
            "handwriting. " + _fix_hint()}


def tesseract_conf(img: Image.Image, xyxy, pad: int = 2, scale: float = 3.0, lang: str | None = None) -> float:
    """Mean Tesseract confidence (0-100) on one box, read as a text line. Print scores 80-95, handwriting usually below 60, an empty read gives 0."""
    import pytesseract
    from pytesseract import Output
    from .engine import resolve_lang
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    x0, y0, x1, y1 = xyxy
    crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))).convert("L")
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    crop = ImageOps.expand(ImageOps.autocontrast(crop, cutoff=1), border=16, fill=255)
    d = pytesseract.image_to_data(crop, lang=resolve_lang(lang or OCR_LANG), config="--psm 7", output_type=Output.DICT)
    confs = [float(c) for c, t in zip(d["conf"], d["text"]) if str(t).strip() and float(c) >= 0]
    return float(np.mean(confs)) if confs else 0.0


def wanted(field: str | None, conf: float, plausibility: float) -> bool:
    """Ask the handwriting recognizer about this box? Yes for a field that is handwritten by nature, when Tesseract was unsure, or when its text does not fit the field."""
    return (field in ALWAYS) or conf < LOW_CONF or plausibility < 0.6


# ---------------------------------------------------------------------------------------------- full page
def _segments(words: list[dict], lh: float) -> list[list[dict]]:
    """Words -> runs on one line, close together. Lines first (a tall handwritten word and its neighbour differ by a few pixels in height), then left to right inside a line."""
    cy = lambda w: (w["y0"] + w["y1"]) / 2
    lines: list[list[dict]] = []
    for w in sorted(words, key=cy):
        for ln in lines:
            if abs(cy(w) - float(np.mean([cy(v) for v in ln]))) < 0.7 * max(lh, w["y1"] - w["y0"]):
                ln.append(w)
                break
        else:
            lines.append([w])
    segs: list[list[dict]] = []
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
        cur = [ln[0]]
        for w in ln[1:]:
            if w["x0"] - max(v["x1"] for v in cur) < 4 * lh:
                cur.append(w)
            else:
                segs.append(cur)
                cur = [w]
        segs.append(cur)
    return segs


def rescue(gray: np.ndarray, words: list[dict], scale: float = 1.0, lang: str | None = None, engine=None) -> list[dict]:
    """Full page: the lines where Tesseract found only low-confidence junk (what handwriting looks like to it) are re-read by the handwriting recognizer.
    `words` = run_ocr output (original-image coordinates), `gray` = the image they were read on, `scale` = its enlargement. A scribble next to the word "Signature"
    is a signature, not text: it is left alone. Returns the words with each re-read run replaced by the recognized words (conf 60, "handwriting": True)."""
    engine = engine or get_engine()
    if engine is None or not words:
        return words
    hs = [w["y1"] - w["y0"] for w in words]
    lh = float(np.median(hs))
    low = [w for w in words if w["conf"] < LOW_CONF and (w["y1"] - w["y0"]) >= 0.5 * lh]
    sig = [w for w in words if re.fullmatch(r"signature|sign[eé]e?|signatures", re.sub(r"[:.\-\s]", "", w["text"].lower()))]
    todo = []
    for s in _segments(low, lh):
        x0, y0, x1, y1 = min(w["x0"] for w in s), min(w["y0"] for w in s), max(w["x1"] for w in s), max(w["y1"] for w in s)
        if (len(s) < 2 and x1 - x0 < 5 * lh) or x1 - x0 < 2.5 * lh:                                          # one stray low-confidence word is noise
            continue
        if any(abs((g["y0"] + g["y1"]) / 2 - (y0 + y1) / 2) < 0.9 * lh and g["x1"] <= x0 + lh for g in sig):    # right of "Signature:"
            continue
        todo.append((s, (x0, y0, x1, y1)))
    if not todo:
        return words
    H, W = gray.shape[:2]
    crops = []
    for _, (x0, y0, x1, y1) in todo:
        a, b, c, d = (int(round(v * scale)) for v in (x0 - 0.3 * lh, y0 - 0.4 * lh, x1 + 0.3 * lh, y1 + 0.4 * lh))
        crops.append(Image.fromarray(gray[max(0, b):min(H, d), max(0, a):min(W, c)]).convert("RGB"))
    out = list(words)
    for (seg, (x0, y0, x1, y1)), text in zip(todo, engine.read(crops, batch_size=16)):
        toks = (text or "").split()
        if not toks or not re.search(r"[A-Za-zÀ-ÿ0-9]", text):
            continue
        ids = {id(w) for w in seg}
        out = [w for w in out if id(w) not in ids]
        total, x = sum(len(t) for t in toks), x0
        for t in toks:                                                                                    # the run's width shared between the words, by length
            nx = x + (x1 - x0) * len(t) / total
            out.append({"text": t, "conf": 60.0, "x0": x, "y0": y0, "x1": nx, "y1": y1, "handwriting": True})
            x = nx
    return out


def main() -> None:
    """python -m documind.ocr.handwriting [line.png ...]   downloads / loads the model (do it once, before the first document) and reads the given line pictures."""
    import sys
    eng = get_engine()
    if eng is None:
        raise SystemExit("The handwriting recognizer is not available (see the warning above).")
    print(f"loaded {MODEL} on {eng.device}")
    for path in sys.argv[1:]:
        print(f"{path}: {eng.read([Image.open(path)])[0]}")


if __name__ == "__main__":
    main()
