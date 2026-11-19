"""OCR reconstruction: read the picture, then rebuild its content in the form the models were trained on.

    Stage A  documind.preprocessing.canonical   the PICTURE is cut out, turned upright, flattened          (page in training form)
    Stage B  reconstruct_words()                the WORDS are re-read by the trained recognizer and cleaned (text in training form)
    Stage C  render_clean()                     the words are drawn again on a clean page                   (to look at what the models see)

Stage B. Tesseract finds the words and their boxes. The words are grouped into short chunks (a key, a value, a table cell: the size of the crops the
recognizer was fine-tuned on, documind.training.train_ocr). Each chunk is re-read by the trained TrOCR and the two readings go through
documind.ocr.ensemble.choose (same rules as everywhere else: agreed / format fits the field / the primary engine). When the recognizer wins, its text
is written back on the Tesseract boxes (one token per box; when the number of tokens differs, the chunk is cut again in proportion to the characters).
Then every token is put in one canonical spelling (typographic dashes and quotes, look-alike letters inside numbers: "2O25" -> "2025", table-border junk removed).

It runs when a picture is UPLOADED. Nothing is retrained and the dataset is not touched: the models you already have receive a page and words in a
better form. The recognizer is your fine-tuned models/ocr_trocr when it exists, otherwise the PRETRAINED microsoft/trocr-base-printed (downloaded once, ~1.3 GB;
RECONSTRUCT_OCR_MODEL / DOCUMIND_PRETRAINED_OCR in .env).

DOCUMIND_RECONSTRUCT (environment / .env):
    auto (default)  Stage A (the picture) on every upload; Stage B (the words) off
    on              Stage A + Stage B on every upload
    off             nothing (the pipeline works as before)
Stage B is opt-in: it was measured WORSE than `off` (36.1 % -> 33.1 % exact) with the old word classifier, which had learned from Tesseract's text.
With the YOLO detector (fields read box by box) that argument no longer applies: use `python -m documind.training.compare_reconstruct` to see which setting wins on your files.

Look at what it does on one picture (writes the rectified page, the rebuilt page and the words to reports/reconstruct):
    python -m documind.ocr.reconstruct photo.jpg --doc receipt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STATS: Counter = Counter()
VERSION = 1                                                        # written to the model meta: a change of the rules means "retrain"
MAX_CHUNK_CHARS = 40                                               # the recognizer was trained on field-sized crops, not on whole sentences
SKIP_CONF = float(os.getenv("DOCUMIND_RECONSTRUCT_SKIP_CONF", "96"))   # chunks Tesseract reads with this confidence or more are not re-read (speed)
REPLACED_CONF = 95.0                                               # same value as ocr/refine.py: the recognizer gives no confidence of its own


def mode() -> str:
    m = os.getenv("DOCUMIND_RECONSTRUCT", "auto").lower()
    return m if m in ("on", "off", "auto") else "auto"


def pages_active() -> bool:
    """Stage A (picture) runs unless DOCUMIND_RECONSTRUCT=off."""
    return mode() != "off"


def words_active(meta: dict | None = None) -> bool:
    """Stage B (words re-read by the OCR model) runs only with DOCUMIND_RECONSTRUCT=on (`meta` is kept for older callers and ignored)."""
    return mode() == "on"


# ------------------------------------------------------------------------------------------------ canonical spelling
_MAP = str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "\u00a0": " ", "\u2009": " "})
_JUNK_EDGE = "|_~`^\\"
_LOOK = {"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8", "Z": "2"}
_NUMERIC = re.compile(r"^[-(]?[$€£]?\d[\d,.:/\-]*\d[$€£%]?\)?$")


def normalize_token(t: str) -> str:
    """One word in canonical spelling. Conservative: letters are only turned into digits inside a token that is clearly a number / date / amount."""
    t = unicodedata.normalize("NFC", t or "").translate(_MAP)
    t = "".join(c for c in t if c.isprintable()).strip().strip(_JUNK_EDGE)
    if len(t) >= 3:
        look = sum(c in _LOOK for c in t)
        digits = sum(c.isdigit() for c in t)
        if 0 < look <= 3 and look <= 0.35 * len(t) and digits >= 2 and digits >= 0.4 * len(t):
            fixed = "".join(_LOOK.get(c, c) for c in t)
            if _NUMERIC.match(fixed):
                return fixed
    return t


# ------------------------------------------------------------------------------------------------ chunks and writing back
def _chunks(words: list[dict], max_chars: int = MAX_CHUNK_CHARS) -> list[list[int]]:
    """Word indices grouped in chunks: the words of one cell (documind.features.layout), cut when a chunk would be longer than max_chars."""
    from ..features.layout import assign_lines, build_structure
    struct = build_structure(words, assign_lines(words))
    out: list[list[int]] = []
    for cell in struct.cells:
        cur: list[int] = []
        n = 0
        for i in cell["ids"]:
            L = len(words[i]["text"]) + 1
            if cur and n + L > max_chars:
                out.append(cur)
                cur, n = [], 0
            cur.append(i)
            n += L
        if cur:
            out.append(cur)
    return out


def _union(ws: list[dict]) -> tuple[float, float, float, float]:
    return min(w["x0"] for w in ws), min(w["y0"] for w in ws), max(w["x1"] for w in ws), max(w["y1"] for w in ws)


def _write_back(ws: list[dict], text: str) -> list[dict]:
    """The new text of a chunk on the old boxes: token k on box k, or (different counts) the chunk's span cut in proportion to the characters."""
    tokens = text.split()
    if not tokens:
        return ws
    if len(tokens) == len(ws):
        return [{**w, "text": t, "conf": REPLACED_CONF, "refined": True} for w, t in zip(ws, tokens)]
    x0, y0, x1, y1 = _union(ws)
    total = sum(len(t) for t in tokens) + len(tokens) - 1
    out, pos = [], 0
    for t in tokens:
        a = x0 + (x1 - x0) * pos / total
        pos += len(t)
        b = x0 + (x1 - x0) * pos / total
        pos += 1
        out.append({"text": t, "conf": REPLACED_CONF, "x0": a, "y0": y0, "x1": b, "y1": y1, "refined": True})
    return out


def reconstruct_words(gray: np.ndarray, words: list[dict], scale: float = 1.0, engine=None, *, normalize: bool = True) -> list[dict]:
    """Tesseract words (boxes in ORIGINAL coordinates; `gray` is the image they were read on, `scale` its enlargement) -> the same words in training form.

    The input list is not modified. Without a trained recognizer only the canonical spelling is applied.
    """
    words = [dict(w) for w in words]
    if not words:
        return words
    if engine is None:
        from . import trocr
        engine = trocr.get_engine()
    if engine is not None:
        from .ensemble import choose
        H, W = gray.shape[:2]
        todo, crops = [], []
        for idx in _chunks(words):
            if any(words[i].get("handwriting") for i in idx):                       # handwriting has its own recognizer: a printed-text model would make it worse
                continue
            ws = [words[i] for i in idx]
            if float(np.mean([w["conf"] for w in ws])) >= SKIP_CONF:
                STATS["skipped_confident"] += 1
                continue
            x0, y0, x1, y1 = (int(round(v * scale)) for v in _union(ws))
            x0, y0, x1, y1 = max(0, x0 - 3), max(0, y0 - 3), min(W, x1 + 3), min(H, y1 + 3)
            if x1 - x0 >= 6 and y1 - y0 >= 6:
                crops.append(Image.fromarray(gray[y0:y1, x0:x1]).convert("RGB"))
                todo.append(idx)
        replaced: dict[int, list[dict]] = {}
        reads = engine.read(crops, batch_size=64) if crops else []
        for idx, read in zip(todo, reads):
            ws = [words[i] for i in idx]
            tess = " ".join(w["text"] for w in ws)
            conf = float(np.mean([w["conf"] for w in ws]))
            pick = choose({"trocr": (read or "").strip(), "tesseract": tess}, primary="trocr" if conf < 50 else "tesseract")
            if pick.source == "trocr" and " ".join(pick.text.split()) != tess:
                STATS["replaced"] += 1
                replaced[idx[0]] = _write_back(ws, pick.text)
                for i in idx[1:]:
                    replaced[i] = []
            else:
                STATS["kept"] += 1
        new: list[dict] = []
        for i, w in enumerate(words):
            new += replaced[i] if i in replaced else [w]
        words = new
    if normalize:
        out = []
        for w in words:
            t = normalize_token(w["text"])
            if t:
                out.append({**w, "text": t} if t != w["text"] else w)
            else:
                STATS["junk_dropped"] += 1
        words = out
    return words


def summary() -> dict:
    """How many chunks were kept / replaced / skipped so far in this process."""
    return dict(STATS)


# ------------------------------------------------------------------------------------------------ Stage C: the page drawn again
def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_clean(words: list[dict], size: tuple[int, int]) -> Image.Image:
    """The words drawn black on a white page of `size` (width, height), each one fitted to its own box: the layout of the picture, none of its noise.

    Only printed text is rebuilt (no logos, lines, stamps or handwriting), so use it to see what was read, not as the input of a detector.
    """
    W, H = size
    page = Image.new("L", (W, H), 255)
    for w in words:
        x0, y0, x1, y1 = (int(round(w[k])) for k in ("x0", "y0", "x1", "y1"))
        bw, bh = x1 - x0, y1 - y0
        if bw < 2 or bh < 4 or not w["text"]:
            continue
        f = _font(max(8, min(120, int(bh * 0.9))))
        tmp = Image.new("L", (int(f.getlength(w["text"])) + 8, int(bh * 1.6) + 8), 255)
        ImageDraw.Draw(tmp).text((4, 4), w["text"], font=f, fill=0)
        ink = tmp.point(lambda v: 255 if v < 200 else 0).getbbox()
        if not ink:
            continue
        glyphs = tmp.crop(ink).resize((bw, bh), Image.LANCZOS)
        region = page.crop((x0, y0, x1, y1))
        page.paste(Image.fromarray(np.minimum(np.asarray(region), np.asarray(glyphs))), (x0, y0))
    return page.convert("RGB")


# ------------------------------------------------------------------------------------------------ look at one picture
def main() -> None:
    import cv2
    from ..config import OCR_LANG, OCR_PSM, REPORTS_DIR
    from ..preprocessing import preprocess, read_image
    from ..preprocessing.canonical import canonicalize
    from . import run_ocr

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture")
    ap.add_argument("--doc", default="invoice", choices=["invoice", "receipt", "cheque", "contract"])
    ap.add_argument("--out", default=str(REPORTS_DIR / "reconstruct"))
    ap.add_argument("--lang", default=OCR_LANG)
    ap.add_argument("--psm", type=int, default=OCR_PSM)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    img = read_image(a.picture)
    canon = canonicalize(img, a.doc)
    print(f"Stage A: steps = {canon.steps or 'none (already in training form)'}; {img.shape[1]}x{img.shape[0]} -> {canon.image.shape[1]}x{canon.image.shape[0]}")
    cv2.imencode(".png", canon.image)[1].tofile(str(out / "1_canonical.png"))
    prep = preprocess(canon.image)
    raw = run_ocr(prep.image, scale=prep.scale, lang=a.lang, psm=a.psm)
    words = reconstruct_words(prep.image, raw, prep.scale)
    changed = [(r["text"], w["text"]) for r, w in zip(raw, words) if r["text"] != w["text"]] if len(raw) == len(words) else []
    print(f"Stage B: {len(raw)} Tesseract words -> {len(words)} rebuilt, {summary() or 'no recognizer installed: spelling only'}")
    for before, after in changed[:15]:
        print(f"    {before!r} -> {after!r}")
    page = render_clean(words, (canon.image.shape[1], canon.image.shape[0]))
    page.save(out / "2_rebuilt.png")
    both = Image.new("RGB", (canon.image.shape[1] * 2, canon.image.shape[0]), "white")
    both.paste(Image.fromarray(canon.image[:, :, ::-1]), (0, 0))
    both.paste(page, (canon.image.shape[1], 0))
    both.save(out / "3_side_by_side.png")
    (out / "words.json").write_text(json.dumps(words, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved in {out}: 1_canonical.png, 2_rebuilt.png, 3_side_by_side.png, words.json")


if __name__ == "__main__":
    main()
