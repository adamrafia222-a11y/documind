"""Handwriting: Tesseract reads print only. A handwriting recognizer is consulted for boxes that are not print and for low-confidence lines of a page.
These tests use a FAKE recognizer (the real one needs torch + a 1.3 GB download): they check WHEN it is used and how its text is merged, not its accuracy."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from documind.ocr import handwriting
from documind.inference.receipts import ocr_box


class Fake:
    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    def read(self, images, batch_size: int = 32):
        self.calls += len(images)
        return [self.text] * len(images)


@pytest.fixture
def fake(monkeypatch):
    def install(text: str) -> Fake:
        f = Fake(text)
        monkeypatch.setattr(handwriting, "_engine", f)
        monkeypatch.setattr(handwriting, "_tried", True)
        from documind.ocr import trocr
        monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: None)          # no trained printed model: Tesseract + handwriting only
        return f
    return install


def _ocr_ready() -> bool:
    try:
        from documind.ocr import installed_langs
        return "eng" in installed_langs()
    except Exception:
        return False


def _scribble() -> Image.Image:
    """Something Tesseract cannot read: a wavy hand-drawn-looking line."""
    rng = np.random.default_rng(3)
    im = Image.new("RGB", (420, 70), "white")
    d = ImageDraw.Draw(im)
    xs = np.arange(10, 410, 6)
    ys = 35 + 18 * np.sin(xs / 9.0) + rng.normal(0, 5, xs.size)
    d.line(list(zip(xs.tolist(), ys.tolist())), fill=(10, 10, 90), width=3)
    return im


def _printed(text: str) -> Image.Image:
    im = Image.new("RGB", (420, 70), "white")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((12, 14), text, font=font, fill=(0, 0, 0))
    return im


@pytest.mark.skipif(not _ocr_ready(), reason="needs Tesseract (eng)")
def test_handwritten_box_is_read_by_the_handwriting_recognizer(fake):
    f = fake("Mohamed Benali")
    assert ocr_box(_scribble(), (0, 0, 420, 70), field="payee") == "Mohamed Benali" and f.calls == 1


@pytest.mark.skipif(not _ocr_ready(), reason="needs Tesseract (eng)")
def test_clean_printed_amount_never_reaches_the_handwriting_recognizer(fake):
    f = fake("9999,99")
    assert ocr_box(_printed("1250,00"), (0, 0, 420, 70), field="total", whitelist="0123456789,. ") == "1250,00" and f.calls == 0


def test_missing_model_is_not_fatal(monkeypatch):
    monkeypatch.setattr(handwriting, "_engine", None)
    monkeypatch.setattr(handwriting, "_tried", False)
    monkeypatch.delenv("DOCUMIND_HANDWRITING", raising=False)
    monkeypatch.setenv("HANDWRITING_MODEL", "/nonexistent/folder")
    monkeypatch.setattr(handwriting, "MODEL", "/nonexistent/folder")
    with pytest.warns(UserWarning, match="Tesseract only"):
        assert handwriting.get_engine() is None
    monkeypatch.setenv("DOCUMIND_HANDWRITING", "off")
    assert handwriting.mode() == "off"
    monkeypatch.setenv("DOCUMIND_HANDWRITING", "auto")
    monkeypatch.setenv("DOCUMIND_OCR", "tesseract")
    assert handwriting.mode() == "off"                                           # "Tesseract only" means only


def _w(text, x0, y0, x1, y1, conf):
    return {"text": text, "conf": conf, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def test_page_rescue_rereads_junk_lines_and_leaves_signatures_alone():
    gray = np.full((400, 900), 255, np.uint8)
    words = [_w("Total", 20, 20, 90, 40, 95), _w("due:", 100, 20, 150, 40, 94),                       # printed, sure
             _w("Ihe", 160, 18, 220, 44, 22), _w("sz7", 230, 18, 300, 46, 15), _w("xx", 310, 20, 380, 44, 18),       # handwriting read as junk
             _w("Signature:", 20, 200, 130, 220, 95), _w("PE", 140, 190, 200, 230, 20), _w("art", 210, 195, 280, 225, 18),   # a signature scribble
             _w("zz", 20, 300, 60, 320, 12)]                                                          # one stray low word
    f = Fake("paid in full")
    out = handwriting.rescue(gray, words, 1.0, engine=f)
    texts = [w["text"] for w in out]
    assert {"Total", "due:", "Signature:", "PE", "art", "zz"} <= set(texts) and {"paid", "in", "full"} <= set(texts)
    assert not ({"Ihe", "sz7", "xx"} & set(texts)) and f.calls == 1                                   # only the one run was re-read
    new = [w for w in out if w.get("handwriting")]
    assert len(new) == 3 and min(w["x0"] for w in new) == 160 and max(w["x1"] for w in new) == pytest.approx(380)
    assert handwriting.rescue(gray, words, 1.0, engine=Fake("")) == words                                  # nothing recognized: unchanged
