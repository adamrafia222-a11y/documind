"""A handwritten date is read by voting between engines/sizes; the handwriting recognizer is always consulted for dates; its absence is reported in the result."""
from __future__ import annotations

from datetime import date

import pytest
from PIL import Image

from documind.inference import cheques, date_reader
from documind.ocr import handwriting, trocr


class Fake:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), 0

    def read(self, images, batch_size=32):
        self.calls += len(images)
        return [self.texts[i % len(self.texts)] for i in range(len(images))]


@pytest.fixture
def img():
    return Image.new("RGB", (400, 100), "white")


def _setup(monkeypatch, tess: dict, hw=None):
    monkeypatch.setattr(date_reader, "_tesseract_box", lambda im, xyxy, psm, pad, sc, lang, wl: tess[sc])
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: None)
    monkeypatch.setattr(handwriting, "get_engine", lambda: hw)


def test_handwriting_model_wins_over_three_confused_tesseract_readings(monkeypatch, img):
    _setup(monkeypatch, {1.0: "22/O1/2OO8", 0.7: "ZZ/0I/2006x", 1.4: "2 2 0 1"}, Fake("22/01/2006", "22 / 01 / 2006"))
    r = date_reader.read_date_box(img, [0, 0, 400, 100])
    assert r["date"] == date(2006, 1, 22) and r["text"] == "22/01/2006"


def test_without_the_model_the_tesseract_readings_vote(monkeypatch, img):
    _setup(monkeypatch, {1.0: "1 decembre 2005", 0.7: "1 decembre 2OO5", 1.4: "7 décembre 2005"})
    r = date_reader.read_date_box(img, [0, 0, 400, 100])
    assert r["date"] == date(2005, 12, 1)                                                      # two of three agree (the OCR slip 2OO5 is repaired)


def test_written_month_from_the_handwriting_model_and_the_page_reading_vote(monkeypatch, img):
    _setup(monkeypatch, {1.0: "", 0.7: "", 1.4: ""}, Fake("1 December 2005", "1 Decembre 2005"))
    r = date_reader.read_date_box(img, [0, 0, 400, 100], extra={"page": "1 decembre 2005"})
    assert r["date"] == date(2005, 12, 1) and set(r["readings"]) >= {"handwriting", "handwriting_wide", "page"}


def test_nothing_is_a_date_returns_the_best_raw_reading(monkeypatch, img):
    _setup(monkeypatch, {1.0: "~~", 0.7: "x", 1.4: ""}, Fake("scribble"))
    r = date_reader.read_date_box(img, [0, 0, 400, 100])
    assert r["date"] is None and r["text"] == "scribble"


def test_dates_and_figures_always_go_to_the_handwriting_recognizer():
    assert {"date", "amount_digits", "payee", "amount_words", "signing_date"} <= handwriting.ALWAYS
    assert handwriting.wanted("date", 95.0, 1.0) and not handwriting.wanted("total", 95.0, 1.0)


def test_missing_handwriting_model_is_reported_in_the_result(monkeypatch):
    monkeypatch.setattr(handwriting, "get_engine", lambda: None)
    monkeypatch.setattr(handwriting, "_error", "ModuleNotFoundError: No module named 'torch'")
    monkeypatch.delenv("DOCUMIND_HANDWRITING", raising=False)
    monkeypatch.delenv("DOCUMIND_OCR", raising=False)
    w = handwriting.unavailable_warning()
    assert w["code"] == "handwriting_unavailable" and "torch" in w["message"] and "documind.ocr.handwriting" in w["message"] and "pip install -r requirements.txt" in w["message"]
    monkeypatch.setattr(handwriting, "get_engine", lambda: object())
    assert handwriting.unavailable_warning() is None
    monkeypatch.setattr(handwriting, "get_engine", lambda: None)
    monkeypatch.setenv("DOCUMIND_HANDWRITING", "off")
    assert handwriting.unavailable_warning() is None                                           # switched off on purpose: no warning


def test_detector_date_box_goes_through_the_vote(monkeypatch, img):
    _setup(monkeypatch, {1.0: "22/01/2006", 0.7: "22/01/2006", 1.4: "22/O1/2006"})
    got = cheques.fields_from_boxes(img, [("date", [10, 10, 300, 80], 0.9)])
    assert cheques.parse_date(got["date"]["raw_text"]) == ("22/01/2006", True) and "tesseract_x1.0" in got["date"]["readings"]


class Boom:
    def read(self, images, batch_size=32):
        raise RuntimeError("CUDA out of memory")


def test_a_recognizer_that_crashes_does_not_lose_the_date(monkeypatch, img):
    _setup(monkeypatch, {1.0: "2 May 2025", 0.7: "2 May 2025", 1.4: "2g 208"}, Boom())
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: Boom())
    monkeypatch.delenv("DOCUMIND_HANDWRITING", raising=False)
    monkeypatch.delenv("DOCUMIND_OCR", raising=False)
    r = date_reader.read_date_box(img, [0, 0, 400, 100])
    assert r["date"] == date(2025, 5, 2)                                                       # the Tesseract readings still vote
    w = handwriting.unavailable_warning()
    assert w["code"] == "handwriting_failed" and "CUDA out of memory" in w["message"]


def test_a_failing_layout_reading_is_reported_not_swallowed(monkeypatch):
    from documind.inference import cheque_text
    monkeypatch.setattr(cheque_text, "read_cheque_layout", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    import io
    buf = io.BytesIO()
    Image.new("RGB", (300, 120), "white").save(buf, "PNG")
    res = cheques.extract_cheque_bytes(buf.getvalue(), "x.png")
    assert "layout_failed" in {w["code"] for w in res["warnings"]} and any("boom" in w["message"] for w in res["warnings"])


def test_three_sizes_of_one_engine_are_one_vote_not_three(monkeypatch, img):
    today = date.today()
    wrong, right = f"27052{str(today.year + 50)[-3:]}", "27.05." + str(today.year)           # Tesseract: the same wrong year at all three sizes; the handwriting model: the right one
    _setup(monkeypatch, {1.0: wrong, 0.7: wrong, 1.4: wrong}, Fake(right, right))
    r = date_reader.read_date_box(img, [0, 0, 400, 100], extra={"page": wrong})
    assert r["date"] == date(today.year, 5, 27)
