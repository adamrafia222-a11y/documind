"""Tesseract and the trained recognizer working together: which reading is kept. No GPU, PyTorch or model needed (the trained engine is a fake).
    pytest -q tests/test_ensemble.py
"""
import pytest
from PIL import Image

from documind.inference import cheques, receipts
from documind.ocr import ensemble, trocr
from documind.ocr.ensemble import choose, plausibility


class Fake:
    def __init__(self, text):
        self.text = text

    def read(self, images, batch_size=32):
        return [self.text] * len(images)


# ------------------------------------------------------------------------------------------------ does a reading fit the field?
def test_plausibility_knows_the_format_of_each_field():
    p = plausibility
    assert p("total", "12,50") == 1.0 and p("total", "1 234,50 DH") == 1.0 and p("total", "l2,5O") < 0.5 and p("item_price", "7.20") == 1.0
    assert p("date", "13/10/2025") == 1.0 and p("date", "12 mars 2022") == 1.0 and p("date", "13/1O/2025") < 0.5 and p("date", "31/02/2025") < 0.5
    assert p("time", "20:49") == 1.0 and p("time", "2O:49") < 0.5 and p("phone", "05 24 11 22 33") == 1.0 and p("phone", "05 24 11 2") < 0.5
    assert p("item_qty", "3") == 1.0 and p("item_qty", "S") == 0.0 and p("receipt_no", "03347048") == 1.0 and p("receipt_no", "334") < 1.0
    assert p("cheque_number", "N° 1234567") == 1.0 and p("interest_rate", "5,45 %") == 1.0 and p("interest_rate", "5,45") < 1.0 and p("duration", "36 mois") == 1.0
    assert p("iban", "MA64 1234 5678 9012 3456 7890 1234") == 1.0 and p("iban", "MA64 1234") < 0.5 and p("iban", "FR76 " + "1" * 23) == 1.0
    assert p("amount_words", "cent mille dirhams") == 1.0 and p("amount_words", "xx yy zz") < 0.3 and p("micr_line", "<1234567<  4560:  1234<45") > 0.9
    assert p("payee", "Omar Tazi") > p("payee", "Om|r T@zi") and p("payee", "Omar Tazi") > p("payee", "0mar T4zi 123") and p("item_name", "T0TAL") < p("item_name", "TOTAL")
    assert p("total", "") == 0.0 and p(None, "anything") > 0.5


# ------------------------------------------------------------------------------------------------ which reading is kept
def test_one_engine_empty_or_both_equal():
    assert choose({"trocr": "", "tesseract": "12,50"}, "total").reason == "only" and choose({"trocr": "", "tesseract": "12,50"}, "total").source == "tesseract"
    assert choose({"trocr": "12,50", "tesseract": None}, "total").source == "trocr" and choose({"trocr": "", "tesseract": ""}, "total").text == ""
    c = choose({"trocr": "Marché  Central", "tesseract": "marché central"}, "store")
    assert c.reason == "agreed" and c.text == "Marché  Central"                                   # same text up to spacing and case: the primary engine's own spelling


def test_the_engine_that_fits_the_field_wins():
    c = choose({"trocr": "13/1O/2025", "tesseract": "13/10/2025"}, "date")
    assert (c.text, c.source, c.reason) == ("13/10/2025", "tesseract", "format")
    c = choose({"trocr": "1 234,50 DH", "tesseract": "1 234.5O DH"}, "total")
    assert c.source == "trocr" and c.reason == "format"
    assert choose({"trocr": "0S 24 11 22 33", "tesseract": "05 24 11 22 33"}, "phone").source == "tesseract"


def test_when_both_look_fine_the_primary_engine_decides(monkeypatch):
    c = choose({"trocr": "Marché Central", "tesseract": "Marche Central"}, "store")
    assert (c.source, c.reason, c.text) == ("trocr", "primary", "Marché Central")                 # Tesseract drops accents, the trained model keeps them
    assert choose({"trocr": "Marché Central", "tesseract": "Marche Central"}, "store", primary="tesseract").text == "Marche Central"
    monkeypatch.setenv("DOCUMIND_OCR_PRIMARY", "tesseract")
    assert choose({"trocr": "Marché Central", "tesseract": "Marche Central"}, "store").source == "tesseract"


def test_a_cross_check_beats_the_format():
    only_hundred = lambda t: t.startswith("cent")
    c = choose({"trocr": "mille dirhams", "tesseract": "cent mille dirhams"}, "amount_words", prefer=only_hundred)
    assert (c.source, c.reason) == ("tesseract", "cross_check")
    both = choose({"trocr": "cent dirhams", "tesseract": "cent mille dirhams"}, "amount_words", prefer=only_hundred)
    assert both.reason != "cross_check"                                                           # the check accepts both: it decides nothing


def test_stats_count_how_readings_were_decided():
    ensemble.STATS.clear()
    choose({"trocr": "a", "tesseract": "a"}, "store")
    choose({"trocr": "13/1O/2025", "tesseract": "13/10/2025"}, "date")
    choose({"trocr": "", "tesseract": "x"}, "store")
    assert ensemble.summary() == {"agreed": 1, "format_tesseract": 1, "only_tesseract": 1}


# ------------------------------------------------------------------------------------------------ wired into the readers
@pytest.fixture()
def blank():
    return Image.new("RGB", (120, 40), "white")


def test_ocr_box_modes(monkeypatch, blank):
    monkeypatch.setattr(receipts, "_tesseract_box", lambda *a, **k: "13/10/2025")
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: Fake("13/1O/2025"))
    box = (5, 5, 100, 30)
    monkeypatch.setenv("DOCUMIND_OCR", "auto")
    assert receipts.ocr_box(blank, box, field="date") == "13/10/2025"                              # both engines read, the date that is a date wins
    monkeypatch.setenv("DOCUMIND_OCR", "both")
    assert receipts.ocr_box(blank, box, field="date") == "13/10/2025"
    monkeypatch.setenv("DOCUMIND_OCR", "trocr")
    assert receipts.ocr_box(blank, box, field="date") == "13/1O/2025"                              # the trained model alone, as asked
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: None)
    monkeypatch.setenv("DOCUMIND_OCR", "auto")
    assert receipts.ocr_box(blank, box, field="date") == "13/10/2025"                              # no trained model: Tesseract alone


def test_ocr_box_whitelist_is_applied_to_the_trained_model_before_the_vote(monkeypatch, blank):
    monkeypatch.setattr(receipts, "_tesseract_box", lambda *a, **k: "")
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: Fake("1 234,50 DH"))
    monkeypatch.setenv("DOCUMIND_OCR", "auto")
    assert receipts.ocr_box(blank, (5, 5, 100, 30), whitelist="0123456789,. ", field="total") == "1 234,50"


def test_the_missing_model_is_an_error_only_when_asked_for_explicitly(monkeypatch, tmp_path):
    monkeypatch.setattr(trocr, "_engine", None)
    monkeypatch.setattr(trocr, "_tried", False)
    monkeypatch.setenv("DOCUMIND_OCR", "auto")
    assert trocr.get_engine(tmp_path / "nothing") is None
    for mode in ("both", "trocr"):
        monkeypatch.setenv("DOCUMIND_OCR", mode)
        with pytest.raises(SystemExit):
            trocr.get_engine(tmp_path / "nothing")
    monkeypatch.setenv("DOCUMIND_OCR", "tesseract")
    assert trocr.get_engine(tmp_path / "nothing") is None


def test_cheque_reader_passes_the_field_to_the_ensemble(monkeypatch):
    seen = {}
    monkeypatch.setattr(cheques, "ocr_box", lambda img, xyxy, **kw: seen.setdefault(kw.get("field"), "x") or "x")
    cheques.read_text(Image.new("RGB", (50, 50)), (0, 0, 10, 10), "amount_digits")
    assert "amount_digits" in seen
