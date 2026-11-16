"""Cheques on a layout the YOLO detector was not trained on (British cheque: date, payee and amount in figures missed, "amount in words" put on the payee line)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from documind.inference import cheque_text, cheques
from documind.inference.cheque_text import clean_payee, locate_fields, numeric_text
from documind.inference.contract_text import english_words_to_number

IMG = Path(__file__).parent / "data" / "british_cheque.png"

# the words Tesseract finds on that cheque (610 x 322, sparse reading): (text, conf, x0, y0, x1, y1). The date and the digits are read badly ("{00", "bare").
WORDS = [("FIRST", 95, 74, 18, 140, 36), ("BANK", 96, 150, 18, 216, 36), ("01-02-03", 96, 508, 30, 578, 40), ("OF", 95, 73, 42, 104, 59), ("WIKI", 95, 113, 42, 164, 58),
         ("22/01/2006", 15, 471, 64, 589, 82), ("HEFFIELD", 91, 28, 66, 85, 76), ("CITY", 96, 89, 66, 116, 76), ("CENTRE", 92, 120, 66, 169, 76), ("BRANCH", 90, 174, 66, 224, 76),
         ("bare", 33, 436, 76, 467, 86), ("Weeumedia", 25, 79, 99, 200, 123), ("Fo", 95, 213, 102, 237, 122), ("Lind", 41, 259, 108, 287, 124),
         ("One", 95, 19, 124, 64, 160), ("Hundred", 94, 75, 124, 167, 160), ("Pounds", 59, 184, 124, 262, 160), ("£0", 79, 273, 124, 298, 160),
         ("{00", 36, 462, 127, 504, 158), ("—00", 79, 510, 127, 558, 158), ("MR.", 92, 454, 170, 480, 181), ("JOHN", 93, 485, 170, 526, 182), ("JONES", 96, 531, 170, 581, 182),
         ("Cheque", 96, 22, 253, 53, 262), ("No.", 96, 57, 253, 70, 260), ("Branch", 96, 149, 253, 178, 260), ("Sort", 97, 182, 253, 199, 260), ("Code", 96, 202, 253, 224, 260)]


def _words(rows=WORDS):
    return [{"text": t, "conf": float(c), "x0": float(a), "y0": float(b), "x1": float(d), "y1": float(e)} for t, c, a, b, d, e in rows]


def test_fields_are_located_from_the_layout():
    f = locate_fields(_words(), 610.0, 322.0)
    assert {"date", "amount_digits", "amount_words", "payee"} <= set(f)
    assert f["date"]["candidate"] == "22/01/2006" and f["date"]["box"][0] < 475 and f["date"]["box"][2] > 585 and f["date"]["box"][1] < 70       # not the sort code 01-02-03
    assert f["amount_digits"]["box"][0] < 465 and f["amount_digits"]["box"][2] >= 558 and 120 < f["amount_digits"]["box"][1] < 130            # includes the "1" read as "{"
    aw, py = f["amount_words"]["box"], f["payee"]["box"]
    assert aw[0] < 25 and aw[2] <= f["amount_digits"]["box"][0] and aw[1] < 128 and aw[3] > 158                   # the whole line, up to the amount box (no overlap)
    assert py[3] <= aw[1] + 50 and py[1] > 70 and py[0] < 90 and py[2] > 280                                 # the line above it, not the bank header


def test_helpers():
    assert numeric_text(": 100—00") == "100-00" and numeric_text("l00-OO") == "100-00" and numeric_text("22/O1/2006") == "22/01/2006"
    assert clean_payee("PAY Wikimedia Foundation") == "Wikimedia Foundation" and clean_payee("Payez à l'ordre de Salma Idrissi") == "Salma Idrissi" and clean_payee("Pay to the order of ACME Ltd") == "ACME Ltd" and clean_payee("; Salma Idrissi |") == "Salma Idrissi"
    assert cheques.parse_amount("100—00") == 100.0 and cheques.parse_amount("£100-50") == 100.5 and cheques.parse_amount("1 250,50") == 1250.5
    assert english_words_to_number("One Hundred Pounds Only") == (100, 0, "GBP") and english_words_to_number("one hundred and fifty pounds and twenty pence")[:2] == (150, 20)
    assert cheques.parse_date("22nd January 2006") == ("22/01/2006", True) and cheques.parse_date("12 mars 2024") == ("12/03/2024", True) and cheques.parse_date("22/01/2006")[1]


def test_amount_words_on_the_payee_line_become_the_payee():
    box = lambda t: {"value": t, "raw_text": t, "box_confidence": 0.9, "bbox": [0, 0, 1, 1], "n_boxes": 1}
    got = {"amount_words": box("Wikimedia Foundation"), "signature": box("")}
    layout = {"amount_words": box("One Hundred Pounds Only"), "date": box("22/01/2006"), "amount_digits": box("100.00")}
    m = cheques.merge_layout(got, layout, {"date", "payee", "amount_digits", "amount_words"})
    assert m["payee"]["raw_text"] == "Wikimedia Foundation" and m["amount_words"]["raw_text"] == "One Hundred Pounds Only" and "signature" in m
    badly_read = {"amount_words": box("One Hunderd Pound")}                                                   # number words, just misread: NOT a payee
    assert "payee" not in cheques.merge_layout(badly_read, layout, {"date", "payee", "amount_digits", "amount_words"})
    assert cheques.interpret({**m, "payee": m["payee"]})["checks"]["amount_match"] is True


def _ocr_ready() -> bool:
    try:
        from documind.ocr import installed_langs
        return "eng" in installed_langs()
    except Exception:
        return False


@pytest.mark.skipif(not (_ocr_ready() and IMG.exists()), reason="needs Tesseract (eng) and tests/data/british_cheque.png")
def test_british_cheque_end_to_end_with_a_detector_that_misses_most(monkeypatch):
    class Model:                                                  # what the YOLO model did on this cheque: bank, cheque number, a signature, and "amount words" on the payee line
        names = {0: "bank_name", 1: "amount_words", 2: "signature"}
        overrides = {"imgsz": 640}

        def predict(self, img, **kw):
            mk = lambda c, xyxy: SimpleNamespace(cls=c, conf=0.9, xyxy=[SimpleNamespace(tolist=lambda: xyxy)])
            return [SimpleNamespace(boxes=[mk(0, [10, 10, 220, 40]), mk(1, [12, 90, 335, 125]), mk(2, [370, 190, 600, 280])])]

    real = cheques.ocr_box
    monkeypatch.setattr(cheques, "ocr_box", lambda img, box, **kw: "Wikimedia Foundation" if kw.get("field") == "amount_words" else real(img, box, **kw))
    monkeypatch.setattr(cheque_text, "ocr_box", lambda img, box, **kw: "PAY Wikimedia Foundation" if kw.get("field") == "payee" else real(img, box, **kw))
    res = cheques.extract_cheque_bytes(IMG.read_bytes(), "BritishCheque.png", model=Model())
    r = res["row"]
    assert (r["DATE"], r["AMOUNT"], r["CURRENCY"], r["PAYEE"], r["AMOUNT_CHECK"], r["SIGNATURE"]) == ("22/01/2006", "100.00", "GBP", "Wikimedia Foundation", "ok", "yes")
    assert r["AMOUNT_WORDS"].lower().startswith("one hundred pound") and not [w for w in res["warnings"] if w["code"] in ("missing_field", "amount_mismatch")]


def test_junk_is_not_presented_as_a_payee_or_an_amount():
    assert not cheque_text.looks_like_name("Que Nationwide hundred = and") and not cheque_text.looks_like_name("|") and not cheque_text.looks_like_name("12 ##")
    assert cheque_text.looks_like_name("Wikimedia Foundation") and cheque_text.looks_like_name("Salma Idrissi") and cheque_text.looks_like_name("O'Brien & Sons Ltd.")
    assert cheque_text._enough_number_words("One Hundred Pounds Only") and not cheque_text._enough_number_words("hundred = and")
    r = cheques.interpret({"amount_digits": {"raw_text": "0.00"}})
    assert r["values"]["amount"] == "" and "amount_unreadable" in {w["code"] for w in r["warnings"]}
    assert cheques._readable("amount_digits", "0.00") is False and cheques._readable("amount_digits", "100-00") is True


def test_amount_words_line_is_located_even_when_only_a_fragment_was_read():
    rows = [("hundred", 40, 120, 124, 240, 150), ("=", 30, 250, 130, 262, 142), ("and", 55, 270, 124, 320, 150), ("Nationwide", 90, 30, 20, 220, 60), ("£", 60, 480, 122, 500, 150), ("1250", 35, 505, 120, 580, 152)]
    f = locate_fields(_words(rows), 610.0, 322.0)
    assert "amount_words" in f and f["amount_words"]["box"][1] < 126 and f["amount_words"]["box"][3] > 148 and f["amount_words"]["box"][2] <= f["amount_digits"]["box"][0]


def test_streamlit_app_does_not_import_torch_itself():
    """Importing torch at the top of the app changed the DLL load order ("OMP: Error #15" on Windows + Anaconda)."""
    src = (Path(__file__).parent.parent / "streamlit_app.py").read_text(encoding="utf-8")
    assert "import torch" not in src and 'sys.modules.get("torch")' in src


def test_swapped_payee_and_amount_words_labels_are_put_right():
    box = lambda t: {"value": t, "raw_text": t, "box_confidence": 0.9, "bbox": [0, 0, 1, 1], "n_boxes": 1}
    got = {"amount_words": box("Nationwide - John Smith"), "payee": box("One hundred and fifty"), "amount_digits": box("150.00")}
    fixed = cheques.fix_swapped(got)
    assert fixed["payee"]["raw_text"] == "Nationwide - John Smith" and fixed["amount_words"]["raw_text"] == "One hundred and fifty"
    r = cheques.interpret(fixed)
    assert r["checks"]["amount_match"] is True and r["values"]["payee"] == "Nationwide - John Smith"
    ok = {"amount_words": box("One hundred and fifty"), "payee": box("John Smith")}
    assert cheques.fix_swapped(ok) is ok                                                          # correct labels: untouched
    only = cheques.fix_swapped({"payee": box("Mille deux cents dirhams")})
    assert "payee" not in only and only["amount_words"]["raw_text"] == "Mille deux cents dirhams"


def test_words_in_the_amount_box_are_not_an_amount():
    assert cheque_text.amountish("£ 150 - 00") and cheque_text.amountish("1 250,00 MAD") and cheque_text.amountish("l5O-OO")
    assert not cheque_text.amountish("2 May 2025") and not cheque_text.amountish("Nationwide")


def test_british_style_amounts_and_payee_validation():
    for t, v in [("150 00", 150.0), ("£ 150 - 00", 150.0), ("150–00", 150.0), ("1 250 00 MAD", 1250.0), ("1 250", 1250.0), ("12 345,50", 12345.5), ("1 200 000", 1200000.0)]:
        assert cheques.parse_amount(t) == v, t
    assert cheques.payee_ok("John Smith") and cheques.payee_ok("Nationwide - John Smith") and cheques.payee_ok("3M Ltd") and cheques.payee_ok("Wikimedia Foundation")
    assert not any(cheques.payee_ok(t) for t in ("2 May 2025", "One hundred and fifty", "|", "12 05 2025", "Mille deux cents dirhams"))
    from documind.ocr import trocr
    assert trocr.filter_chars("£ 150 – 00", "0123456789,.- £$€") == "£ 150 - 00" and trocr.filter_chars("1 250,00", "0123456789,. ") == "1 250,00"


# ---- the French specimen cheque ("Spécimen de votre Chèque"): the date read as 27052074, the printed sentence "Payez contre ce chèque ..." that is not a payee label
FR_WORDS = [("de", 96, 175, 36, 208, 59), ("Chèque", 95, 295, 36, 401, 65), ("votre", 96, 216, 39, 286, 59), ("Payez", 96, 33, 79, 83, 97), ("contre", 96, 89, 81, 145, 93), ("ce", 95, 150, 83, 171, 93),
            ("chèque", 96, 176, 79, 240, 97), ("non", 92, 247, 83, 277, 93), ("endossable", 92, 283, 79, 383, 93), ("sauf", 96, 388, 79, 425, 93), ("au", 96, 429, 83, 450, 93), ("profit", 96, 457, 79, 503, 97),
            ("buick", 77, 151, 122, 189, 171), ("cures", 62, 201, 122, 238, 171), ("CQuaraste", 17, 43, 125, 241, 165), ("dinquañts", 9, 247, 127, 357, 169), ("corbines", 43, 361, 127, 437, 157),
            ("[48,50", 48, 775, 137, 887, 175), ("€", 96, 725, 140, 751, 171), ("À", 89, 726, 211, 739, 229), ("\\VYparseille", 14, 762, 203, 881, 245), ("Litine", 25, 73, 217, 158, 257),
            ("Dubois", 42, 163, 216, 243, 257), ("8", 33, 43, 219, 53, 234), ("27052074", 85, 764, 264, 885, 287), ("Le", 95, 727, 270, 746, 284), ("Payable", 92, 100, 282, 159, 298),
            ("en", 96, 164, 286, 181, 295), ("France", 96, 187, 283, 237, 295), ("Votre", 96, 334, 299, 374, 311), ("numéro", 96, 379, 298, 435, 311), ("compte", 96, 463, 300, 519, 313)]
FR_HINTS = {"amount_words": [37, 120, 444, 175], "amount_digits": [788, 135, 888, 177], "signature": [709, 339, 901, 422]}          # what the detector found (it missed payee, place and date)


def test_french_specimen_cheque_date_payee_and_amount():
    f = locate_fields(_words(FR_WORDS), 1060.0, 507.0, FR_HINTS)
    assert f["date"]["candidate"] == "27052074" and f["date"]["box"][0] > 740 and 255 < f["date"]["box"][1] < 275                       # "Le 27052074": compact date after the label "Le"
    py = f["payee"]["box"]
    assert py[1] > 200 and py[3] < 270 and py[0] < 90 and py[2] < 520                                                                   # the handwritten line below the words, NOT the printed "Payez contre ce chèque ..." sentence
    assert f["amount_digits"]["candidate"] == "48,50"


def test_printed_sentence_is_not_a_payee_label():
    f = locate_fields(_words(FR_WORDS), 1060.0, 507.0)                                                                                   # no hints at all
    assert "contre" not in f.get("payee", {}).get("candidate", "")


def test_compact_date_and_to_the_year_prior():
    assert cheques.parse_date("27052024") == ("27/05/2024", True) and cheques.parse_date("27.05.2024") == ("27.05.2024", True)
    from datetime import date, timedelta
    far = (date.today() + timedelta(days=800)).strftime("%d/%m/%Y")
    r = cheques.interpret({"date": {"raw_text": far}})
    assert "date_in_future" in {w["code"] for w in r["warnings"]}
    assert "date_in_future" not in {w["code"] for w in cheques.interpret({"date": {"raw_text": "27/05/2024"}})["warnings"]}


class NBest:
    """A recognizer whose best reading dropped 'huit' (40.50) but whose 3rd choice is right."""
    def __init__(self):
        self.calls = 0

    def read(self, images, batch_size=32):
        return ["Quarante euros et cinquante centimes"] * len(images)

    def read_nbest(self, image, n=5):
        self.calls += 1
        return ["Quarante euros et cinquante centimes", "Quarante-neuf euros et cinquante centimes", "Quarante-huit euros et cinquante centimes"]


def test_disagreeing_amounts_are_reconciled_with_the_next_best_reading(monkeypatch):
    from documind.ocr import handwriting, trocr
    eng = NBest()
    monkeypatch.setattr(handwriting, "get_engine", lambda: eng)
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: None)
    box = lambda t, bb: {"value": t, "raw_text": t, "box_confidence": 0.9, "bbox": bb, "n_boxes": 1}
    got = {"amount_words": box("Quarante euros et cinquante centimes", [10, 10, 200, 40]), "amount_digits": box("48,50", [300, 10, 380, 40])}
    img = Image.new("RGB", (400, 100), "white")
    out = cheques.reconcile_amounts(img, got)
    assert out["amount_words"]["raw_text"] == "Quarante-huit euros et cinquante centimes" and out["amount_words"]["reconciled"] and eng.calls == 1
    assert cheques.interpret(out)["checks"]["amount_match"] is True
    agree = {"amount_words": box("Quarante-huit euros et cinquante centimes", [10, 10, 200, 40]), "amount_digits": box("48,50", [300, 10, 380, 40])}
    assert cheques.reconcile_amounts(img, agree) is agree and eng.calls == 1                                  # already agree: the recognizers are not asked again
    nothing = {"amount_words": box("Quarante euros", [10, 10, 200, 40]), "amount_digits": box("99,99", [300, 10, 380, 40])}
    assert cheques.reconcile_amounts(img, nothing) is nothing                                                  # nothing agrees: the mismatch stays (never invented)
