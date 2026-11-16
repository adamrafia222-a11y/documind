"""Dates written with a month name ("1 decembre 2005") are read like 22/01/2006: cheque boxes, the cheque layout reader, invoices, receipts, contracts."""
from __future__ import annotations

import io
from datetime import date

import pytest
from PIL import Image, ImageDraw, ImageFont

from documind.inference import cheque_text, cheques, contracts, postprocess
from documind.inference.dates import find_dates, month_of, parse_date


@pytest.mark.parametrize("text,expected", [
    ("1 decembre 2005", date(2005, 12, 1)), ("1er décembre 2005", date(2005, 12, 1)), ("le 1 Déc. 05", date(2005, 12, 1)), ("December 1st, 2005", date(2005, 12, 1)),
    ("1-Dec-2005", date(2005, 12, 1)), ("١ ديسمبر ٢٠٠٥", date(2005, 12, 1)), ("l décembre 2OO5", date(2005, 12, 1)), ("2005-12-01", date(2005, 12, 1)), ("22 01 2006", date(2006, 1, 22)),
    ("22/01/2006", date(2006, 1, 22)), ("22nd of January, 2006", date(2006, 1, 22)), ("Fait à Rabat, le 12 mars 2024", date(2024, 3, 12)), ("3 aout 2023", date(2023, 8, 3)),
    ("decembe 1 2005", date(2005, 12, 1)), ("le premier mars 2024", date(2024, 3, 1)), ("12/25/2005", date(2005, 12, 25)), ("30 Juin 2024", date(2024, 6, 30)),
    ("31 février 2024", None), ("12 mars", None), ("not a date", None), ("", None),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_find_dates_in_a_sentence_and_month_names():
    f = find_dates("Fait à Rabat, le 1er déc. 2005 et le 22/01/2006")
    assert [(x.date, x.kind) for x in f] == [(date(2005, 12, 1), "named"), (date(2006, 1, 22), "numeric")] and f[0].text == "1er déc. 2005"
    assert month_of("Décembre") == month_of("decembre") == month_of("DEC.") == month_of("decembe") == 12 and month_of("table") is None and month_of("12") is None


def test_cheque_date_in_letters_is_valid_and_normalized():
    assert cheques.parse_date("1 decembre 2005") == ("01/12/2005", True) and cheques.parse_date("22/01/2006") == ("22/01/2006", True)
    assert cheques.parse_date("31 février 2024")[1] is False and "date" not in cheques.WHITELIST          # no digits-only whitelist: the month letters are kept
    assert contracts.parse_date_fr("1 decembre 2005") == date(2005, 12, 1) and contracts.parse_date_fr("31/02/2024") is None
    assert contracts.parse_date_fr("l décembre 2OO5") is None and contracts.parse_date_fr("13/1O/2025") is None        # strict: a format check does not repair OCR slips


def test_invoice_date_with_a_month_name():
    def ln(words, y):
        out, x = [], 10
        for t in words.split():
            out.append({"text": t, "conf": 90.0, "x0": float(x), "x1": float(x + 8 * len(t)), "y0": float(y), "y1": float(y + 12)})
            x += 8 * len(t) + 8
        return out
    rows = ["Invoice Date: 1 decembre 2005", "Due Date: March 5, 2026", "Date de facture : 12 mars 2024"]
    for r, want in zip(rows, ["1 decembre 2005", "March 5, 2026", "12 mars 2024"]):
        got = postprocess.extract_pairs(postprocess._lines(ln(r, 10)))
        assert want in {v["text"] for v in got.values()}, (r, got)


def _cheque_png(date_text: str) -> bytes:
    F = lambda n: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{n}.ttf", 30)
    im = Image.new("RGB", (1500, 640), (235, 238, 250))
    d = ImageDraw.Draw(im)
    d.text((40, 30), "BANQUE ATLAS - Agence Rabat", font=F("DejaVuSans-Bold"), fill=(0, 0, 0))
    d.text((900, 40), "Date", font=F("DejaVuSans"), fill=(0, 0, 0))
    d.text((990, 30), date_text, font=F("DejaVuSerif"), fill=(20, 20, 120))
    d.text((40, 260), "Mille deux cents dirhams", font=F("DejaVuSerif"), fill=(20, 20, 120))
    d.text((1000, 250), "MAD", font=F("DejaVuSans-Bold"), fill=(0, 0, 0))
    d.text((1100, 250), "1 200,00", font=F("DejaVuSerif"), fill=(20, 20, 120))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _ocr_ready() -> bool:
    try:
        from documind.ocr import installed_langs
        return {"eng", "fra"} <= installed_langs() or "eng" in installed_langs()
    except Exception:
        return False


@pytest.mark.skipif(not _ocr_ready(), reason="needs Tesseract")
@pytest.mark.parametrize("written,expected", [("1 decembre 2005", "01/12/2005"), ("12 mars 2024", "12/03/2024"), ("December 1st, 2005", "01/12/2005")])
def test_cheque_with_a_written_date_end_to_end(written, expected):
    res = cheques.extract_cheque_bytes(_cheque_png(written), "c.png")                     # no detector here: the layout reader finds the date
    assert res["row"]["DATE"] == expected and res["checks"]["date_valid"] is True


@pytest.mark.skipif(not _ocr_ready(), reason="needs Tesseract")
def test_detector_box_with_a_written_date_is_read_not_dropped(monkeypatch):
    """The detector found the box; before, a digits-only whitelist turned "1 decembre 2005" into digits/empty. Now no whitelist is applied to a date box."""
    from documind.inference import date_reader
    seen = []
    real = date_reader._tesseract_box

    def spy(img, xyxy, psm, pad, scale, lang, whitelist):
        seen.append(whitelist)
        return real(img, xyxy, psm, pad, scale, lang, whitelist)

    monkeypatch.setattr(date_reader, "_tesseract_box", spy)
    img = Image.open(io.BytesIO(_cheque_png("1 decembre 2005")))
    got = cheques.fields_from_boxes(img, [("date", [980, 20, 1400, 80], 0.9)])
    assert seen and all(w is None for w in seen) and cheques.parse_date(got["date"]["raw_text"]) == ("01/12/2005", True), got
