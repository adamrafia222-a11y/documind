"""English AND French documents: labels in both languages are understood, and a missing Tesseract language pack does not blank the page."""
from __future__ import annotations

import pytest

from documind.inference.postprocess import _key_of, _lines, extract_pairs, extract_totals
from documind.ocr import engine


def _rows(rows: list[str]) -> list[dict]:
    words, y = [], 10
    for r in rows:
        x = 10
        for t in r.split():
            words.append({"text": t, "conf": 90, "x0": x, "x1": x + 8 * len(t), "y0": y, "y1": y + 12})
            x += 8 * len(t) + 8
        y += 30
    return words


@pytest.mark.parametrize("text,key", [
    ("Subtotal 100.00", "SUB_TOTAL"), ("Sous-total 100,00", "SUB_TOTAL"), ("Total HT : 100,00", "SUB_TOTAL"),
    ("Tax 5.00", "TAX"), ("TVA 20% 20,00", "TAX"), ("Taxe 5,00", "TAX"),
    ("Total 120.00", "TOTAL"), ("Total TTC 120,00", "TOTAL"), ("Montant TTC 120,00", "TOTAL"),
    ("Discount 5.00", "DISCOUNT"), ("Remise 5,00", "DISCOUNT"),
    ("Amount due 50.00", "AMOUNT_DUE"), ("Net à payer 50,00", "AMOUNT_DUE"), ("Net a payer 50,00", "AMOUNT_DUE"),
])
def test_amount_labels_in_english_and_french(text, key):
    assert _key_of(text) == key


def test_french_totals_block():
    t = extract_totals(_lines(_rows(["Sous-total 100,00", "TVA 20% 20,00", "Total TTC 120,00"])))
    assert {k: v["text"] for k, v in t.items()} == {"SUB_TOTAL": "100,00", "TAX": "20,00", "TOTAL": "120,00"}


def test_key_value_pairs_in_french_and_english():
    fr = extract_pairs(_lines(_rows(["Facture N° FA-2026-014", "Date de facture : 12/11/2026", "Date d'échéance : 12/12/2026", "Bon de commande 4587"])))
    assert {k: v["text"] for k, v in fr.items()} == {"NUMBER": "FA-2026-014", "DATE": "12/11/2026", "DUE_DATE": "12/12/2026", "PO_NUMBER": "4587"}
    en = extract_pairs(_lines(_rows(["Invoice No: INV-77", "Invoice Date: 12/11/2026", "Due Date: 12/12/2026", "PO Number 4587"])))
    assert {k: v["text"] for k, v in en.items()} == {"NUMBER": "INV-77", "DATE": "12/11/2026", "DUE_DATE": "12/12/2026", "PO_NUMBER": "4587"}


def test_missing_language_pack_is_skipped_not_fatal(monkeypatch):
    monkeypatch.setattr(engine, "_WARNED", set())
    monkeypatch.setattr(engine, "_INSTALLED", {"eng", "osd"})
    with pytest.warns(UserWarning, match="fra"):
        assert engine.resolve_lang("eng+fra") == "eng"
    monkeypatch.setattr(engine, "_INSTALLED", {"eng", "fra", "osd"})
    assert engine.resolve_lang("eng+fra") == "eng+fra"
    monkeypatch.setattr(engine, "_INSTALLED", {"fra", "osd"})
    assert engine.resolve_lang("eng+ara") == "fra"          # nothing requested is installed: read with what exists
    monkeypatch.setattr(engine, "_INSTALLED", set())          # packs cannot be listed: pass the request through
    assert engine.resolve_lang("eng+fra") == "eng+fra"
