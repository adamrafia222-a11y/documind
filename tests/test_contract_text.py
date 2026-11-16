"""Loan contracts in English or French are read from their text (contract_text.py) when the YOLO detector misses them or was not trained on that layout."""
from __future__ import annotations

from pathlib import Path

import pytest

from documind.inference import contracts
from documind.inference.contract_text import english_words_to_number, parse_terms, read_contract_text, to_float
from documind.inference.router import score_types

DATA = Path(__file__).parent / "data" / "loan_agreement_en.jpg"

EN = ["Loan Agreement", "Contract n° LA20261114-001 Date: 14/11/2026", "The Lender: Ahmed El Mansouri, holder of ID card n° CD123456, domiciled at 12 Avenue Hassan II, Rabat, on the one part, and",
      "The Borrower: Mohamed Benali, holder of ID card n° BB987654, domiciled at 45 Rue des Fleurs, Casablanca, Morocco, hereinafter referred to as the Borrower, on the other part,",
      "The Lender hereby agrees to lend the Borrower the sum of 20,000 MAD (twenty thousand Moroccan Dirhams), destined for the purchase of a vehicle.",
      "This loan is granted for a period of 6 months, starting from November 14, 2026. The loan is interest-free."]
FR = ["CONTRAT DE PRÊT PERSONNEL", "Contrat n° CP-2024-0087 Date : 12/03/2024", "Le prêteur : Banque Atlas, représentée par Karim Alaoui, d'une part, et",
      "L'emprunteur : Salma Idrissi, titulaire de la CIN n° AB123456, demeurant à 12 Rue des Roses, Casablanca, d'autre part.",
      "Le prêteur accorde la somme de 100 000,00 MAD (cent mille dirhams). Durée de 60 mois au taux annuel de 6 %.",
      "La mensualité est de 1 933,28 MAD, la première échéance étant le 12/04/2024. IBAN : MA64 1234 5678 9012 3456 7890 1234"]


def _words(rows: list[str]) -> list[dict]:
    out, y = [], 20
    for r in rows:
        x = 20
        for t in r.split():
            out.append({"text": t, "conf": 92.0, "x0": x, "x1": x + 9 * len(t), "y0": y, "y1": y + 16})
            x += 9 * len(t) + 9
        y += 40
    return out


def _raw(rows):
    f, na = parse_terms(_words(rows), 1500.0, 1000.0)
    return {k: v["raw_text"] for k, v in f.items()}, na


def test_numbers_with_english_or_french_separators():
    assert [to_float(t) for t in ("20,000", "20 000,50", "100.000,00", "1,250.50", "450", "1 933,28")] == [20000.0, 20000.5, 100000.0, 1250.5, 450.0, 1933.28]
    assert english_words_to_number("twenty thousand Moroccan Dirhams") == (20000, 0, "MAD")
    assert english_words_to_number("one hundred and fifty euros") == (150, 0, "EUR") and english_words_to_number("blah blah") is None


def test_english_terms():
    f, na = _raw(EN)
    assert f["title"] == "Loan Agreement" and f["contract_number"] == "LA20261114-001" and f["contract_date"] == "14/11/2026"
    assert (f["lender_name"], f["borrower_name"], f["borrower_id"]) == ("Ahmed El Mansouri", "Mohamed Benali", "BB987654")
    assert f["borrower_address"] == "45 Rue des Fleurs, Casablanca, Morocco"
    assert (f["amount"], f["amount_words"], f["duration"], f["interest_rate"]) == ("20000.00 MAD", "twenty thousand Moroccan Dirhams", "6 months", "0 %")
    assert na == {"monthly_payment", "iban"}                 # a single-repayment loan has neither: not "missing"


def test_french_terms():
    f, na = _raw(FR)
    assert (f["lender_name"], f["borrower_name"], f["borrower_id"]) == ("Banque Atlas", "Salma Idrissi", "AB123456")
    assert (f["amount"], f["duration"], f["interest_rate"], f["monthly_payment"]) == ("100000.00 MAD", "60 mois", "6 %", "1933.28 MAD")
    assert f["first_payment_date"] == "12/04/2024" and f["iban"].startswith("MA64") and na == set()


def test_english_dates_durations_and_words_are_checked():
    assert contracts.parse_date_fr("November 14, 2026").isoformat() == "2026-11-14" and contracts.parse_date_fr("14 November 2026").isoformat() == "2026-11-14"
    assert contracts.parse_date_fr("2026-11-14").isoformat() == "2026-11-14" and contracts.parse_date_fr("12 mars 2022").isoformat() == "2022-03-12"
    assert (contracts.parse_duration("6 months"), contracts.parse_duration("2 years"), contracts.parse_duration("5 ans")) == (6, 24, 60)
    f, na = parse_terms(_words(EN), 1500.0, 1000.0)
    got = {k: {**v, "value": v["raw_text"]} for k, v in f.items()}
    r = contracts.interpret(got, na)
    assert r["checks"]["amount_match"] is True and r["values"]["rate"] == "0.00" and r["values"]["duration_months"] == "6"
    assert "missing_field" not in {w["code"] for w in r["warnings"]}


def test_detector_garbage_is_replaced_by_the_text_reading():
    f, _ = parse_terms(_words(EN), 1500.0, 1000.0)
    det = {"amount": {"value": "xx", "raw_text": "xx", "box_confidence": 0.9, "bbox": [0, 0, 1, 1], "n_boxes": 1},
           "lender_name": {"value": "Ahmed El Mansouri", "raw_text": "Ahmed El Mansouri", "box_confidence": 0.9, "bbox": [0, 0, 1, 1], "n_boxes": 1}}
    m = contracts.merge_readings(det, f)
    assert m["amount"]["raw_text"] == "20000.00 MAD" and m["lender_name"]["box_confidence"] == 0.9 and "duration" in m      # kept / repaired / added


def test_english_contract_is_routed_to_the_contract_reader():
    s = score_types("Loan Agreement between the undersigned: the Lender and the Borrower, hereinafter referred to", 1.41)
    assert max(s, key=s.get) == "contract"


def _ocr_ready() -> bool:
    try:
        from documind.ocr import installed_langs
        return "eng" in installed_langs()
    except Exception:
        return False


@pytest.mark.skipif(not (_ocr_ready() and DATA.exists()), reason="needs Tesseract (eng) and tests/data/loan_agreement_en.jpg")
def test_real_english_loan_agreement_and_its_signatures():
    from PIL import Image
    img = Image.open(DATA)
    r = read_contract_text(img)
    f = {k: v["raw_text"] for k, v in r["fields"].items()}
    assert (f["amount"], f["duration"], f["interest_rate"], f["borrower_name"]) == ("20000.00 MAD", "6 months", "0 %", "Mohamed Benali")
    assert {"signature_lender", "signature_borrower"} <= set(r["fields"]) and r["not_applicable"] == {"monthly_payment", "iban"}
    import numpy as np
    arr = np.array(img.convert("RGB"))
    for k in ("signature_lender", "signature_borrower"):                                   # erase the handwriting: no signature any more
        x0, y0, x1, y1 = (int(v) for v in r["fields"][k]["bbox"])
        arr[y0 - 3: y1 + 3, x0 - 3: x1 + 3] = 255
    assert not [k for k in read_contract_text(Image.fromarray(arr))["fields"] if k.startswith("signature")]
