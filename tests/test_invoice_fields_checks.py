"""Invoice fields found from their labels (French and English), the invoice checks, and the ruled-table reader on an invoice with a dark header and a page frame."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from documind.inference import invoice_yolo
from documind.inference.extractor import FieldExtractor
from documind.inference.fieldrules import fill_missing
from documind.inference.invoice_checks import parse_date, run_checks

sys.path.insert(0, str(Path(__file__).parent))
from make_ruled_invoice import make  # noqa: E402


def words_of(lines, cw=11, lh=34):
    words = []
    for i, (txt, x0, size) in enumerate(lines):
        x = x0
        for tok in txt.split(" "):
            w = len(tok) * cw * size
            words.append({"text": tok, "conf": 90.0, "x0": x, "y0": 40 + i * lh, "x1": x + w, "y1": 40 + i * lh + 22 * size})
            x += w + cw * size
    return words


FR = [("TechPro Solutions SARL", 40, 1.6), ("12 Rue des Orangers Casablanca", 40, 1), ("contact@techpro.ma www.techpro.ma", 40, 1), ("ICE : 001234567890123", 40, 1),
      ("FACTURE", 500, 1.8), ("Facture N° : FACT-2025-0027", 500, 1), ("Date d'émission : 18/09/2025", 500, 1), ("Date d'échéance : 18/10/2025", 500, 1),
      ("Client", 40, 1), ("Société Atlas SARL", 40, 1), ("ICE : 009876543210987", 40, 1),
      ("Arrêtée la présente facture à la somme de soixante-quatorze mille six cent vingt-trois dirhams", 40, 1),
      ("Conditions de paiement : 30 jours fin de mois", 40, 1),
      ("Coordonnées bancaires", 40, 1), ("IBAN : MA64 0111 5200 0001 2345 6789 0123", 40, 1), ("BIC : BCMAMAMC", 40, 1)]
EN = [("Invoice No. : INV-2025-0876", 660, 1), ("Issue Date : 18/09/2025", 660, 1), ("Due Date : 02/10/2025", 660, 1), ("Currency : MAD", 660, 1),
      ("Bill To :", 66, 1), ("Knowvault Corp.", 66, 1), ("456 Innovation Street, Rabat, Morocco", 66, 1), ("VAT Number : 123456789012345", 66, 1),
      ("Payment Terms", 120, 1), ("Net 14 days from the invoice date.", 120, 1)]


def got(lines, **kw):
    return {f["label"]: f["text"] for f in fill_missing([], words_of(lines), 1, **kw)}


def test_french_invoice_fields_from_labels():
    g = got(FR)
    assert g["NUMBER"] == "FACT-2025-0027" and g["DATE"] == "18/09/2025" and g["DUE_DATE"] == "18/10/2025" and g["TITLE"] == "FACTURE"
    assert g["BUYER"] == "Société Atlas SARL" and g["SELLER_NAME"] == "TechPro Solutions SARL" and g["SELLER_EMAIL"] == "contact@techpro.ma" and g["SELLER_SITE"] == "www.techpro.ma"
    assert g["GSTIN_SELLER"] == "001234567890123" and g["GSTIN_BUYER"] == "009876543210987"
    assert g["TOTAL_WORDS"] == "soixante-quatorze mille six cent vingt-trois dirhams" and g["CONDITIONS"] == "30 jours fin de mois"
    assert g["PAYMENT_DETAILS"].startswith("Coordonnées bancaires IBAN : MA64") and g["PAYMENT_DETAILS"].endswith("BCMAMAMC")


def test_english_invoice_fields_and_no_invention():
    g = got(EN)
    assert g["NUMBER"] == "INV-2025-0876" and g["DUE_DATE"] == "02/10/2025" and g["BUYER"] == "Knowvault Corp." and g["CONDITIONS"] == "Net 14 days from the invoice date."
    assert "SELLER_NAME" not in g and "SELLER_EMAIL" not in g and "TOTAL_WORDS" not in g                  # no label, no pattern: nothing is invented
    assert got([("Thank you for your business", 40, 1), ("Page 1 of 2", 40, 1)]) == {}


def test_detector_fields_are_never_overridden_and_only_known_classes_are_added():
    det = [{"label": "DATE", "text": "01/01/2020", "bbox": [0, 0, 1, 1], "page": 1}]
    out = {f["label"]: f["text"] for f in fill_missing(det, words_of(FR), 1, classes=["DATE", "NUMBER"])}
    assert "DATE" not in out and out == {"NUMBER": "FACT-2025-0027"}


def test_two_columns_on_one_line_are_not_mixed():
    words = words_of([("Bill To :", 40, 1)]) + [{"text": t, "conf": 90.0, "x0": x, "y0": 40, "x1": x + 60, "y1": 62} for t, x in (("Invoice", 600), ("No.", 670), (":", 720), ("INV-77", 740))]
    words += [{"text": "Knowvault", "conf": 90.0, "x0": 40, "y0": 74, "x1": 140, "y1": 96}, {"text": "Corp.", "conf": 90.0, "x0": 150, "y0": 74, "x1": 200, "y1": 96}]
    g = {f["label"]: f["text"] for f in fill_missing([], words, 1)}
    assert g["BUYER"] == "Knowvault Corp." and g["NUMBER"] == "INV-77"


# ---------------------------------------------------------------- checks
def F(**k):
    return [{"label": a, "text": b} for a, b in k.items()]


def test_dates():
    assert str(parse_date("18/09/2025")) == "2025-09-18" and str(parse_date("18 septembre 2025")) == "2025-09-18" and str(parse_date("Sep 18, 2025")) == "2025-09-18"
    assert parse_date("31/02/2025") is None and parse_date("no date") is None


def test_checks_pass_on_a_consistent_french_invoice():
    r = run_checks(F(NUMBER="F-1", DATE="18/09/2025", DUE_DATE="18/10/2025", SUB_TOTAL="63 190,00", TAX="11 433,00", TOTAL="74 623,00", SELLER_EMAIL="a@b.ma",
                     TOTAL_WORDS="soixante-quatorze mille six cent vingt-trois dirhams", PAYMENT_DETAILS="IBAN GB82 WEST 1234 5698 7654 32"))
    assert r["warnings"] == [] and r["checks"]["total_arithmetic"] and r["checks"]["amount_in_words"] and r["checks"]["dates_valid"] and r["checks"]["due_after_issue"]
    assert r["checks"]["required_fields"] and r["checks"]["email_valid"] and r["checks"]["iban_valid"] is True and r["checks"]["tax_rate_consistent"]


def test_checks_flag_what_is_wrong():
    r = run_checks(F(NUMBER="F-1", DATE="18/10/2025", DUE_DATE="18/09/2025", SUB_TOTAL="100,00", TAX="20,00", TOTAL="150,00", AMOUNT_DUE="900,00",
                     TOTAL_WORDS="cent vingt euros", SELLER_EMAIL="not-an-email", PAYMENT_DETAILS="IBAN GB82 WEST 1234 5698 7654 33"),
                    [{"description": "x", "qty": "2", "unit_price": "10,00", "amount": "50,00"}], {"items_sum": 50.0})
    c = r["checks"]
    assert c["total_arithmetic"] is False and c["due_after_issue"] is False and c["amount_in_words"] is False and c["amount_due_vs_total"] is False
    assert c["rows_arithmetic"] is False and c["items_sum"] is False and c["email_valid"] is False and c["iban_valid"] is False
    assert {w["check"] for w in r["warnings"]} >= {"total_arithmetic", "due_after_issue", "amount_in_words", "rows_arithmetic", "iban_valid"}


def test_checks_say_none_when_a_field_is_missing_and_ignore_computed_values():
    r = run_checks(F(NUMBER="F-1"))
    assert r["checks"]["total_arithmetic"] is None and r["checks"]["required_fields"] is False and r["checks"]["dates_valid"] is None
    fields = F(SUB_TOTAL="100,00", TOTAL="9999,00") + [{"label": "TAX", "text": "9899,00", "source": "computed"}]
    assert run_checks(fields)["checks"]["total_arithmetic"] is None                       # a tax computed from the total cannot check the total


def test_tax_included_row_is_consistent():
    it = [{"description": "x", "qty": "1", "unit_price": "25 000,00", "vat_amount": "5 000,00", "vat_rate_pct": "20%", "amount": "30 000,00"}]
    assert run_checks(F(NUMBER="1", DATE="1/1/2025", TOTAL="30 000,00"), it, {"items_sum": 30000.0})["checks"]["rows_arithmetic"] is True


# ---------------------------------------------------------------- the ruled table with a dark header, and a page frame
def test_grid_is_found_even_when_a_page_frame_is_the_longest_line():
    from documind.inference import tableprep as tp
    for framed in (False, True):
        img = cv2.cvtColor(np.asarray(make(page_border=framed)), cv2.COLOR_RGB2BGR)
        g, _s, _a = tp.flat(img)
        res = tp.grid(g)
        assert res is not None and len(res[0]) >= 20, framed                                # 4 rows x 7 columns


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs Tesseract")
@pytest.mark.parametrize("framed", [False, True])
def test_all_rows_of_a_ruled_table_with_dark_header_are_read(monkeypatch, tmp_path, framed):
    class Fake:
        names = {0: "NUMBER"}
        overrides = {"imgsz": 1280}

        def predict(self, *a, **k):
            return [type("R", (), {"boxes": []})()]

    monkeypatch.setattr(invoice_yolo, "model_available", lambda path=None: True)
    monkeypatch.setattr(invoice_yolo, "load_model", lambda path=None: Fake())
    path = tmp_path / "inv.png"
    make(page_border=framed).save(path)
    r = FieldExtractor(lang="eng", psm=6).extract_bytes(path.read_bytes(), "inv.png")
    amounts = [it["amount"] for it in r["items"]]
    assert amounts == ["30 000,00", "6 000,00", "6 000,00"], amounts
    assert [it["unit_price"] for it in r["items"]] == ["25 000,00", "5 000,00", "500,00"]
    assert [it["description"].split(" (")[0].strip().lower()[:12] for it in r["items"]][2].startswith("consulting")
    assert "vat_amount" in r["items"][0] and r["checks"]["required_fields"] is not None


# ---------------------------------------------------------------- confidence from evidence, not a constant
def test_confidence_comes_from_evidence():
    from documind.inference.confidence import evidence_confidence, valid_format
    good = [{"text": "x", "conf": 95.0}]
    blurry = [{"text": "x", "conf": 40.0}]
    assert evidence_confidence(good, True) > evidence_confidence(good, None) > evidence_confidence(good, False)
    assert evidence_confidence(good, True) > evidence_confidence(blurry, True)                   # a better read is a surer reading
    assert evidence_confidence(good, True, labelled=True) > evidence_confidence(good, True, labelled=False)
    assert 0.05 <= evidence_confidence(blurry, False, False) < evidence_confidence(good, True) <= 0.92
    assert valid_format("DATE", "18/09/2025") is True and valid_format("DATE", "31/02/2025") is False and valid_format("SELLER_EMAIL", "a@b.ma") is True and valid_format("BUYER", "Atlas") is None


def test_rule_fields_no_longer_share_one_fixed_confidence():
    confs = {f["label"]: f["confidence"] for f in fill_missing([], words_of(FR), 1)}
    assert len(set(confs.values())) >= 3 and confs["DATE"] > confs["SELLER_NAME"]                  # a valid date beats a guessed company name


def test_checks_raise_confirmed_fields_and_lower_contradicted_ones():
    from documind.inference.invoice_checks import adjust_confidence
    fields = [{"label": "SUB_TOTAL", "text": "100,00", "confidence": 0.7}, {"label": "TAX", "text": "20,00", "confidence": 0.7},
              {"label": "TOTAL", "text": "120,00", "confidence": 0.7}, {"label": "DATE", "text": "01/01/2025", "confidence": 0.7},
              {"label": "DUE_DATE", "text": "01/12/2024", "confidence": 0.7}]
    adjust_confidence(fields, run_checks(fields)["checks"])
    by = {f["label"]: f for f in fields}
    assert by["TOTAL"]["confidence"] == 0.78 and by["TOTAL"]["verified_by"] == ["total_arithmetic"]
    assert by["DUE_DATE"]["confidence"] == 0.55 and by["DUE_DATE"]["contradicted_by"] == ["due_after_issue"]
    computed = [{"label": "TAX", "text": "20,00", "confidence": 0.5, "source": "computed"}, {"label": "SUB_TOTAL", "text": "100,00", "confidence": 0.7}, {"label": "TOTAL", "text": "120,00", "confidence": 0.7}]
    adjust_confidence(computed, run_checks(computed)["checks"])
    assert computed[0]["confidence"] == 0.5                                                        # a computed value is not confirmed by the check it was computed for


def test_buyer_vat_number_is_the_buyers():
    lines = [("Bill To :", 66, 1), ("Knowvault Corp.", 66, 1), ("456 Innovation Street,", 66, 1), ("Rabat, Morocco", 66, 1), ("VAT Number : 123456789012345", 66, 1), ("Invoice No. : INV-1", 660, 1)]
    g = got(lines)
    assert g.get("GSTIN_BUYER") == "123456789012345" and "GSTIN_SELLER" not in g
