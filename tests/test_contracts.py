"""Loan contracts: reading and checking the key terms, the contract dataset preprocessing, the fourth document type in the router / API (one CSV per type),
the OCR crops of wrapped values. No Tesseract, Ultralytics or trained model needed: the detector and the OCR are replaced by small fakes.
    pytest -q tests/test_contracts.py
"""
import importlib.util
import io
import random
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from documind.api import accounts, main
from documind.data import prepare_contracts
from documind.data.prepare_receipts import run
from documind.inference import cheques, contracts, receipts, router, service
from documind.ocr import crops, lines
from documind.training import train_cheques, train_contracts

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "generate_contracts.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_router_service import FakeInvoice, FakeYolo, OCR_TEXT, _png                      # noqa: E402  (fakes of the receipt / invoice tests)
from test_cheques import CHEQUE_BOXES, CHEQUE_TEXT, FakeChequeYolo                          # noqa: E402


# ------------------------------------------------------------------------------------------------ numbers and dates
def test_amount_in_words_understands_millions():
    w = cheques.words_to_number
    assert w("Un million de dirhams") == (1_000_000, 0, "MAD") and w("deux millions cinq cent mille euros") == (2_500_000, 0, "EUR")
    assert w("un million cinq cent mille dirhams et cinquante centimes") == (1_500_000, 50, "MAD")


@pytest.mark.skipif(not GENERATOR.exists(), reason="scripts/generate_contracts.py not present")
def test_words_round_trip_the_contract_generators_own_words():
    spec = importlib.util.spec_from_file_location("generate_contracts", GENERATOR)
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    rng = random.Random(0)
    for v in list(range(0, 800)) + [rng.randint(0, 999_999_999) for _ in range(1500)] + [1_000_000, 2_000_000, 80_000_000, 1_000_080]:
        for cur, name in (("MAD", "dirham"), ("EUR", "euro")):
            assert cheques.words_to_number(gc.amount_words(v, 0, name)) == (v, 0, cur), gc.amount_words(v, 0, name)


def test_parsers():
    assert contracts.parse_date_fr("12 mars 2022") == date(2022, 3, 12) and contracts.parse_date_fr("1er août 2020") == date(2020, 8, 1)
    assert contracts.parse_date_fr("05/12/24") == date(2024, 12, 5) and contracts.parse_date_fr("31/02/2024") is None and contracts.parse_date_fr("n'importe quoi") is None
    assert contracts.parse_duration("36 mois") == 36 and contracts.parse_duration("5 ans") == 60 and contracts.parse_rate("5,45 %") == 5.45
    assert contracts.currency_of("150 000,00 DH") == "MAD" and contracts.currency_of("1 200,00 €") == "EUR" and contracts.currency_of("120") == ""
    assert contracts.contract_type("CONTRAT DE PRÊT IMMOBILIER") == "mortgage" and contracts.contract_type("CONTRAT DE CRÉDIT AUTOMOBILE") == "car"
    assert contracts.contract_type("CONTRAT DE PRET PERSONNEL") == "personal" and contracts.clean_iban("ma64 1234 5678") == "MA6412345678"
    assert round(contracts.annuity(100000, 6.0, 60), 2) == 1933.28 and contracts.annuity(1200, 0, 12) == 100.0
    assert contracts.clean_edges(" Khadija AMRANI, ") == "Khadija AMRANI" and contracts.clean_edges("(5,45 %)") == "5,45 %"


@pytest.mark.skipif(not GENERATOR.exists(), reason="scripts/generate_contracts.py not present")
def test_loan_formula_matches_the_generators_monthly_payment(tmp_path):
    subprocess.run([sys.executable, str(GENERATOR), "--n", "8", "--out", str(tmp_path / "d"), "--workers", "1"], check=True, capture_output=True)
    import json
    for line in (tmp_path / "d" / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)["ground_truth"]
        assert round(contracts.annuity(t["amount_value"], t["rate_value"], t["months"]), 2) == t["monthly_value"]


# ------------------------------------------------------------------------------------------------ the checks
def _got(**raw) -> dict:
    base = {"title": "CONTRAT DE PRÊT PERSONNEL", "contract_number": "CP-2024-123456", "contract_date": "12 mars 2024", "lender_name": "Banque Atlas",
            "borrower_name": "Khadija AMRANI", "borrower_id": "BE82421", "borrower_address": "12 Rue Ibn Sina, 40000 Marrakech", "amount": "100 000,00 MAD",
            "amount_words": "cent mille dirhams", "interest_rate": "6,00 %", "duration": "60 mois", "monthly_payment": "1 933,28 MAD", "first_payment_date": "12 avril 2024",
            "iban": "MA64 1234 5678 9012 3456 7890 1234", "place": "Marrakech", "signing_date": "12 mars 2024", "signature_lender": "", "signature_borrower": "", "stamp": ""}
    base.update(raw)
    return {k: {"value": v, "raw_text": v, "box_confidence": 0.9, "bbox": [0, 0, 1, 1], "n_boxes": 1} for k, v in base.items()}


def test_consistent_contract_has_no_warning():
    r = contracts.interpret(_got())
    assert r["warnings"] == [] and r["checks"]["amount_match"] is True and r["checks"]["payment_match"] is True and r["checks"]["iban_length_ok"] is True
    v = r["values"]
    assert (v["amount"], v["currency"], v["rate"], v["duration_months"], v["monthly_payment"], v["type"]) == ("100000.00", "MAD", "6.00", "60", "1933.28", "personal")
    assert v["first_payment_date"] == "2024-04-12" and v["contract_date"] == "2024-03-12" and v["iban"] == "MA64123456789012345678901234"


def test_each_check_reports_its_own_problem():
    code = lambda **kw: {w["code"] for w in contracts.interpret(_got(**kw))["warnings"]}
    assert code(amount="150 000,00 MAD") >= {"amount_mismatch", "payment_mismatch"}
    assert "payment_mismatch" in code(monthly_payment="2 500,00 MAD") and "payment_mismatch" not in code(monthly_payment="1 940,00 MAD")      # 0.4 % off: insurance and fees
    assert "amount_not_checked" in code(amount_words="xx yy zz") and "iban_length" in code(iban="MA64 1234 5678")
    assert "first_payment_early" in code(first_payment_date="01 mars 2024") and "signed_before_contract" in code(signing_date="01 mars 2024")
    assert "date_invalid" in code(signing_date="31/02/2024")
    got = _got()
    del got["signature_borrower"], got["iban"]
    assert {"no_signature_borrower", "missing_field"} <= {w["code"] for w in contracts.interpret(got)["warnings"]}


# ------------------------------------------------------------------------------------------------ boxes of one field -> one text
def test_wrapped_value_is_read_in_reading_order_even_on_a_tilted_page(monkeypatch):
    """'150 000,00 MAD' wrapped: '150 000,00' ends line 1 on the right, 'MAD' starts line 2 on the left. The page is tilted so that the right end of
    a line is lower than the left end of the next one: sorting by y alone would put 'MAD' first."""
    texts = {(700, 100, 900, 125): "150 000,00", (100, 90, 150, 115): "MAD"}       # line 2 sits ABOVE line 1 in the picture (tilt), but is read second
    monkeypatch.setattr(contracts, "ocr_box", lambda img, xyxy, **kw: texts[tuple(int(v) for v in xyxy)])
    dets = [("amount", [100, 90, 150, 115], 0.9), ("amount", [700, 100, 900, 125], 0.8), ("lender_name", [10, 10, 200, 40], 0.6), ("lender_name", [10, 300, 200, 330], 0.95),
            ("signature_lender", [10, 500, 100, 560], 0.7)]
    texts[(10, 10, 200, 40)], texts[(10, 300, 200, 330)] = "Banque Atl", "Banque Atlas,"
    # the lines slope down to the right by about 3.6 degrees (right end 43 px lower over 675 px): the deskew angle is +3.6 (counter-clockwise, as estimate_skew)
    got = contracts.fields_from_boxes(Image.new("RGB", (1000, 800), "white"), dets, angle=3.6)
    assert got["amount"]["value"] == "150 000,00 MAD" and got["amount"]["n_boxes"] == 2 and got["amount"]["box_confidence"] == 0.8
    assert got["lender_name"]["value"] == "Banque Atlas" and got["signature_lender"]["value"] == "signed"                     # the more confident copy of the name; edges cleaned


def test_reading_order_and_split_value():
    boxes = [[700, 100, 900, 125], [100, 90, 150, 115]]
    assert lines.reading_order(boxes, 0.0, (1000, 800)) == [1, 0]                       # no tilt correction: plain top to bottom
    assert lines.reading_order([[700, 100, 900, 125], [100, 128, 150, 153]], 0.0, (1000, 800)) == [0, 1]
    assert lines.split_value("150 000,00 MAD", [190, 58]) == ["150 000,00", "MAD"] and lines.split_value("150 000,00 MAD", [60, 190]) == ["150", "000,00 MAD"]
    assert lines.split_value("a b", [50, 50]) is None and lines.split_value("seul", [50, 50]) is None                # a guess is never returned


def test_contract_texts_for_ocr_crops():
    class Ctx:
        size = (1000, 800)

        def angle(self):
            return 0.0
    truth = {"title": "CONTRAT DE PRÊT", "lender_name": "Banque Atlas", "amount": "150 000,00 MAD", "iban": "MA64 1234", "interest_rate": "5,45 %"}
    wrapped = [("title", (0, 0, 9, 9)), ("lender_name", (0, 0, 9, 9)), ("lender_name", (0, 20, 9, 29)), ("amount", (700, 100, 890, 125)), ("amount", (100, 130, 148, 155)),
               ("interest_rate", (0, 0, 9, 9)), ("stamp", (0, 0, 9, 9)), ("signature_lender", (0, 0, 9, 9))]
    got = [(c, t) for c, _, t in crops.contract_texts(truth, wrapped, Ctx())]
    assert got == [("title", "CONTRAT DE PRÊT"), ("lender_name", "Banque Atlas"), ("lender_name", "Banque Atlas"), ("amount", "150 000,00"), ("amount", "MAD"), ("interest_rate", "5,45 %")]
    three = [("lender_name", (0, 0, 9, 9))] * 3 + [("iban", (0, 0, 9, 9)), ("iban", (0, 0, 9, 9)), ("iban", (0, 0, 9, 9)), ("iban", (0, 0, 9, 9))]
    assert crops.contract_texts(truth, three, Ctx()) == []                               # 3 name boxes / 4 iban boxes: not understood, nothing guessed


# ------------------------------------------------------------------------------------------------ dataset preparation and training profile
@pytest.mark.skipif(not GENERATOR.exists(), reason="scripts/generate_contracts.py not present")
def test_prepare_contracts_reads_the_generators_dataset(tmp_path):
    raw = tmp_path / "raw"
    subprocess.run([sys.executable, str(GENERATOR), "--n", "10", "--out", str(raw), "--workers", "1"], check=True, capture_output=True)
    stats = run(raw, tmp_path / "out", workers=1, reports_dir=tmp_path / "rep", tag="contracts", default_names=prepare_contracts.CONTRACT_NAMES, max_side=1800)
    assert sum(v["images"] for v in stats["splits"].values()) == 10 and list(stats["names"].values()) == prepare_contracts.CONTRACT_NAMES and len(stats["names"]) == 19
    assert (tmp_path / "out" / "data.yaml").exists() and (tmp_path / "rep" / "contracts_prepare_report.txt").exists() and (tmp_path / "rep" / "contracts_stats.json").exists()
    assert not (tmp_path / "rep" / "cheques_stats.json").exists() and not (tmp_path / "rep" / "receipts_stats.json").exists()      # reports of the other types are never overwritten
    assert stats["splits"]["train"]["per_class"]["lender_name"] == 2 * stats["splits"]["train"]["images"]                         # the name is printed twice: two boxes per page are kept


def test_find_raw_contracts_falls_back_to_the_generators_folder(tmp_path):
    (tmp_path / "contract_dataset" / "images" / "train").mkdir(parents=True)
    assert prepare_contracts.find_raw_contracts(tmp_path) == tmp_path / "contract_dataset"
    assert prepare_contracts.find_raw_contracts(tmp_path / "nothing") == tmp_path / "nothing" / "contracts"


def test_contract_training_profile():
    c, q = train_contracts.profile(), train_cheques.profile()
    assert c.kind == "contracts" and c.model_out.name == "contracts_yolo.pt" and c.model_out != q.model_out and c.report_path.name == "contracts_report.json"
    assert c.imgsz == 1280 and c.batch == 4 and c.aug["mosaic"] == 0.0 and c.aug["fliplr"] == 0.0 and "prepare_contracts" in c.prepare_cmd
    from documind.training import receipts_eval
    assert {"amount", "interest_rate", "stamp"} <= set(receipts_eval.SINGLE) and "iban" not in receipts_eval.SINGLE and "borrower_address" not in receipts_eval.SINGLE


# ------------------------------------------------------------------------------------------------ router and API: four types, four CSV files
def test_router_scores_the_contract_page():
    text = "CONTRAT DE PRET PERSONNEL ENTRE LES SOUSSIGNES Banque Atlas ci-apres denommee le Preteur Article 1 Taux d'interet TAEG mensualites Lu et approuve"
    s = router.score_types(text, aspect=1.41)
    assert s["contract"] > 10 and s["contract"] > s["invoice"] and s["contract"] > s["cheque"] and s["contract"] > s["receipt"]
    assert router.score_types("CONTRAT DE PRÊT PERSONNEL Prêteur Emprunteur", 1.41)["contract"] == router.score_types("CONTRAT DE PRET PERSONNEL Preteur Emprunteur", 1.41)["contract"]      # accents do not matter
    inv = router.score_types("TAX INVOICE Bill To Invoice No Unit price Qty Due date", aspect=1.41)
    assert inv["invoice"] > inv["contract"]
    assert router.score_types("Payez contre ce cheque a l'ordre de", aspect=0.44)["cheque"] > router.score_types("Payez contre ce cheque a l'ordre de", aspect=0.44)["contract"]


CONTRACT_BOXES = [("title", [100, 40, 400, 70]), ("contract_number", [420, 10, 560, 30]), ("contract_date", [420, 35, 560, 55]), ("lender_name", [20, 10, 200, 35]),
                  ("lender_name", [20, 120, 130, 140]), ("borrower_name", [20, 150, 180, 170]), ("borrower_id", [200, 150, 260, 170]), ("borrower_address", [20, 180, 400, 200]),
                  ("amount", [300, 300, 450, 320]), ("amount", [20, 325, 60, 345]), ("amount_words", [20, 350, 400, 370]), ("interest_rate", [300, 400, 360, 420]),
                  ("duration", [300, 450, 360, 470]), ("monthly_payment", [300, 500, 430, 520]), ("first_payment_date", [20, 550, 160, 570]), ("iban", [20, 580, 420, 600]),
                  ("place", [20, 700, 100, 720]), ("signing_date", [140, 700, 260, 720]), ("signature_lender", [20, 740, 200, 800]), ("signature_borrower", [300, 740, 480, 800]),
                  ("stamp", [60, 730, 160, 830])]
CONTRACT_TEXT = {"title": "CONTRAT DE PRÊT PERSONNEL", "contract_number": "CP-2024-123456", "contract_date": "12 mars 2024", "borrower_name": "Khadija AMRANI", "borrower_id": "BE82421",
                 "borrower_address": "12 Rue Ibn Sina, 40000 Marrakech", "amount_words": "cent mille dirhams", "interest_rate": "6,00 %", "duration": "60 mois",
                 "monthly_payment": "1 933,28 MAD", "first_payment_date": "12 avril 2024", "iban": "MA64 1234 5678 9012 3456 7890 1234", "place": "Marrakech", "signing_date": "12 mars 2024"}


class FakeContractYolo:
    names = dict(enumerate(contracts.FIELD_LABELS))
    overrides = {"imgsz": 1280}

    def predict(self, img, imgsz=1280, conf=0.25, verbose=False):
        class B:
            def __init__(s, cls, xyxy):
                s.cls, s.xyxy, s.conf = cls, np.array([xyxy], dtype=float), 0.9

        class R:
            boxes = [B(list(contracts.FIELD_LABELS).index(n), b) for n, b in CONTRACT_BOXES]
        return [R()]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    by_box = {}
    for n, b in CONTRACT_BOXES:
        by_box[tuple(b)] = {"lender_name": "Banque Atlas", "amount": "100 000,00" if b[0] == 300 else "MAD"}.get(n, CONTRACT_TEXT.get(n, ""))
    monkeypatch.setattr(contracts, "_model", FakeContractYolo())
    monkeypatch.setattr(contracts, "model_available", lambda path=contracts.MODEL_PATH: True)
    monkeypatch.setattr(contracts, "ocr_box", lambda img, xyxy, **kw: by_box.get(tuple(int(v) for v in xyxy), ""))
    monkeypatch.setattr(cheques, "_model", FakeChequeYolo())
    monkeypatch.setattr(cheques, "model_available", lambda path=cheques.MODEL_PATH: True)
    cheque_by_box = {tuple(b): CHEQUE_TEXT.get(k, "") for k, b in CHEQUE_BOXES.items()}
    monkeypatch.setattr(cheques, "ocr_box", lambda img, xyxy, **kw: cheque_by_box.get(tuple(int(v) for v in xyxy), ""))
    monkeypatch.setattr(receipts, "_model", FakeYolo())
    monkeypatch.setattr(receipts, "model_available", lambda path=receipts.MODEL_PATH: True)
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, **kw: OCR_TEXT[tuple(int(v) for v in xyxy)])
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": True, "cheque": True, "contract": True})
    monkeypatch.setattr(service.DocumentService, "invoice", lambda self: FakeInvoice())

    def words(img, max_side=1400):
        w = img.shape[1]
        return "contrat preteur emprunteur soussignes article 1 taeg mensualites" if w == 600 else "payez contre ce cheque a l'ordre de" if img.shape[0] / w < 0.65 else \
            "invoice bill to unit price" if img.shape[0] / w < 1.6 else "cashier merci visa"
    monkeypatch.setattr(router, "quick_text", words)
    monkeypatch.setattr(accounts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(accounts, "DB", tmp_path / "documind.db")
    monkeypatch.setattr(accounts, "SCANS", tmp_path / "scans")
    return TestClient(main.app)


INVOICE_PNG, RECEIPT_PNG, CHEQUE_PNG, CONTRACT_PNG = _png(800, 1130, 1), _png(400, 1000, 2), _png(800, 350, 3), _png(600, 848, 4)


def _post(client, name, data, **kw):
    return client.post("/extract", files={"file": (name, data, "image/png")}, **kw)


def test_contract_is_routed_read_and_checked(client):
    r = _post(client, "k.png", CONTRACT_PNG).json()
    assert r["doc_type"] == "contract" and r["doc_type_source"] == "auto" and r["columns"] == contracts.CONTRACT_COLUMNS
    row = r["row"]
    assert (row["AMOUNT"], row["CURRENCY"], row["RATE"], row["DURATION_MONTHS"], row["MONTHLY_PAYMENT"], row["CONTRACT_TYPE"]) == ("100000.00", "MAD", "6.00", "60", "1933.28", "personal")
    assert (row["AMOUNT_CHECK"], row["PAYMENT_CHECK"], row["SIGNED_LENDER"], row["SIGNED_BORROWER"], row["STAMP"], row["LENDER"]) == ("ok", "ok", "yes", "yes", "yes", "Banque Atlas")
    assert row["FIRST_PAYMENT_DATE"] == "2024-04-12" and r["warnings"] == [] and r["items"] == []
    labels = {f["label"]: f for f in r["fields"]}
    assert labels["AMOUNT"]["text"] == "100000.00" and len(labels["AMOUNT"]["bbox"]) == 4 and labels["BORROWER"]["text"] == "Khadija AMRANI"      # the wrapped amount is ONE field
    assert len(labels["AMOUNT"]["boxes"]) == 2 and "boxes" not in labels["BORROWER"]                                 # ... drawn as one box per printed line, not one big rectangle
    forced = _post(client, "k.png", CONTRACT_PNG, params={"doc_type": "invoice"}).json()
    assert forced["doc_type"] == "invoice" and _post(client, "k.png", CONTRACT_PNG, params={"doc_type": "passport"}).status_code == 422


def test_contract_csv_and_batch_never_merge_the_four_types(client):
    head = _post(client, "k.png", CONTRACT_PNG, params={"format": "csv"})
    assert head.content.decode("utf-8-sig").splitlines()[0] == ",".join(contracts.CONTRACT_COLUMNS) and "contracts.csv" in head.headers["content-disposition"]
    files = [("files", ("a.png", INVOICE_PNG, "image/png")), ("files", ("b.png", RECEIPT_PNG, "image/png")), ("files", ("c.png", CHEQUE_PNG, "image/png")),
             ("files", ("k.png", CONTRACT_PNG, "image/png")), ("files", ("l.png", CONTRACT_PNG, "image/png"))]
    z = zipfile.ZipFile(io.BytesIO(client.post("/extract/batch", files=files).content))
    assert sorted(z.namelist()) == ["cheques.csv", "contracts.csv", "invoices.csv", "receipts.csv"]
    rows = {n: z.read(n).decode("utf-8-sig").splitlines() for n in z.namelist()}
    assert len(rows["contracts.csv"]) == 3 and rows["contracts.csv"][0].startswith("file,CONTRACT_TYPE,TITLE") and rows["cheques.csv"][0].startswith("file,BANK")
    assert rows["receipts.csv"][0].startswith("file,STORE") and rows["invoices.csv"][0].startswith("file,SELLER_NAME")
    assert client.post("/extract/batch", files=files, params={"format": "json"}).json()["csv_files"] == {"invoices.csv": 1, "receipts.csv": 1, "cheques.csv": 1, "contracts.csv": 2}


def test_history_and_dataset_export_with_contracts(client):
    tok = client.post("/auth/register", json={"name": "Gogo", "email": "g@example.com", "password": "12345678"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    _post(client, "a.png", INVOICE_PNG, headers=h)
    _post(client, "k.png", CONTRACT_PNG, headers=h)
    rows = {r["doc_type"]: r for r in client.get("/history", headers=h).json()}
    assert rows["contract"]["total"] == "100000.00" and rows["invoice"]["total"] == "100.00"
    z = zipfile.ZipFile(io.BytesIO(client.get("/dataset/export", headers=h).content))
    assert {"dataset_invoices.csv", "dataset_contracts.csv"} <= set(z.namelist()) and "BORROWER" in z.read("dataset_contracts.csv").decode("utf-8-sig").splitlines()[0]


def test_missing_contract_model_gives_the_train_command(client, monkeypatch):
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": True, "cheque": True, "contract": False})
    assert _post(client, "k.png", CONTRACT_PNG).json()["doc_type"] != "contract"                    # not installed: the router never picks it
    r = _post(client, "k.png", CONTRACT_PNG, params={"doc_type": "contract"})
    assert r.status_code == 503 and "train_contracts" in r.json()["detail"] and client.get("/health").json()["models"]["contract"] is False
