"""Cheques: reading and checking the fields, the cheque dataset preprocessing, the third document type in the router / API (one CSV per type).
No Tesseract, Ultralytics or trained model needed: the detector and the OCR are replaced by small fakes.
    pytest -q tests/test_cheques.py
"""
import importlib.util
import io
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from documind.api import accounts, main
from documind.data import prepare_cheques
from documind.data.prepare_receipts import run
from documind.inference import cheques, receipts, router, service
from documind.training import train_cheques, train_receipts

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "generate_cheques.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_router_service import FakeInvoice, FakeYolo, OCR_TEXT, _png                      # noqa: E402  (fakes of the receipt / invoice tests)


# ------------------------------------------------------------------------------------------------ amount in words -> number
def test_words_to_number_examples():
    w = cheques.words_to_number
    assert w("Mille deux cent trente dirhams et cinquante centimes") == (1230, 50, "MAD")
    assert w("quatre-vingt-dix-neuf euros") == (99, 0, "EUR")
    assert w("quatre-vingts dirhams") == (80, 0, "MAD") and w("deux cents euros") == (200, 0, "EUR")
    assert w("soixante et onze dirhams") == (71, 0, "MAD") and w("cent mille dirhams") == (100000, 0, "MAD")
    assert w("cinquante centimes") == (0, 50, "")
    assert w("Mille deux cent qvarante dirhams") == (1240, 0, "MAD")                       # one letter wrong (OCR): the closest word is used
    assert w("") is None and w("n'importe quoi du tout") is None


@pytest.mark.skipif(not GENERATOR.exists(), reason="scripts/generate_cheques.py not present")
def test_words_to_number_round_trips_the_generators_own_words():
    spec = importlib.util.spec_from_file_location("generate_cheques", GENERATOR)
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    rng = random.Random(0)
    for n in list(range(0, 1200)) + [rng.randint(0, 999_999) for _ in range(1500)] + [80_000, 280, 380_000, 999_999, 71_000, 91]:
        for cents, cur in ((0, "MAD"), (50, "EUR"), (1, "MAD")):
            assert cheques.words_to_number(gc.amount_to_words(n, cents, cur)) == (n, cents, cur), gc.amount_to_words(n, cents, cur)


def test_parse_helpers():
    assert cheques.parse_amount("12 345,50") == 12345.5 and cheques.parse_amount("450") == 450.0 and cheques.parse_amount("1200.5 DH") == 1200.5 and cheques.parse_amount("abc") is None
    assert cheques.parse_date("13/10/2025") == ("13/10/2025", True) and cheques.parse_date("2025-10-13")[1] and cheques.parse_date("31/02/2025")[1] is False
    assert cheques.parse_date("not a date") == ("not a date", False)
    assert cheques.micr_cheque_number("<1234567<  4560123:  123456789012<45") == "1234567" and cheques.micr_cheque_number("garbage") == ""


# ------------------------------------------------------------------------------------------------ the checks
def _got(**raw) -> dict:
    base = {"bank_name": "Banque Atlas", "cheque_number": "1234567", "amount_words": "mille deux cents dirhams", "payee": "Omar Tazi", "amount_digits": "1 200,00",
            "place": "Rabat", "date": "13/10/2025", "signature": "", "account_number": "1234 5678 9012 3456 7890 1234", "micr_line": "<1234567<  45601234:  123456789012<45"}
    base.update(raw)
    return {k: {"value": v, "raw_text": v, "box_confidence": 0.9, "bbox": [0, 0, 1, 1]} for k, v in base.items()}


def test_consistent_cheque_has_no_warning():
    r = cheques.interpret(_got())
    assert r["warnings"] == [] and r["checks"]["amount_match"] is True and r["values"]["amount"] == "1200.00" and r["values"]["currency"] == "MAD"
    assert r["checks"]["micr_number_match"] is True and r["values"]["amount_check"] == "ok" and r["values"]["cheque_no"] == "1234567"


def test_amount_mismatch_is_reported_and_never_corrected():
    r = cheques.interpret(_got(amount_digits="1 500,00"))
    assert r["checks"]["amount_match"] is False and r["values"]["amount_check"] == "mismatch" and r["values"]["amount"] == "1500.00"
    assert [w["code"] for w in r["warnings"]] == ["amount_mismatch"]


def test_unreadable_words_are_not_compared_and_other_problems_are_reported():
    r = cheques.interpret(_got(amount_words="xx yy zz"))
    assert r["checks"]["amount_match"] is None and r["values"]["amount_check"] == "not checked" and "amount_not_checked" in {w["code"] for w in r["warnings"]}
    got = _got(micr_line="<7654321<  1:", date="31/02/2025", account_number="123")
    del got["signature"]
    codes = {w["code"] for w in cheques.interpret(got)["warnings"]}
    assert {"micr_mismatch", "date_invalid", "account_length", "no_signature"} <= codes
    got = _got()
    del got["payee"], got["amount_words"]
    assert "missing_field" in {w["code"] for w in cheques.interpret(got)["warnings"]}


def test_amount_words_get_a_second_reading_when_the_first_one_disagrees(monkeypatch):
    texts = iter(["garbled xx", "mille deux cents dirhams"])                          # first pass is garbage, the second pass is right
    monkeypatch.setattr(cheques, "ocr_box", lambda img, xyxy, **kw: next(texts))
    monkeypatch.setattr(cheques, "read_text", lambda img, box, field, lang=None: "1 200,00" if field == "amount_digits" else "garbled xx")
    dets = [("amount_digits", [0, 0, 10, 10], 0.9), ("amount_words", [0, 20, 10, 30], 0.9)]
    got = cheques.fields_from_boxes(Image.new("RGB", (50, 50)), dets)
    assert got["amount_words"]["raw_text"] == "mille deux cents dirhams" and cheques.interpret(got)["checks"]["amount_match"] is True


# ------------------------------------------------------------------------------------------------ dataset preparation and training profile
@pytest.mark.skipif(not GENERATOR.exists(), reason="scripts/generate_cheques.py not present")
def test_prepare_cheques_reads_the_generators_dataset(tmp_path):
    raw = tmp_path / "raw"
    subprocess.run([sys.executable, str(GENERATOR), "--n", "10", "--out", str(raw), "--workers", "1"], check=True, capture_output=True)
    stats = run(raw, tmp_path / "out", workers=1, reports_dir=tmp_path / "rep", tag="cheques", default_names=prepare_cheques.CHEQUE_NAMES, max_side=1600)
    assert sum(v["images"] for v in stats["splits"].values()) == 10 and list(stats["names"].values()) == prepare_cheques.CHEQUE_NAMES
    assert (tmp_path / "out" / "data.yaml").exists() and (tmp_path / "rep" / "cheques_prepare_report.txt").exists() and (tmp_path / "rep" / "cheques_stats.json").exists()
    assert (tmp_path / "rep" / "cheques_preview").is_dir() and not (tmp_path / "rep" / "receipts_stats.json").exists()      # cheque reports never overwrite receipt reports


def test_find_raw_cheques_falls_back_to_the_generators_folder(tmp_path):
    (tmp_path / "cheque_dataset" / "images" / "train").mkdir(parents=True)
    assert prepare_cheques.find_raw_cheques(tmp_path) == tmp_path / "cheque_dataset"
    assert prepare_cheques.find_raw_cheques(tmp_path / "nothing") == tmp_path / "nothing" / "cheques"


def test_cheque_training_profile_is_separate_from_the_receipt_one():
    c, r = train_cheques.profile(), train_receipts.Profile()
    assert c.kind == "cheques" and c.model_out.name == "cheques_yolo.pt" and r.model_out.name == "receipts_yolo.pt"
    assert c.imgsz == 1280 and c.epochs == 60 and r.imgsz == 960 and c.data_yaml.parent.name == "cheques_yolo" and c.report_path.name == "cheques_report.json"
    assert c.aug["fliplr"] == 0.0 and c.aug["flipud"] == 0.0 and "train_cheques" not in r.prepare_cmd and "prepare_cheques" in c.prepare_cmd
    rows = [{"split": "val", "class": "ALL", "mAP50": 0.95}, {"split": "test", "class": "ALL", "mAP50": 0.93}]
    assert train_receipts.check_target(rows, 0.90)["reached"] and not train_receipts.check_target(rows, 0.94)["reached"]


# ------------------------------------------------------------------------------------------------ router and API: three types, three CSV files
def test_router_scores_the_cheque_strip():
    s = router.score_types("Banque Atlas  Payez contre ce cheque non endossable  a l'ordre de  Signature RIB", aspect=0.44)
    assert s["cheque"] > s["invoice"] and s["cheque"] > s["receipt"]
    assert router.score_types("", aspect=0.44)["cheque"] > router.score_types("", aspect=0.44)["invoice"]
    a4 = router.score_types("TAX INVOICE Bill To Invoice No Unit price Qty Due date Banque", aspect=1.41)
    assert a4["invoice"] > a4["cheque"]
    assert router.score_types("cashier merci visa", aspect=2.4)["receipt"] > router.score_types("cashier merci visa", aspect=2.4)["cheque"]


CHEQUE_BOXES = {"bank_name": [10, 10, 200, 40], "cheque_number": [300, 10, 390, 40], "amount_words": [10, 80, 300, 110], "payee": [10, 130, 300, 160], "amount_digits": [320, 130, 390, 160],
                "place": [10, 180, 100, 210], "date": [120, 180, 200, 210], "signature": [220, 220, 380, 330], "account_number": [10, 250, 200, 280], "micr_line": [10, 300, 200, 330]}
CHEQUE_TEXT = {"bank_name": "Banque Atlas", "cheque_number": "7654321", "amount_words": "mille deux cents dirhams", "payee": "Omar Tazi", "amount_digits": "1 200,00", "place": "Rabat",
               "date": "13/10/2025", "account_number": "1234 5678 9012 3456 7890 1234", "micr_line": "<7654321<  45601234:  123456789012<45"}


class FakeChequeYolo:
    names = dict(enumerate(CHEQUE_BOXES))
    overrides = {"imgsz": 1280}

    def predict(self, img, imgsz=1280, conf=0.25, verbose=False):
        class B:
            def __init__(s, cls, xyxy):
                s.cls, s.xyxy, s.conf = cls, np.array([xyxy], dtype=float), 0.9

        class R:
            boxes = [B(i, b) for i, b in enumerate(CHEQUE_BOXES.values())]
        return [R()]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    by_box = {tuple(b): CHEQUE_TEXT.get(k, "") for k, b in CHEQUE_BOXES.items()}
    monkeypatch.setattr(cheques, "_model", FakeChequeYolo())
    monkeypatch.setattr(cheques, "model_available", lambda path=cheques.MODEL_PATH: True)
    monkeypatch.setattr(cheques, "ocr_box", lambda img, xyxy, **kw: by_box.get(tuple(int(v) for v in xyxy), ""))
    monkeypatch.setattr(receipts, "_model", FakeYolo())
    monkeypatch.setattr(receipts, "model_available", lambda path=receipts.MODEL_PATH: True)
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, **kw: OCR_TEXT[tuple(int(v) for v in xyxy)])
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": True, "cheque": True})
    monkeypatch.setattr(service.DocumentService, "invoice", lambda self: FakeInvoice())

    def words(img, max_side=1400):
        ratio = img.shape[0] / img.shape[1]
        return "payez contre ce cheque a l'ordre de signature" if ratio < 0.65 else "invoice bill to unit price" if ratio < 1.6 else "cashier merci visa"
    monkeypatch.setattr(router, "quick_text", words)
    monkeypatch.setattr(accounts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(accounts, "DB", tmp_path / "documind.db")
    monkeypatch.setattr(accounts, "SCANS", tmp_path / "scans")
    return TestClient(main.app)


INVOICE_PNG, RECEIPT_PNG, CHEQUE_PNG = _png(800, 1130, 1), _png(400, 1000, 2), _png(800, 350, 3)


def _post(client, name, data, **kw):
    return client.post("/extract", files={"file": (name, data, "image/png")}, **kw)


def test_cheque_is_routed_read_and_checked(client):
    r = _post(client, "c.png", CHEQUE_PNG).json()
    assert r["doc_type"] == "cheque" and r["doc_type_source"] == "auto" and r["columns"] == cheques.CHEQUE_COLUMNS
    labels = {f["label"]: f["text"] for f in r["fields"]}
    assert labels["AMOUNT"] == "1200.00" and labels["PAYEE"] == "Omar Tazi" and labels["CHEQUE_NO"] == "7654321" and labels["SIGNATURE"] == "signed"
    assert r["row"]["AMOUNT_CHECK"] == "ok" and r["row"]["CURRENCY"] == "MAD" and r["row"]["SIGNATURE"] == "yes" and r["warnings"] == []
    assert all(len(f["bbox"]) == 4 for f in r["fields"]) and r["items"] == []
    forced = _post(client, "c.png", CHEQUE_PNG, params={"doc_type": "invoice"}).json()
    assert forced["doc_type"] == "invoice" and _post(client, "c.png", CHEQUE_PNG, params={"doc_type": "passport"}).status_code == 422


def test_cheque_csv_and_batch_never_merge_the_three_types(client):
    head = _post(client, "c.png", CHEQUE_PNG, params={"format": "csv"})
    assert head.content.decode("utf-8-sig").splitlines()[0] == ",".join(cheques.CHEQUE_COLUMNS) and "cheques.csv" in head.headers["content-disposition"]
    files = [("files", ("a.png", INVOICE_PNG, "image/png")), ("files", ("b.png", RECEIPT_PNG, "image/png")), ("files", ("c.png", CHEQUE_PNG, "image/png")),
             ("files", ("d.png", CHEQUE_PNG, "image/png"))]
    z = zipfile.ZipFile(io.BytesIO(client.post("/extract/batch", files=files).content))
    assert sorted(z.namelist()) == ["cheques.csv", "invoices.csv", "receipts.csv"]
    heads = {n: z.read(n).decode("utf-8-sig").splitlines() for n in z.namelist()}
    assert len(heads["cheques.csv"]) == 3 and heads["cheques.csv"][0].startswith("file,BANK,CHEQUE_NO") and heads["receipts.csv"][0].startswith("file,STORE")
    assert heads["invoices.csv"][0].startswith("file,SELLER_NAME")
    assert client.post("/extract/batch", files=files, params={"format": "json"}).json()["csv_files"] == {"invoices.csv": 1, "receipts.csv": 1, "cheques.csv": 2}


def test_history_shows_the_cheque_amount_and_the_dataset_keeps_a_csv_per_type(client):
    tok = client.post("/auth/register", json={"name": "Gogo", "email": "g@example.com", "password": "12345678"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    for name, png in (("a.png", INVOICE_PNG), ("c.png", CHEQUE_PNG)):
        _post(client, name, png, headers=h)
    rows = {r["doc_type"]: r for r in client.get("/history", headers=h).json()}
    assert rows["cheque"]["total"] == "1200.00" and rows["invoice"]["total"] == "100.00"
    z = zipfile.ZipFile(io.BytesIO(client.get("/dataset/export", headers=h).content))
    assert {"dataset_invoices.csv", "dataset_cheques.csv"} <= set(z.namelist())
    assert "PAYEE" in z.read("dataset_cheques.csv").decode("utf-8-sig").splitlines()[0]


def test_missing_cheque_model_gives_the_train_command(client, monkeypatch):
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": True, "cheque": False})
    assert _post(client, "c.png", CHEQUE_PNG).json()["doc_type"] != "cheque"                    # not installed: the router never picks it
    r = _post(client, "c.png", CHEQUE_PNG, params={"doc_type": "cheque"})
    assert r.status_code == 503 and "train_cheques" in r.json()["detail"]
    assert client.get("/health").json()["models"]["cheque"] is False
