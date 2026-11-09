"""Document-type router, receipt extraction through the service and the API, one CSV per type.
No Tesseract, no Ultralytics and no trained model needed: the detector, the OCR and the invoice extractor are replaced by small fakes.
    pytest -q tests/test_router_service.py
"""
import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from documind.api import accounts, main
from documind.inference import receipts, router, service


# ------------------------------------------------------------------------------------------------ router
def test_score_types_text_and_shape():
    a4 = router.score_types("TAX INVOICE  Bill To: ACME  Invoice No 123  Unit price  Qty  Due date", aspect=1.41)
    assert a4["invoice"] > a4["receipt"]
    strip = router.score_types("PANDA MARKET  Cashier: 4  Merci  Visa  Change 0.00", aspect=2.6)
    assert strip["receipt"] > strip["invoice"]
    assert router.score_types("", aspect=2.4)["receipt"] > router.score_types("", aspect=2.4)["invoice"]      # no text: the shape decides
    assert router.score_types("", aspect=1.41)["invoice"] > router.score_types("", aspect=1.41)["receipt"]


def _png(w, h, seed=0) -> bytes:
    arr = (np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_detect_type_uses_available_models(monkeypatch):
    monkeypatch.setattr(router, "quick_text", lambda img, max_side=1400: "cashier merci visa")
    assert router.detect_type(_png(300, 800), "r.png")["doc_type"] == "receipt"
    only_invoice = router.detect_type(_png(300, 800), "r.png", ["invoice"])                    # the receipt model is not installed
    assert only_invoice["doc_type"] == "invoice" and only_invoice["confidence"] == 1.0


# ------------------------------------------------------------------------------------------------ fakes
class _Box:
    def __init__(self, cls, xyxy, conf):
        self.cls, self.xyxy, self.conf = cls, np.array([xyxy], dtype=float), conf


class FakeYolo:
    names = {0: "store", 1: "date", 2: "total", 3: "receipt_no", 4: "item"}
    overrides = {"imgsz": 960}

    def predict(self, img, imgsz=960, conf=0.25, verbose=False):
        class R:
            boxes = [_Box(0, [10, 10, 200, 40], 0.9), _Box(1, [10, 60, 200, 90], 0.8), _Box(2, [10, 700, 200, 740], 0.95),
                     _Box(3, [10, 100, 200, 130], 0.7), _Box(4, [10, 300, 290, 330], 0.9), _Box(4, [10, 340, 290, 370], 0.85)]
        return [R()]


OCR_TEXT = {(10, 10, 200, 40): "PANDA MARKET", (10, 60, 200, 90): "Date 30/09/2022", (10, 700, 200, 740): "Total 548,98",
            (10, 100, 200, 130): "No 3347048", (10, 300, 290, 330): "Milk 2 x 5.00 10.00", (10, 340, 290, 370): "Bread 3.50"}


class FakeInvoice:
    columns = ["file", "SELLER_NAME", "TOTAL", "N_ITEMS", "ITEMS"]

    def extract_bytes(self, data, filename="document"):
        return {"file": filename, "warnings": [], "n_pages": 1, "pages": [{"page": 1, "size": [800, 1130]}],
                "fields": [{"label": "SELLER_NAME", "text": "ACME", "confidence": 0.9, "bbox": [1, 1, 50, 20], "page": 1},
                           {"label": "TOTAL", "text": "100.00", "confidence": 0.9, "bbox": [1, 900, 50, 920], "page": 1}],
                "items": [], "items_check": {}, "row": {"file": filename, "SELLER_NAME": "ACME", "TOTAL": "100.00", "N_ITEMS": "0", "ITEMS": ""}}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(receipts, "_model", FakeYolo())
    monkeypatch.setattr(receipts, "model_available", lambda path=receipts.MODEL_PATH: True)
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, **kw: OCR_TEXT[tuple(int(v) for v in xyxy)])
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": True})
    monkeypatch.setattr(service.DocumentService, "invoice", lambda self: FakeInvoice())
    monkeypatch.setattr(router, "quick_text", lambda img, max_side=1400: "invoice bill to unit price" if img.shape[0] / img.shape[1] < 1.6 else "cashier merci visa")
    monkeypatch.setattr(accounts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(accounts, "DB", tmp_path / "documind.db")
    monkeypatch.setattr(accounts, "SCANS", tmp_path / "scans")
    return TestClient(main.app)


INVOICE_PNG, RECEIPT_PNG = _png(800, 1130, 1), _png(400, 1000, 2)


def _post(client, name, data, **kw):
    return client.post("/extract", files={"file": (name, data, "image/png")}, **kw)


# ------------------------------------------------------------------------------------------------ tests
def test_extract_routes_each_picture_to_its_extractor(client):
    inv = _post(client, "a.png", INVOICE_PNG).json()
    rec = _post(client, "b.png", RECEIPT_PNG).json()
    assert inv["doc_type"] == "invoice" and inv["doc_type_source"] == "auto"
    assert rec["doc_type"] == "receipt" and rec["columns"] == receipts.RECEIPT_COLUMNS
    labels = {f["label"]: f["text"] for f in rec["fields"]}
    assert labels == {"STORE": "PANDA MARKET", "DATE": "30/09/2022", "TOTAL": "548,98", "RECEIPT_NO": "3347048"}
    assert [it["amount"] for it in rec["items"]] == ["10.00", "3.50"] and rec["row"]["N_ITEMS"] == "2"
    assert all(len(f["bbox"]) == 4 for f in rec["fields"])
    assert rec["pages"][0]["width"] == 400 and rec["pages"][0]["height"] == 1000


def test_manual_type_overrides_the_router(client):
    forced = _post(client, "b.png", RECEIPT_PNG, params={"doc_type": "invoice"}).json()
    assert forced["doc_type"] == "invoice" and forced["doc_type_source"] == "manual"
    assert _post(client, "a.png", INVOICE_PNG, params={"doc_type": "receipt"}).json()["doc_type"] == "receipt"
    assert _post(client, "a.png", INVOICE_PNG, params={"doc_type": "passport"}).status_code == 422


def test_single_csv_has_only_its_own_columns(client):
    r = _post(client, "b.png", RECEIPT_PNG, params={"format": "csv"})
    head = r.content.decode("utf-8-sig").splitlines()[0]
    assert head == ",".join(receipts.RECEIPT_COLUMNS) and "receipts.csv" in r.headers["content-disposition"]
    head = _post(client, "a.png", INVOICE_PNG, params={"format": "csv"}).content.decode("utf-8-sig").splitlines()[0]
    assert head == "file,SELLER_NAME,TOTAL,N_ITEMS,ITEMS"


def test_batch_never_merges_types(client):
    files = [("files", ("a.png", INVOICE_PNG, "image/png")), ("files", ("b.png", RECEIPT_PNG, "image/png")), ("files", ("c.png", RECEIPT_PNG, "image/png"))]
    mixed = client.post("/extract/batch", files=files)
    assert mixed.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(mixed.content))
    assert sorted(z.namelist()) == ["invoices.csv", "receipts.csv"]
    inv, rec = (z.read(n).decode("utf-8-sig").splitlines() for n in ("invoices.csv", "receipts.csv"))
    assert len(inv) == 2 and len(rec) == 3                                         # header + 1 invoice, header + 2 receipts
    assert inv[0].startswith("file,SELLER_NAME") and rec[0].startswith("file,STORE")
    same = client.post("/extract/batch", files=files[1:])                         # only receipts: a plain csv
    assert same.headers["content-type"].startswith("text/csv") and "receipts.csv" in same.headers["content-disposition"]
    js = client.post("/extract/batch", files=files, params={"format": "json"}).json()
    assert js["csv_files"] == {"invoices.csv": 1, "receipts.csv": 2}


def test_history_and_dataset_export_keep_types_apart(client):
    tok = client.post("/auth/register", json={"name": "Gogo", "email": "g@example.com", "password": "12345678"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    _post(client, "a.png", INVOICE_PNG, headers=h)
    _post(client, "b.png", RECEIPT_PNG, headers=h)
    rows = client.get("/history", headers=h).json()
    assert sorted(r["doc_type"] for r in rows) == ["invoice", "receipt"]
    assert {r["total"] for r in rows} == {"100.00", "548,98"}
    z = zipfile.ZipFile(io.BytesIO(client.get("/dataset/export", headers=h).content))
    assert {"dataset.jsonl", "dataset_invoices.csv", "dataset_receipts.csv"} <= set(z.namelist()) and "dataset.csv" not in z.namelist()
    assert "STORE" in z.read("dataset_receipts.csv").decode("utf-8-sig").splitlines()[0]
    assert "STORE" not in z.read("dataset_invoices.csv").decode("utf-8-sig").splitlines()[0]


def test_health_and_missing_receipt_model(client, monkeypatch):
    assert client.get("/health").json()["models"] == {"invoice": True, "receipt": True}
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": False})
    assert _post(client, "b.png", RECEIPT_PNG).json()["doc_type"] == "invoice"      # only the invoice model exists: everything goes there
    r = _post(client, "b.png", RECEIPT_PNG, params={"doc_type": "receipt"})
    assert r.status_code == 503 and "train_receipts" in r.json()["detail"]
