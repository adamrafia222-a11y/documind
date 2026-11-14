"""The Streamlit front end: its helper functions and the page itself (run headlessly with Streamlit's AppTest, no browser, no trained model).
    pytest -q tests/test_streamlit_ui.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("streamlit")

from documind.inference import cheques, contracts, receipts                                   # noqa: E402
from documind.ui import helpers as H                                                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cheques import CHEQUE_BOXES, CHEQUE_TEXT, FakeChequeYolo                            # noqa: E402
from test_contracts import CONTRACT_BOXES, CONTRACT_TEXT, FakeContractYolo                     # noqa: E402
from test_router_service import FakeInvoice, FakeYolo, OCR_TEXT, _png                          # noqa: E402


@pytest.fixture()
def entries(monkeypatch):
    """One finished reading of every document type, made with the fake detectors (the real extraction code runs)."""
    by_box = {tuple(b): {"lender_name": "Banque Atlas", "amount": "100 000,00" if b[0] == 300 else "MAD"}.get(n, CONTRACT_TEXT.get(n, "")) for n, b in CONTRACT_BOXES}
    monkeypatch.setattr(contracts, "ocr_box", lambda img, xyxy, **kw: by_box.get(tuple(int(v) for v in xyxy), ""))
    chq = {tuple(b): CHEQUE_TEXT.get(k, "") for k, b in CHEQUE_BOXES.items()}
    monkeypatch.setattr(cheques, "ocr_box", lambda img, xyxy, **kw: chq.get(tuple(int(v) for v in xyxy), ""))
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, **kw: OCR_TEXT[tuple(int(v) for v in xyxy)])
    png = {"a": _png(800, 1130, 1), "b": _png(400, 1000, 2), "c": _png(800, 350, 3), "k": _png(600, 848, 4)}
    inv = FakeInvoice().extract_bytes(png["a"], "a.png")
    inv.update(doc_type="invoice", columns=FakeInvoice.columns, doc_type_source="manual")
    res = [inv, receipts.extract_receipt_bytes(png["b"], "b.png", model=FakeYolo()), cheques.extract_cheque_bytes(png["c"], "c.png", model=FakeChequeYolo()),
           contracts.extract_contract_bytes(png["k"], "k.png", model=FakeContractYolo())]
    res[3]["warnings"].append({"code": "x", "message": "A test warning."})
    return [{"name": r["file"], "data": png[r["file"][0]], "result": r, "forced": "auto"} for r in res]


# ------------------------------------------------------------------------------------------------ helpers
def test_one_csv_per_type_each_with_its_own_columns(entries):
    results = [e["result"] for e in entries] + [entries[3]["result"]]                                    # two contracts
    csvs = H.csv_by_type(results)
    assert sorted(csvs) == ["cheques.csv", "contracts.csv", "invoices.csv", "receipts.csv"]
    lines = {n: c.decode("utf-8-sig").splitlines() for n, c in csvs.items()}
    assert lines["contracts.csv"][0] == ",".join(contracts.CONTRACT_COLUMNS) and len(lines["contracts.csv"]) == 3
    assert lines["receipts.csv"][0] == ",".join(receipts.RECEIPT_COLUMNS) and lines["cheques.csv"][0] == ",".join(cheques.CHEQUE_COLUMNS)
    assert lines["invoices.csv"][0] == ",".join(FakeInvoice.columns)
    z = zipfile.ZipFile(io.BytesIO(H.zip_files(csvs)))
    assert sorted(z.namelist()) == sorted(csvs) and z.read("cheques.csv") == csvs["cheques.csv"]


def test_boxes_are_drawn_on_the_picture(entries):
    e = entries[3]
    img = H.page_image(e["data"], e["name"])
    out = H.draw_boxes(img, e["result"]["fields"], e["result"]["items"], 1, max_width=300)
    assert out.size == (300, int(img.height * 300 / img.width)) and out.getpixel((1, 1)) is not None
    assert list(Image.open(io.BytesIO(e["data"])).convert("RGB").resize(out.size).tobytes()) != list(out.tobytes())        # something was drawn
    assert H.draw_boxes(img, [], [], 1, max_width=5000).size == img.size                                  # never enlarged
    wrapped = {"label": "AMOUNT", "text": "x", "confidence": .9, "bbox": [10, 10, 590, 60], "boxes": [[400, 10, 590, 30], [10, 40, 80, 60]], "page": 1}
    plain = Image.new("RGB", (600, 100), "white")
    two = H.draw_boxes(plain, [wrapped], [], 1, max_width=600)
    assert two.getpixel((200, 20)) == (255, 255, 255) and two.getpixel((500, 20)) != (255, 255, 255)      # the gap between the two lines is not covered
    assert H.label_color("AMOUNT") == H.label_color("AMOUNT") and H.label_color("AMOUNT") != H.label_color("PAYEE")


def test_tables(entries):
    rec = entries[1]["result"]
    f = H.fields_table(rec)
    assert {"field", "text", "confidence", "page"} == set(f[0]) and f[0]["confidence"].endswith("%")
    assert [r["field"] for r in f] == [x["label"] for x in sorted(rec["fields"], key=lambda x: (x["bbox"][1], x["bbox"][0]))]
    items = H.items_table(rec)
    assert items and set(items[0]) <= {"qty", "description", "unit_price", "amount"} and "unit_price" not in items[0]       # empty columns are left out
    rows = H.summary_rows(entries + [{"name": "bad.png", "result": None, "error": "The receipt model is not installed."}])
    assert [r["type"] for r in rows] == ["Invoice", "Receipt", "Cheque", "Contract", "-"] and rows[3]["status"] == "check" and rows[4]["status"].startswith("ERROR")


def test_read_api_explains_what_went_wrong(monkeypatch):
    import requests

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._b, self.text = code, body, json.dumps(body)

        def json(self):
            return self._b
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp(200, {"doc_type": "cheque", "fields": []}))
    assert H.read_api("http://x:1/", b"d", "c.png", "auto")["doc_type"] == "cheque"
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp(503, {"detail": "The cheque model is not installed. Train it first"}))
    with pytest.raises(RuntimeError, match="not installed"):
        H.read_api("http://x:1", b"d", "c.png", "cheque")

    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(RuntimeError, match="Cannot reach"):
        H.read_api("http://x:1", b"d", "c.png")
    monkeypatch.setattr(requests, "get", boom)
    assert H.api_health("http://x:1") is None


# ------------------------------------------------------------------------------------------------ history and profile helpers
def test_history_helpers():
    rows = [{"id": 3, "file": "Facture_A.png", "doc_type": "invoice", "created": 1_700_000_000, "n_fields": 9, "conf": 0.912, "n_warnings": 0, "total": "100,00", "ms": 2400},
            {"id": 2, "file": "cheque_b.png", "doc_type": "cheque", "created": 1_700_000_100, "n_fields": 10, "conf": 0.8, "n_warnings": 2, "total": "1200.00", "ms": 1000},
            {"id": 1, "file": "ticket.jpg", "doc_type": "receipt", "created": 1_700_000_200, "n_fields": 5, "conf": 0.7, "n_warnings": 0, "total": "", "ms": 500}]
    assert [r["id"] for r in H.filter_history(rows, ["cheque", "receipt"])] == [2, 1] and [r["id"] for r in H.filter_history(rows, None, "FACT")] == [3]
    assert [r["id"] for r in H.filter_history(rows, None, "", True)] == [2] and [r["id"] for r in H.filter_history(rows, ["invoice"], "cheque")] == [] and len(H.filter_history(rows)) == 3
    f = H.history_frame(rows)
    assert f[0]["Type"] == "Invoice" and f[0]["Confidence"] == 91 and f[0]["Seconds"] == 2.4 and f[2]["Total"] == "" and f[1]["Warnings"] == 2
    assert H.format_when(None) == "-" and len(H.format_when(1_700_000_000)) >= 16


def test_per_day_series_has_no_holes():
    import datetime
    s = H.per_day_series({"2026-11-12": 3, "2026-11-10": 1}, 4, today=datetime.date(2026, 11, 12))
    assert s == [("2026-11-09", 0), ("2026-11-10", 1), ("2026-11-11", 0), ("2026-11-12", 3)] and len(H.per_day_series({}, 30)) == 30


def test_average_confidence(entries):
    assert H.average_confidence({"fields": []}) == 0.0 and 0.5 < H.average_confidence(entries[3]["result"]) <= 1.0
