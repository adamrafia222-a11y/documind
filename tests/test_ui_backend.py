"""The two backends of the front end (in this process / through the API) must behave the same: one scenario, run against both.
    pytest -q tests/test_ui_backend.py
"""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from documind.api import accounts, main
from documind.inference import service
from documind.inference.extractor import ModelNotFound
from documind.ui.backend import ApiBackend, BackendError, LocalBackend


def fake_result(name, doc_type="invoice"):
    return {"file": name, "doc_type": doc_type, "doc_type_source": "auto", "fields": [{"label": "TOTAL", "text": "10,00", "confidence": 0.9, "bbox": [0, 0, 5, 5], "page": 1}],
            "items": [], "warnings": [{"code": "w", "message": "check this"}], "row": {"file": name, "TOTAL": "10,00"}, "columns": ["file", "TOTAL"], "pages": [{"page": 1, "size": [10, 10]}]}


@pytest.fixture(params=["local", "api"])
def be(request, monkeypatch, tmp_path):
    monkeypatch.setattr(accounts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(accounts, "DB", tmp_path / "documind.db")
    monkeypatch.setattr(accounts, "SCANS", tmp_path / "scans")

    def extract(self, data, filename, doc_type="auto"):
        if filename == "nomodel.png":
            raise ModelNotFound("The cheque model is not installed. Train it first")
        if filename == "bad.png":
            raise ValueError("Unreadable image.")
        return fake_result(filename, "cheque" if doc_type == "cheque" else "invoice")
    monkeypatch.setattr(service.DocumentService, "extract", extract)
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": False, "cheque": True, "contract": False})
    return LocalBackend(service.DocumentService()) if request.param == "local" else ApiBackend("", session=TestClient(main.app))


def test_the_whole_account_life_is_the_same_on_both_backends(be):
    assert be.available()["cheque"] is True
    s = be.register("Gogo", "gogo@example.com", "secret-pass-1")
    tok = s["token"]
    assert s["user"]["name"] == "Gogo" and be.me(tok)["email"] == "gogo@example.com" and be.me(tok)["created"] > 0
    with pytest.raises(BackendError) as e:
        be.register("Gogo", "gogo@example.com", "secret-pass-1")
    assert e.value.status == 409 and "already exists" in e.value.detail
    with pytest.raises(BackendError) as e:
        be.login("gogo@example.com", "wrong-password")
    assert e.value.status == 401 and e.value.detail == "Wrong email or password"

    r = be.extract(tok, b"png-bytes", "a.png", "auto")
    be.extract(tok, b"png-bytes-2", "b.png", "cheque")
    assert r["doc_type"] == "invoice"
    hist = be.history(tok)
    assert [h["file"] for h in hist] == ["b.png", "a.png"] and hist[0]["doc_type"] == "cheque" and hist[0]["n_warnings"] == 1 and hist[0]["has_img"]
    item = be.history_item(tok, hist[1]["id"])
    assert item["result"]["fields"][0]["text"] == "10,00" and be.history_image(tok, hist[1]["id"]) == b"png-bytes"
    assert be.stats(tok)["documents"] == 2 and be.stats(tok)["by_type"] == {"invoice": 1, "cheque": 1}
    z = zipfile.ZipFile(io.BytesIO(be.dataset_zip(tok)))
    assert {"dataset_invoices.csv", "dataset_cheques.csv", "dataset.jsonl"} <= set(z.namelist())

    assert be.rename(tok, "Gogo Dev")["name"] == "Gogo Dev"
    with pytest.raises(BackendError) as e:
        be.change_password(tok, "wrong", "another-pass-2")
    assert e.value.status == 401
    be.change_password(tok, "secret-pass-1", "another-pass-2")
    assert be.login("gogo@example.com", "another-pass-2")["user"]["name"] == "Gogo Dev"

    be.delete_history(tok, hist[0]["id"])
    assert [h["file"] for h in be.history(tok)] == ["a.png"] and be.history_image(tok, hist[0]["id"]) is None
    be.logout(tok)
    with pytest.raises(BackendError) as e:
        be.history(tok)
    assert e.value.status == 401
    tok2 = be.login("gogo@example.com", "another-pass-2")["token"]
    with pytest.raises(BackendError) as e:
        be.delete_account(tok2, "wrong")
    assert e.value.status == 401
    be.delete_account(tok2, "another-pass-2")
    with pytest.raises(BackendError):
        be.login("gogo@example.com", "another-pass-2")


def test_guests_can_read_but_nothing_is_saved_and_errors_are_explained(be):
    assert be.extract(None, b"x", "g.png")["file"] == "g.png"
    tok = be.register("Gogo", "gogo@example.com", "secret-pass-1")["token"]
    assert be.history(tok) == []                                                                   # the guest reading was not stored for anybody
    with pytest.raises(BackendError) as e:
        be.extract(tok, b"x", "nomodel.png", "cheque")
    assert e.value.status == 503 and "not installed" in e.value.detail
    with pytest.raises(BackendError) as e:
        be.extract(None, b"x", "bad.png")
    assert e.value.status == 422 and "Unreadable" in e.value.detail
    with pytest.raises(BackendError) as e:
        be.stats("not-a-token")
    assert e.value.status == 401 and be.history(tok) == []                                         # nothing was saved for the failed readings either


def test_unreachable_api_gives_a_clear_message():
    class Down:
        def get(self, *a, **k):
            raise ConnectionError("refused")
    err = None
    try:
        ApiBackend("http://nowhere:1", session=Down()).available()
    except BackendError as exc:
        err = exc
    assert err and err.status == 0 and "Cannot reach the API at http://nowhere:1" in err.detail
