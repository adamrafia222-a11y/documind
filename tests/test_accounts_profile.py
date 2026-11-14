"""Accounts: the profile features added for the new front end (rename, password, delete account, statistics, history warnings) and who can see what.
    pytest -q tests/test_accounts_profile.py
"""
import time

import pytest
from fastapi.testclient import TestClient

from documind.api import accounts, main


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(accounts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(accounts, "DB", tmp_path / "documind.db")
    monkeypatch.setattr(accounts, "SCANS", tmp_path / "scans")
    return TestClient(main.app)


def signup(client, email="gogo@example.com", name="Gogo", pw="secret-pass-1"):
    r = client.post("/auth/register", json={"name": name, "email": email, "password": pw})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["token"]}


def fake_result(doc_type="invoice", warnings=0, conf=0.9):
    return {"file": "x.png", "doc_type": doc_type, "fields": [{"label": "TOTAL", "text": "10,00", "confidence": conf, "bbox": [0, 0, 1, 1]}],
            "items": [], "warnings": [{"code": "w", "message": "m"}] * warnings, "row": {"file": "x.png"}, "columns": ["file"]}


def test_me_says_when_the_account_was_created(client):
    h = signup(client)
    me = client.get("/auth/me", headers=h).json()
    assert me["email"] == "gogo@example.com" and abs(me["created"] - time.time()) < 60


def test_rename(client):
    h = signup(client)
    assert client.patch("/auth/me", json={"name": "  Gogo Dev "}, headers=h).json()["name"] == "Gogo Dev"
    assert client.get("/auth/me", headers=h).json()["name"] == "Gogo Dev"
    assert client.patch("/auth/me", json={"name": "G"}, headers=h).status_code == 422 and client.patch("/auth/me", json={"name": "Gogo"}).status_code == 401


def test_change_password_closes_the_other_sessions(client):
    h1 = signup(client)
    h2 = {"Authorization": "Bearer " + client.post("/auth/login", json={"email": "gogo@example.com", "password": "secret-pass-1"}).json()["token"]}
    assert client.post("/auth/password", json={"current": "wrong", "new": "another-pass-2"}, headers=h1).status_code == 401
    assert client.post("/auth/password", json={"current": "secret-pass-1", "new": "short"}, headers=h1).status_code == 422
    assert client.post("/auth/password", json={"current": "secret-pass-1", "new": "another-pass-2"}, headers=h1).status_code == 200
    assert client.get("/auth/me", headers=h1).status_code == 200 and client.get("/auth/me", headers=h2).status_code == 401      # the caller stays in, the other session is closed
    assert client.post("/auth/login", json={"email": "gogo@example.com", "password": "secret-pass-1"}).status_code == 401
    assert client.post("/auth/login", json={"email": "gogo@example.com", "password": "another-pass-2"}).status_code == 200


def test_statistics_and_history_warnings(client):
    h = signup(client)
    user = client.get("/auth/me", headers=h).json()
    assert client.get("/stats", headers=h).json()["documents"] == 0 and client.get("/stats").status_code == 401
    accounts.save_history(user, fake_result("invoice", 0, 0.9), 2000)
    accounts.save_history(user, fake_result("cheque", 2, 0.7), 1000)
    accounts.save_history(user, fake_result("cheque", 0, 0.8), 3000)
    s = client.get("/stats", headers=h).json()
    assert s["documents"] == 3 and s["by_type"] == {"invoice": 1, "cheque": 2} and s["with_warnings"] == 1
    assert s["avg_confidence"] == pytest.approx(0.8) and s["avg_seconds"] == 2.0 and sum(s["per_day"].values()) == 3 and s["first"] <= s["last"]
    hist = client.get("/history", headers=h).json()
    assert [x["n_warnings"] for x in hist] == [0, 2, 0] and "result" not in hist[0]                                          # newest first, the heavy result is not sent


def test_delete_account_removes_everything_of_the_account_only(client):
    h = signup(client)
    other = signup(client, "other@example.com", "Other")
    mine, theirs = client.get("/auth/me", headers=h).json(), client.get("/auth/me", headers=other).json()
    accounts.save_history(mine, fake_result(), 10, b"img-bytes")
    accounts.save_history(theirs, fake_result(), 10, b"other-bytes")
    hid = client.get("/history", headers=h).json()[0]["id"]
    assert client.get(f"/scans/{hid}/image", headers=h).content == b"img-bytes"
    assert client.get(f"/scans/{hid}/image", headers=other).status_code == 404                                                # nobody else can open my scans
    assert client.post("/auth/delete", json={"password": "wrong"}, headers=h).status_code == 401
    assert client.post("/auth/delete", json={"password": "secret-pass-1"}, headers=h).status_code == 200
    assert client.get("/auth/me", headers=h).status_code == 401 and client.post("/auth/login", json={"email": "gogo@example.com", "password": "secret-pass-1"}).status_code == 401
    assert len(client.get("/history", headers=other).json()) == 1 and not list((accounts.SCANS / str(mine["id"])).glob("*"))   # their data and files are untouched, mine are gone


def test_statistics_for_the_dashboard(client):
    h = signup(client)
    user = client.get("/auth/me", headers=h).json()
    empty = client.get("/stats", headers=h).json()
    assert empty["documents"] == 0 and empty["confidence_hist"] == [0] * 10 and len(empty["heatmap"]) == 7 and all(len(r) == 24 for r in empty["heatmap"]) and empty["recent"] == []
    def res(doc_type, conf, codes=()):
        r = fake_result(doc_type, 0, conf)
        r["warnings"] = [{"code": c, "message": c} for c in codes]
        r["fields"] += [{"label": "DATE", "text": "x", "confidence": conf / 2, "bbox": [0, 0, 1, 1]}]
        return r
    for dt, conf, codes, ms in (("invoice", 0.95, (), 1000), ("invoice", 0.55, ("amount_mismatch",), 3000), ("cheque", 0.85, ("amount_mismatch", "no_signature"), 2000), ("cheque", 0.8, (), 2000)):
        accounts.save_history(user, res(dt, conf, codes), ms)
    s = client.get("/stats", headers=h).json()
    assert s["documents"] == 4 and s["by_type"] == {"invoice": 2, "cheque": 2} and s["with_warnings"] == 2 and s["warnings_by_type"] == {"invoice": 1, "cheque": 1}
    assert s["confidence_hist"] == [0, 0, 0, 0, 1, 0, 2, 1, 0, 0]                  # a document's confidence is the mean of its fields: 0.41, 0.60, 0.64, 0.71
    assert s["seconds_by_type"] == {"invoice": 2.0, "cheque": 2.0} and dict(s["warning_codes"]) == {"amount_mismatch": 2, "no_signature": 1} and s["warning_codes"][0][0] == "amount_mismatch"
    day = next(iter(s["per_day_type"]))
    assert s["per_day_type"][day] == {"invoice": 2, "cheque": 2} and sum(s["per_day"].values()) == 4 and sum(sum(r) for r in s["heatmap"]) == 4
    weak = {w["label"]: w for w in s["weakest_fields"]}
    assert weak["DATE"]["n"] == 4 and weak["DATE"]["avg"] < weak["TOTAL"]["avg"] and s["weakest_fields"][0]["label"] == "DATE"                           # the field read with the lowest confidence comes first
    assert [r["file"] for r in s["recent"]] == ["x.png"] * 4 and s["recent"][0]["id"] > s["recent"][-1]["id"]                                           # newest first
