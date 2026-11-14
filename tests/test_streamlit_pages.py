"""The pages of the Streamlit front end, driven headlessly with Streamlit's AppTest: sign in / sign up / guest, reading, history, profile.
The models are replaced by a fake service; the account database is a temporary one (the real LocalBackend and accounts code run).
    pytest -q tests/test_streamlit_pages.py
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import streamlit as st                                                                       # noqa: E402
from streamlit.testing.v1 import AppTest                                                      # noqa: E402

from documind.api import accounts                                                              # noqa: E402
from documind.inference import service                                                         # noqa: E402
from documind.inference.extractor import ModelNotFound                                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_router_service import _png                                                           # noqa: E402


def result(name, doc_type="invoice", warnings=0, items=False):
    r = {"file": name, "doc_type": doc_type, "doc_type_source": "auto", "routing": {"confidence": 0.8}, "n_pages": 1, "pages": [{"page": 1, "width": 200, "height": 300, "size": [200, 300]}],
         "fields": [{"label": "TOTAL", "text": "10,00", "confidence": 0.9, "bbox": [10, 10, 80, 30], "page": 1}, {"label": "DATE", "text": "01/01/2025", "confidence": 0.8, "bbox": [10, 50, 80, 70], "page": 1}],
         "items": [{"description": "Lait", "qty": "2", "amount": "10,00", "bbox": [10, 90, 80, 110], "page": 1}] if items else [], "items_check": {}, "checks": {"amount_match": True},
         "warnings": [{"code": "amount_mismatch" if i == 0 else "no_signature", "message": f"Warning {i + 1}"} for i in range(warnings)], "row": {"file": name, "TOTAL": "10,00"}, "columns": ["file", "TOTAL"]}
    return r


class Holder:
    files: list = []


class Upload:
    def __init__(self, name, data):
        self.name, self._d = name, data

    def getvalue(self):
        return self._d


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(accounts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(accounts, "DB", tmp_path / "documind.db")
    monkeypatch.setattr(accounts, "SCANS", tmp_path / "scans")

    def extract(self, data, filename, doc_type="auto"):
        if filename == "nomodel.png":
            raise ModelNotFound("The cheque model is not installed. Train it first")
        t = "cheque" if "chq" in filename else doc_type if doc_type != "auto" else "invoice"
        return result(filename, t, warnings=2 if "warn" in filename else 0, items="items" in filename)
    monkeypatch.setattr(service.DocumentService, "extract", extract)
    monkeypatch.setattr(service.DocumentService, "available", lambda self: {"invoice": True, "receipt": False, "cheque": True, "contract": False})
    Holder.files = []
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: Holder.files)
    return tmp_path


def app(**state) -> AppTest:
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def submit(at, label):
    return [b for b in at.button if b.label == label][0]


def sign_up(at, name="Gogo", email="gogo@example.com", pw="secret-pass-1", pw2=None):
    at.text_input(key="su_name").input(name)
    at.text_input(key="su_email").input(email)
    at.text_input(key="su_pw1").input(pw)
    at.text_input(key="su_pw2").input(pw if pw2 is None else pw2)
    return submit(at, "Create account").click().run()


def signed_in_app():
    at = app().run()
    at = sign_up(at)
    assert not at.exception and at.session_state["token"]
    return at


def body(at) -> str:
    """The HTML of the page itself, without the style sheet (which names every class and would make a "the class is there" check meaningless)."""
    return " ".join(m.value for m in at.markdown if not m.value.lstrip().startswith("<style") and "dm-orb" not in m.value)


def read(at, names):
    Holder.files = [Upload(n, _png(200, 300, i)) for i, n in enumerate(names)]
    if at.session_state["page"] != "read":
        at = at.sidebar.button(key="nav_read").click().run()
    at = at.run()
    at.button(key="read_go").click().run()
    return at


# ------------------------------------------------------------------------------------------------ sign in / sign up / guest
def test_sign_in_page(env):
    at = app().run()
    assert not at.exception and [t.label for t in at.tabs] == ["Sign in", "Create account"]
    assert {b.label for b in at.button} >= {"Sign in", "Create account", "Continue as a guest  →"}
    assert any("Read any" in m.value and "rot" in m.value for m in at.markdown) and any("dm-doc" in m.value for m in at.markdown)         # the animated document is there
    assert len(at.sidebar.button) == 0                                                                                                 # no menu before signing in


def test_wrong_password_and_empty_form(env):
    at = app().run()
    submit(at, "Sign in").click().run()
    assert any("Enter your email" in e.value for e in at.error)
    at = sign_up(at)
    at.session_state["token"] = None
    at = app().run()
    at.text_input(key="si_email").input("gogo@example.com")
    at.text_input(key="si_pw").input("not-the-password")
    at = submit(at, "Sign in").click().run()
    assert any(e.value == "Wrong email or password" for e in at.error) and not at.session_state["token"] if "token" in at.session_state else True


def test_sign_up_checks_the_password_and_the_email_and_then_signs_in(env):
    at = app().run()
    at = sign_up(at, pw="secret-pass-1", pw2="different")
    assert any("not the same" in e.value for e in at.error) and "token" not in at.session_state
    at = sign_up(at, pw="short", pw2="short")
    assert any("8" in e.value for e in at.error)
    at = sign_up(at)
    assert not at.exception and at.session_state["user"]["name"] == "Gogo" and any("Gogo" in m.value and "gogo@example.com" in m.value for m in at.sidebar.markdown)
    assert {b.key for b in at.sidebar.button} >= {"nav_dashboard", "nav_read", "nav_history", "nav_profile", "signout"} and at.session_state["page"] == "dashboard"
    again = sign_up(app().run(), name="Other")
    assert any("already exists" in e.value for e in again.error)                                                                       # the email is taken


def test_sign_out_and_sign_in_again(env):
    at = signed_in_app()
    at = at.sidebar.button(key="signout").click().run()
    assert "token" not in at.session_state and [t.label for t in at.tabs] == ["Sign in", "Create account"]
    at.text_input(key="si_email").input("gogo@example.com")
    at.text_input(key="si_pw").input("secret-pass-1")
    at = submit(at, "Sign in").click().run()
    assert at.session_state["user"]["name"] == "Gogo" and at.session_state["page"] == "dashboard"


def test_guest_can_read_but_nothing_is_saved_and_there_is_no_history(env):
    at = app().run()
    at = at.button(key="guest_btn").click().run()
    assert not at.exception and {b.key for b in at.sidebar.button} == {"nav_read", "to_auth"} and any("Guest mode" in i.value for i in at.info)
    at = read(at, ["a.png"])
    assert not at.exception and at.session_state["results"][0]["result"]["doc_type"] == "invoice"
    assert accounts.q("SELECT COUNT(*) c FROM history")[0][0]["c"] == 0
    assert not any("saved in your history" in m.value for m in at.markdown)
    at = at.sidebar.button(key="to_auth").click().run()
    assert [t.label for t in at.tabs] == ["Sign in", "Create account"]


def test_the_connection_can_point_to_an_unreachable_api(env):
    at = app(conn="API server", api_url="http://127.0.0.1:9").run()
    at.text_input(key="si_email").input("gogo@example.com")
    at.text_input(key="si_pw").input("secret-pass-1")
    at = submit(at, "Sign in").click().run()
    assert not at.exception and any("Cannot reach the API at http://127.0.0.1:9" in e.value for e in at.error)


# ------------------------------------------------------------------------------------------------ read
def test_reading_shows_results_downloads_and_saves_to_the_history(env):
    at = signed_in_app()
    at = read(at, ["a.png", "chq1.png", "nomodel.png"])
    assert not at.exception
    assert any("Read 2 of 3 documents" in m.value and "saved in your history" in m.value for m in at.markdown)
    rows = at.dataframe[0].value
    assert list(rows["file"]) == ["a.png", "chq1.png", "nomodel.png"] and rows["type"].tolist()[:2] == ["Invoice", "Cheque"] and rows["status"].tolist()[2].startswith("ERROR")
    labels = [b.label for b in at.get("download_button")]
    assert any("invoices.csv" in x for x in labels) and any("cheques.csv" in x for x in labels) and any("zip" in x for x in labels)        # a CSV per type, never merged
    assert [t.label for t in at.tabs] == ["Picture", "Fields", "Checks", "CSV row", "JSON"]
    at = [s for s in at.selectbox if s.label == "Document"][0].select_index(2).run()
    assert any("not installed" in e.value for e in at.error)                                                                              # the failed one explains itself
    assert accounts.q("SELECT COUNT(*) c FROM history")[0][0]["c"] == 2                                                                  # only the two readings that worked were saved


def test_read_as_another_type_reads_again(env):
    at = read(signed_in_app(), ["a.png"])
    assert {b.label for b in at.button} >= {"Read as receipt", "Read as cheque", "Read as contract"} and "Read as invoice" not in {b.label for b in at.button}
    at = at.button(key="r0_retype_contract").click().run()
    assert at.session_state["results"][0]["result"]["doc_type"] == "contract" and not at.exception


def test_items_tab_only_when_the_document_has_items(env):
    at = read(signed_in_app(), ["items1.png"])
    assert "Items" in [t.label for t in at.tabs]


# ------------------------------------------------------------------------------------------------ history
def test_history_lists_filters_opens_and_deletes(env):
    at = read(signed_in_app(), ["a.png", "chq_warn.png", "chq2.png"])
    at = at.sidebar.button(key="nav_history").click().run()
    assert not at.exception and at.session_state["page"] == "history"
    df = at.dataframe[0].value
    assert df["File"].tolist() == ["chq2.png", "chq_warn.png", "a.png"] and df["Type"].tolist() == ["Cheque", "Cheque", "Invoice"] and df["Warnings"].tolist() == [0, 2, 0]      # newest first
    assert any("3" in m.value and "Documents" in m.value for m in at.markdown)
    at = at.multiselect(key="hist_types").select("cheque").run()
    assert at.dataframe[0].value["File"].tolist() == ["chq2.png", "chq_warn.png"] and any("2 of 3 documents" in c.value for c in at.caption)
    at = at.text_input(key="hist_q").input("warn").run()
    assert at.dataframe[0].value["File"].tolist() == ["chq_warn.png"]
    assert any(w.value == "Warning 1" for w in at.warning) and any("2 warnings" in m.value for m in at.markdown)                           # the opened document shows its warnings
    hid = at.selectbox(key="hist_open_sel").value
    at = at.checkbox(key=f"hist_sure_{hid}").check().run()
    at = at.button(key=f"hist_del_{hid}").click().run()
    assert not at.exception and accounts.q("SELECT COUNT(*) c FROM history")[0][0]["c"] == 2
    at.text_input(key="hist_q").input("")
    at = at.multiselect(key="hist_types").unselect("cheque").run()
    assert at.dataframe[0].value["File"].tolist() == ["chq2.png", "a.png"]


def test_history_csv_export_has_one_file_per_type(env):
    at = read(signed_in_app(), ["a.png", "chq1.png"])
    at = at.sidebar.button(key="nav_history").click().run()
    at = at.button(key="hist_prep").click().run()
    labels = [b.label for b in at.get("download_button")]
    assert any("invoices.csv" in x for x in labels) and any("cheques.csv" in x for x in labels) and any("all (zip)" in x for x in labels) and not any(x.endswith("extractions.csv") for x in labels)


def test_empty_history(env):
    at = signed_in_app().sidebar.button(key="nav_history").click().run()
    assert not at.exception and any("No history yet" in m.value for m in at.markdown)


# ------------------------------------------------------------------------------------------------ profile
def test_profile_shows_the_numbers_and_edits_the_name(env):
    at = read(signed_in_app(), ["a.png", "chq1.png", "chq_warn.png"])
    at = at.sidebar.button(key="nav_profile").click().run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Documents read" in body and "Member since" in body and "Most read" in body and "gogo@example.com" in body
    assert "--to:3" in body and "--to:1" in body                                                                                              # 3 documents, 1 with warnings (animated counters)
    at.text_input(key="pf_name").input("Gogo Dev")
    at = submit(at, "Save").click().run()
    assert at.session_state["user"]["name"] == "Gogo Dev" and any("Gogo Dev" in m.value for m in at.sidebar.markdown)


def test_password_change_and_then_sign_in_with_the_new_one(env):
    at = signed_in_app().sidebar.button(key="nav_profile").click().run()
    at.text_input(key="pw_cur").input("wrong")
    at.text_input(key="pw_new").input("another-pass-2")
    at.text_input(key="pw_rep").input("another-pass-2")
    at = submit(at, "Change password").click().run()
    assert any(e.value == "Wrong password" for e in at.error)
    at.text_input(key="pw_cur").input("secret-pass-1")
    at.text_input(key="pw_rep").input("different")
    at = submit(at, "Change password").click().run()
    assert any("not the same" in e.value for e in at.error)
    at.text_input(key="pw_rep").input("another-pass-2")
    at = submit(at, "Change password").click().run()
    assert any("Password changed" in s.value for s in at.success)
    at = at.sidebar.button(key="signout").click().run()
    at.text_input(key="si_email").input("gogo@example.com")
    at.text_input(key="si_pw").input("another-pass-2")
    assert at.session_state["page"] == "read" and submit(at, "Sign in").click().run().session_state["user"]["name"] == "Gogo"


def test_data_export_and_account_deletion(env):
    at = read(signed_in_app(), ["a.png"])
    at = at.sidebar.button(key="nav_profile").click().run()
    at = at.button(key="export_prep").click().run()
    assert any("documind_dataset.zip" in b.label for b in at.get("download_button"))
    at.text_input(key="del_pw").input("secret-pass-1")
    at = at.checkbox(key="del_sure").check().run()
    at = at.button(key="del_go").click().run()
    assert not at.exception and "token" not in at.session_state and any("account was deleted" in i.value for i in at.info)
    assert accounts.q("SELECT COUNT(*) c FROM users")[0][0]["c"] == 0 and accounts.q("SELECT COUNT(*) c FROM history")[0][0]["c"] == 0


def test_a_closed_session_goes_back_to_the_sign_in_page(env):
    at = signed_in_app()
    accounts.q("DELETE FROM sessions")                                                                                                       # e.g. the password was changed on another device
    at = at.sidebar.button(key="nav_history").click().run()
    assert not at.exception and "token" not in at.session_state and any("session ended" in i.value for i in at.info)


# ------------------------------------------------------------------------------------------------ dashboard
def test_the_dashboard_is_the_landing_page_and_guests_do_not_have_one(env):
    at = signed_in_app()
    assert at.session_state["page"] == "dashboard" and not at.exception
    guest = app().run().button(key="guest_btn").click().run()
    assert "nav_dashboard" not in {b.key for b in guest.sidebar.button}


def test_empty_dashboard_invites_to_read_a_first_document(env):
    at = signed_in_app()
    assert "waiting for its first document" in body(at) and 'class="dm-card dm-kpi' not in body(at)
    at = at.button(key="dash_to_read").click().run()
    assert at.session_state["page"] == "read"


def test_the_dashboard_draws_every_graph(env):
    at = read(signed_in_app(), ["a.png", "chq1.png", "chq_warn.png", "items.png"])
    at = at.sidebar.button(key="nav_dashboard").click().run()
    assert not at.exception
    html = body(at)
    for part in ('class="dm-card dm-kpi', '<svg class="dm-spark"', 'class="dm-donut"', 'class="dm-hist"', 'class="dm-hbars"', 'class="dm-heat"', 'class="dm-feed"'):
        assert part in html, part
    assert "Documents, last 30 days" in html and "--to:4" in html and "4 in total" in html                                         # the counters know the numbers
    assert "Checks that fail most" in html and "Amount in figures ≠ amount in words" in html and "No signature found" in html          # friendly names, not codes
    assert "Fields read with the least confidence" in html and "DATE" in html and "Warning rate by type" in html and "Average time by type" in html
    assert html.count('class="seg"') == 2                                                            # two types were read: two donut segments
    assert [e.type for e in at.main if type(e).__name__ == "UnknownElement"] == ["vega_lite_chart"] and not at.get("arrow_bar_chart")     # the stacked time series is the Altair chart, not the fallback
    assert "chq" in html and "chq_warn.png" in html                                                                                  # the latest documents feed


def test_the_period_changes_the_graphs(env):
    at = read(signed_in_app(), ["a.png", "chq1.png"])
    at = at.sidebar.button(key="nav_dashboard").click().run()
    at = at.radio(key="dash_days").set_value(7).run()
    assert not at.exception and "Documents, last 7 days" in body(at)
    at = at.radio(key="dash_days").set_value(90).run()
    html = body(at)
    assert not at.exception and "Documents, last 90 days" in html and "vs the previous" not in html                               # nothing to compare a 90-day period with


def test_open_a_latest_document_in_the_history(env):
    at = read(signed_in_app(), ["a.png", "chq1.png"])
    at = at.sidebar.button(key="nav_dashboard").click().run()
    pick = at.selectbox(key="dash_open").value
    at = at.button(key="dash_open_go").click().run()
    assert not at.exception and at.session_state["page"] == "history" and at.selectbox(key="hist_open_sel").value == pick
