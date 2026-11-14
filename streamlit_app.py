"""DocuMind - the Streamlit front end: sign in / create account, dashboard, read documents, history, profile.

    pip install -r requirements-streamlit.txt
    streamlit run streamlit_app.py                    # opens http://localhost:8501

Drop invoices, receipts, cheques and loan contracts: the page finds the type (or you choose it), reads the fields, shows them on the picture, lists the
checks that failed and gives ONE CSV PER DOCUMENT TYPE (invoices.csv, receipts.csv, cheques.csv, contracts.csv: the types are never mixed).
Signed in, every reading is kept in your history; as a guest you can read but nothing is saved.

Where it runs (sign-in page, "Connection"): "This computer" loads the models and the accounts database inside this process (nothing else to start);
"API server" talks to the FastAPI server (uvicorn documind.api.main:app). Both use the same account database format, so the HTML page and this one share accounts.
The pages are in documind/ui/views.py, the look in documind/ui/theme.py, the two backends in documind/ui/backend.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                                  # `import documind` works wherever streamlit is started from

import streamlit as st                                         # noqa: E402

st.set_page_config(page_title="DocuMind", page_icon="🧾", layout="wide", initial_sidebar_state="expanded")

from documind.ui import views as V                             # noqa: E402


@st.cache_resource(show_spinner="Loading the models ...")
def get_service():
    from documind.inference.service import DocumentService
    return DocumentService()


V.inject_theme()
V.init_state()

if not (V.signed_in() or st.session_state["guest"]):
    V.auth_view(get_service)
else:
    backend = V.current_backend(get_service)
    V.sidebar(backend)
    page = st.session_state["page"] if V.signed_in() else "read"
    V.guard({"dashboard": V.dashboard_view, "read": V.read_view, "history": V.history_view, "profile": V.profile_view}[page], backend)
