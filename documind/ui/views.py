"""The pages of the Streamlit front end: sign in / create account, read documents, history, profile.

Each page is a function; they talk to a backend (documind.ui.backend: in this process or through the API) and never to the database or the models directly.
State kept in st.session_state: token + user (signed in) or guest, page, results (the last reading), conn / api_url (how to reach the backend).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from . import charts as C
from . import helpers as H
from . import theme as T
from .backend import ApiBackend, BackendError, LocalBackend

LOCAL, API = "This computer", "API server"
EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"]
PAGES = [("dashboard", "📊  Dashboard"), ("read", "📄  Read documents"), ("history", "🕘  History"), ("profile", "👤  Profile")]
SESSION_KEYS = ("token", "user", "guest", "page", "results", "hist_csv", "export_zip", "dash_days")


# ------------------------------------------------------------------------------------------------ state
def inject_theme() -> None:
    st.markdown(T.CSS + T.ORBS_HTML, unsafe_allow_html=True)


def init_state() -> None:
    for k, v in (("conn", LOCAL), ("api_url", "http://localhost:8000"), ("page", "read"), ("results", []), ("guest", False)):
        st.session_state.setdefault(k, v)


def signed_in() -> bool:
    return bool(st.session_state.get("token"))


def start_session(session: dict) -> None:
    for k in SESSION_KEYS:
        st.session_state.pop(k, None)
    st.session_state.update(token=session["token"], user=session["user"], guest=False, page="dashboard", results=[])


def end_session() -> None:
    for k in SESSION_KEYS:
        st.session_state.pop(k, None)
    st.session_state.update(guest=False, page="read", results=[])


def current_backend(get_service):
    """The backend chosen on the sign-in page: this computer (models + accounts database in this process) or an API server."""
    if st.session_state.get("conn") == API:
        return ApiBackend(st.session_state.get("api_url", "http://localhost:8000"))
    return LocalBackend(get_service())


def guard(fn, *args, **kwargs):
    """Runs a page. A closed session (the password was changed elsewhere, the server restarted) sends the person back to the sign-in page instead of showing an error."""
    try:
        return fn(*args, **kwargs)
    except BackendError as exc:
        if exc.status == 401 and signed_in():
            end_session()
            st.session_state["flash"] = "Your session ended. Please sign in again."
            st.rerun()
        st.error(exc.detail)


# ------------------------------------------------------------------------------------------------ sign in / create account
def auth_view(get_service) -> None:
    st.markdown(T.AUTH_CSS, unsafe_allow_html=True)                          # no sidebar before signing in
    left, right = st.columns([1.12, 1], gap="large")
    with left:
        st.markdown(T.hero_html(), unsafe_allow_html=True)
    with right:
        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        if st.session_state.get("flash"):
            st.info(st.session_state.pop("flash"))
        be = current_backend(get_service)
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("signin"):
                st.markdown("#### Welcome back 👋")
                st.caption("Sign in to find your history and your profile.")
                email = st.text_input("Email", key="si_email", placeholder="you@example.com")
                pw = st.text_input("Password", type="password", key="si_pw")
                go = st.form_submit_button("Sign in", type="primary")
            if go:
                if not email.strip() or not pw:
                    st.error("Enter your email and your password.")
                else:
                    try:
                        start_session(be.login(email.strip(), pw))
                        st.rerun()
                    except BackendError as exc:
                        st.error(exc.detail)
        with tab_up:
            with st.form("signup"):
                st.markdown("#### Create your account ✨")
                st.caption("Your readings are saved in your history, only you can see them.")
                name = st.text_input("Your name", key="su_name", placeholder="Gogo")
                email2 = st.text_input("Email", key="su_email", placeholder="you@example.com")
                pw1 = st.text_input("Password (8 characters or more)", type="password", key="su_pw1")
                pw2 = st.text_input("Repeat the password", type="password", key="su_pw2")
                go2 = st.form_submit_button("Create account", type="primary")
            if go2:
                if pw1 != pw2:
                    st.error("The two passwords are not the same.")
                else:
                    try:
                        start_session(be.register(name.strip(), email2.strip(), pw1))
                        st.rerun()
                    except BackendError as exc:
                        st.error(exc.detail)
        st.markdown('<div class="dm-muted" style="text-align:center;margin:.6rem 0 .2rem">or</div>', unsafe_allow_html=True)
        if st.button("Continue as a guest  →", key="guest_btn"):
            st.session_state["guest"] = True
            st.rerun()
        with st.expander("Connection"):
            st.radio("Where do the models and the accounts live?", [LOCAL, API], key="conn", help="This computer: nothing else to start. API server: the FastAPI server (uvicorn documind.api.main:app).")
            st.text_input("API address", key="api_url", disabled=st.session_state["conn"] == LOCAL)


# ------------------------------------------------------------------------------------------------ sidebar
def sidebar(be) -> None:
    with st.sidebar:
        st.markdown(T.brand_html(), unsafe_allow_html=True)
        user = st.session_state.get("user")
        if user:
            st.markdown(T.user_chip_html(user["name"], user["email"]), unsafe_allow_html=True)
        else:
            st.markdown(T.user_chip_html("Guest", "results are not saved"), unsafe_allow_html=True)
        for key, label in PAGES:
            if not user and key != "read":
                continue
            if st.button(label, key=f"nav_{key}", type="primary" if st.session_state["page"] == key else "secondary"):
                st.session_state["page"] = key
                st.rerun()
        st.divider()
        try:
            av = be.available()
            st.markdown(T.live_html("models") + "<br>" + "".join(T.chip_html(("● " if av.get(t) else "○ ") + T.TYPE_LABEL[t], "ok" if av.get(t) else "") + " " for t in H.TYPES), unsafe_allow_html=True)
            st.caption("● model installed  ○ not trained yet")
        except BackendError as exc:
            st.error(exc.detail)
        if user:
            if st.button("↩  Sign out", key="signout"):
                try:
                    be.logout(st.session_state["token"])
                except BackendError:
                    pass
                end_session()
                st.rerun()
        elif st.button("🔑  Sign in / create account", key="to_auth"):
            st.session_state["guest"] = False
            st.rerun()
        st.caption("Connected: " + ("this computer" if st.session_state["conn"] == LOCAL else st.session_state["api_url"]))


# ------------------------------------------------------------------------------------------------ one result
def result_view(r: dict, data: bytes | None, name: str, key: str, allow_retype: bool = False) -> str | None:
    """Everything about one reading: header with the confidence ring, warnings, picture with boxes, fields, items, checks, CSV row, JSON.
    Returns the type chosen with a "Read as ..." button (only when allow_retype), else None."""
    t = H.type_of(r)
    conf, warns = H.average_confidence(r), len(r.get("warnings", []))
    how = (f"auto-detected{' ' + str(round(r['routing']['confidence'] * 100)) + ' %' if r.get('routing') else ''}" if r.get("doc_type_source") == "auto" else "type chosen by you")
    chips = T.chip_html(f"{len(r.get('fields', []))} fields") + T.chip_html(how) + (T.chip_html(f"{warns} warning{'s' if warns > 1 else ''}", "warn") if warns else T.chip_html("all checks passed", "ok"))
    st.markdown(f'<div class="dm-card dm-up" style="display:flex;gap:1.2rem;align-items:center;margin:.4rem 0 1rem">{T.ring_html(conf * 100)}'
                f'<div><div>{T.type_badge_html(t)}</div><div style="font-size:1.15rem;font-weight:700;margin:.35rem 0">{T.escape(name)}</div><div class="dm-row">{chips}</div></div></div>', unsafe_allow_html=True)
    chosen = None
    if allow_retype:
        others = [x for x in H.TYPES if x != t]
        for col, o in zip(st.columns(len(others) + 2)[:len(others)], others):
            if col.button(f"Read as {H.TYPE_LABEL[o].lower()}", key=f"{key}_retype_{o}"):
                chosen = o
    for w in r.get("warnings", []):
        st.warning(w["message"])
    names = ["Picture", "Fields"] + (["Items"] if r.get("items") else []) + ["Checks", "CSV row", "JSON"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Picture"]:
        if data is None:
            st.caption("The original picture is not stored for this reading.")
        else:
            pages = max(1, int(r.get("n_pages", 1)))
            page = st.slider("Page", 1, pages, 1, key=f"{key}_page") if pages > 1 else 1
            st.image(H.draw_boxes(_cached_page(data, name, page), r.get("fields", []), r.get("items", []), page), caption=f"{name} - the detected fields")
    with tabs["Fields"]:
        st.dataframe(pd.DataFrame(H.fields_table(r)), hide_index=True)
    if "Items" in tabs:
        with tabs["Items"]:
            st.dataframe(pd.DataFrame(H.items_table(r)), hide_index=True)
            if r.get("items_check"):
                st.json(r["items_check"], expanded=False)
    with tabs["Checks"]:
        if r.get("checks"):
            st.dataframe(pd.DataFrame([{"check": k, "result": {True: "✅ passed", False: "❌ failed", None: "not checked"}.get(v, str(v))} for k, v in r["checks"].items()]), hide_index=True)
        else:
            st.caption("This document type has no cross-checks; see the warnings above.")
    with tabs["CSV row"]:
        st.dataframe(pd.DataFrame([r["row"]], columns=r.get("columns") or list(r["row"])), hide_index=True)
    with tabs["JSON"]:
        st.download_button("⬇ this result (JSON)", json.dumps(r, ensure_ascii=False, indent=2), file_name=f"{Path(name).stem}.json", mime="application/json", key=f"{key}_json")
        st.json(r, expanded=False)
    return chosen


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_page(data: bytes, name: str, page: int):
    return H.page_image(data, name, page)


# ------------------------------------------------------------------------------------------------ read documents
def read_entry(be, token: str | None, name: str, data: bytes, forced: str) -> dict:
    try:
        return {"name": name, "data": data, "result": be.extract(token, data, name, forced), "forced": forced}
    except BackendError as exc:                                  # a missing model or an unreadable file is shown in the table, the other files go on
        return {"name": name, "data": data, "result": None, "error": exc.detail, "forced": forced}


def read_view(be) -> None:
    token = st.session_state.get("token")
    st.markdown(T.page_header_html("Read documents", "Drop invoices, receipts, cheques or loan contracts: DocuMind finds the type, reads the fields and checks the numbers.", "📄"), unsafe_allow_html=True)
    if not token:
        st.info("Guest mode: your readings are not saved. Create an account to keep a history.")
    c1, c2 = st.columns([3, 1])
    with c1:
        files = st.file_uploader("Drop documents here", type=EXTENSIONS, accept_multiple_files=True, label_visibility="collapsed")
    with c2:
        forced = st.selectbox("Document type", ["auto", *H.TYPES], format_func=lambda t: "Auto-detect" if t == "auto" else T.TYPE_LABEL[t],
                              help="Auto-detect looks at the shape of the page and its words. If it guesses wrong, choose the type here or use the buttons under a result.")
    if st.button("Read documents", type="primary", disabled=not files, key="read_go"):
        slot, entries = st.empty(), []
        for i, f in enumerate(files, 1):
            slot.markdown(T.reading_html(f.name, len(files), i), unsafe_allow_html=True)
            entries.append(read_entry(be, token, f.name, f.getvalue(), forced))
        ok = sum(1 for e in entries if e["result"])
        slot.markdown(T.done_html(f"Read {ok} of {len(entries)} document{'s' if len(entries) > 1 else ''}" + (" - saved in your history" if token and ok else "")), unsafe_allow_html=True)
        st.session_state["results"] = entries
    entries = st.session_state["results"]
    if not entries:
        st.markdown(T.empty_html("📥", "Nothing read yet", "Drop one or more documents above, then press Read documents."), unsafe_allow_html=True)
        return
    st.markdown("### Results")
    st.dataframe(pd.DataFrame(H.summary_rows(entries)), hide_index=True)
    ok = [e["result"] for e in entries if e.get("result")]
    if ok:
        csvs = H.csv_by_type(ok)
        st.markdown("**Download** · one file per document type, never merged")
        cols = st.columns(len(csvs) + (1 if len(csvs) > 1 else 0))
        for col, (fname, content) in zip(cols, csvs.items()):
            n = sum(1 for r in ok if H.CSV_NAMES.get(H.type_of(r)) == fname)
            col.download_button(f"⬇ {fname} ({n})", content, file_name=fname, mime="text/csv", key=f"dl_{fname}")
        if len(csvs) > 1:
            cols[-1].download_button("⬇ all CSV files (zip)", H.zip_files(csvs), file_name="extractions.zip", mime="application/zip", key="dl_zip")
    st.divider()
    idx = st.selectbox("Document", range(len(entries)), format_func=lambda i: f"{i + 1}. {entries[i]['name']}", key="read_sel")
    e = entries[idx]
    if e.get("result") is None:
        st.error(e.get("error", "This document could not be read."))
        return
    again = result_view(e["result"], e["data"], e["name"], key=f"r{idx}", allow_retype=True)
    if again:
        entries[idx] = read_entry(be, st.session_state.get("token"), e["name"], e["data"], again)
        st.rerun()


# ------------------------------------------------------------------------------------------------ dashboard
def _show_altair(chart) -> None:
    import inspect
    st.altair_chart(chart, **({"width": "stretch"} if "width" in inspect.signature(st.altair_chart).parameters else {"use_container_width": True}))


def _timeseries(rows: list[dict]) -> None:
    """Documents per day, stacked by type, with tooltips (Altair: interactive); a plain bar chart if Altair is not usable."""
    if not rows:
        st.markdown(T.empty_html("📭", "No document in this period", "Choose a longer period or read a document."), unsafe_allow_html=True)
        return
    df = pd.DataFrame(rows)
    try:
        import altair as alt
        chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)), y=alt.Y("documents:Q", title=None, stack="zero"),
            color=alt.Color("type:N", scale=alt.Scale(domain=[T.TYPE_LABEL[t] for t in H.TYPES], range=[T.TYPE_COLOR[t] for t in H.TYPES]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("type:N"), alt.Tooltip("documents:Q")]).properties(height=270)
        _show_altair(chart)
    except Exception:
        st.bar_chart(df.pivot_table(index="date", columns="type", values="documents", aggfunc="sum").fillna(0))


def open_in_history(hid: int) -> None:
    for k in ("hist_types", "hist_q", "hist_warn"):
        st.session_state.pop(k, None)
    st.session_state.update(page="history", hist_open_sel=hid)
    st.rerun()


def dashboard_view(be) -> None:
    token = st.session_state["token"]
    stats = be.stats(token)
    st.markdown(T.page_header_html("Dashboard", "How your documents are going: volume, quality, and what needs a look.", "📊"), unsafe_allow_html=True)
    if not stats["documents"]:
        st.markdown(T.empty_html("🚀", "Your dashboard is waiting for its first document", "Read a document and the graphs fill in."), unsafe_allow_html=True)
        if st.button("Read a document", type="primary", key="dash_to_read"):
            st.session_state["page"] = "read"
            st.rerun()
        return
    days = st.radio("Period", [7, 30, 90], index=1, horizontal=True, format_func=lambda d: f"Last {d} days", key="dash_days", label_visibility="collapsed")
    rs = H.range_summary(stats, days)
    n = stats["documents"]
    warn_rate = stats["with_warnings"] / n * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(C.kpi_card_html("📚", f"Documents, last {days} days", rs["docs"], f"{n} in total", rs["spark"], H.delta_chip(rs["docs"], rs["prev"], days), "#8b5cf6", 1), unsafe_allow_html=True)
    k2.markdown(C.kpi_card_html("🎯", "Average confidence", f"{stats['avg_confidence'] * 100:.0f} %", "all time", None, None, "#34d399", 2), unsafe_allow_html=True)
    k3.markdown(C.kpi_card_html("⏱️", "Average time", f"{stats['avg_seconds']:.1f} s", "per document", None, None, "#22d3ee", 3), unsafe_allow_html=True)
    k4.markdown(C.kpi_card_html("⚠️", "Need a look", f"{warn_rate:.0f} %", f"{stats['with_warnings']} of {n} have a warning", None, ("healthy" if warn_rate < 20 else "check the warnings", "up" if warn_rate < 20 else "down"), "#fbbf24", 4), unsafe_allow_html=True)
    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown("##### Documents per day")
        _timeseries(rs["rows"])
    with right:
        parts = [(T.TYPE_LABEL[t], rs["by_type"][t], T.TYPE_COLOR[t]) for t in H.TYPES if rs["by_type"].get(t)]
        st.markdown(C.card_html("By document type", C.donut_html(parts, str(rs["docs"]), "documents") if parts else '<p class="dm-muted">Nothing in this period.</p>', f"last {days} days"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    labels = [f"{i * 10}-{i * 10 + 10}" for i in range(10)]
    c1.markdown(C.card_html("How sure was the reading?", C.histogram_html(stats["confidence_hist"], labels), "documents by average confidence (%), all time", 1), unsafe_allow_html=True)
    weak = [(w["label"], w["avg"] * 100, "var(--bad)" if w["avg"] < 0.6 else "var(--warn)" if w["avg"] < 0.85 else "var(--ok)") for w in stats["weakest_fields"]]
    c2.markdown(C.card_html("Fields read with the least confidence", C.hbars_html(weak, 100, "{:.0f} %", "Read a few more documents to see which fields are hardest."), "average confidence, all time", 2), unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    rate = [(T.TYPE_LABEL[t], stats["warnings_by_type"].get(t, 0) / stats["by_type"][t] * 100, T.TYPE_COLOR[t]) for t in H.TYPES if stats["by_type"].get(t)]
    d1.markdown(C.card_html("Warning rate by type", C.hbars_html(rate, 100, "{:.0f} %"), "documents with at least one warning"), unsafe_allow_html=True)
    codes = [(H.warning_label(c), k, "var(--warn)") for c, k in stats["warning_codes"]]
    d2.markdown(C.card_html("Checks that fail most", C.hbars_html(codes, None, "{:.0f}", "No warning so far: every check passed."), "times a check raised a warning", 1), unsafe_allow_html=True)
    secs = [(T.TYPE_LABEL[t], stats["seconds_by_type"][t], T.TYPE_COLOR[t]) for t in H.TYPES if t in stats["seconds_by_type"]]
    d3.markdown(C.card_html("Average time by type", C.hbars_html(secs, None, "{:.1f} s"), "seconds per document", 2), unsafe_allow_html=True)

    st.markdown(C.card_html("When do you read documents?", C.heatmap_html(stats["heatmap"]), "weekday × hour of the day, all time, your local time", 1), unsafe_allow_html=True)

    recent = [{**r, "when": H.format_when(r["created"])} for r in stats["recent"]]
    st.markdown(C.card_html("Latest documents", C.activity_html(recent), "newest first"), unsafe_allow_html=True)
    pick = st.selectbox("Open one in the history", [r["id"] for r in stats["recent"]], format_func=lambda i: next(f"{r['file']} · {H.format_when(r['created'])}" for r in stats["recent"] if r["id"] == i), key="dash_open")
    if st.button("Open in History", key="dash_open_go"):
        open_in_history(pick)


# ------------------------------------------------------------------------------------------------ history
def history_view(be) -> None:
    token = st.session_state["token"]
    st.markdown(T.page_header_html("History", "Everything you have read, newest first. Open a document to see it again with its fields drawn on it.", "🕘"), unsafe_allow_html=True)
    rows = be.history(token)
    if not rows:
        st.markdown(T.empty_html("🗂️", "No history yet", "Read a document and it will appear here."), unsafe_allow_html=True)
        if st.button("Read a document", type="primary", key="hist_to_read"):
            st.session_state["page"] = "read"
            st.rerun()
        return
    warned = sum(1 for r in rows if r.get("n_warnings"))
    avg = sum(r.get("conf") or 0 for r in rows) / len(rows)
    st.markdown(T.stat_grid_html([("📚", "Documents", len(rows), ""), ("⚠️", "With warnings", warned, "need a look"), ("🎯", "Average confidence", f"{avg * 100:.0f} %", "")]), unsafe_allow_html=True)
    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([2, 2, 1])
    present = [t for t in H.TYPES if any(r.get("doc_type") == t for r in rows)]
    types = f1.multiselect("Type", present, format_func=lambda t: T.TYPE_LABEL[t], key="hist_types")
    query = f2.text_input("Search a file name", key="hist_q", placeholder="invoice, cheque ...")
    only = f3.toggle("Only with warnings", key="hist_warn") if hasattr(st, "toggle") else f3.checkbox("Only with warnings", key="hist_warn")
    shown = H.filter_history(rows, types, query, only)
    st.caption(f"{len(shown)} of {len(rows)} documents")
    if not shown:
        st.markdown(T.empty_html("🔍", "Nothing matches", "Change or clear the filters."), unsafe_allow_html=True)
        return
    cfg = {"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%")} if hasattr(st.column_config, "ProgressColumn") else {}
    st.dataframe(pd.DataFrame(H.history_frame(shown)), hide_index=True, column_config=cfg)

    with st.expander("Export these documents as CSV (one file per type)"):
        if st.button("Prepare the CSV files", key="hist_prep"):
            bar, results = st.progress(0.0, text="Collecting ..."), []
            for i, r in enumerate(shown):
                results.append(be.history_item(token, r["id"])["result"])
                bar.progress((i + 1) / len(shown), text=f"{i + 1} / {len(shown)}")
            bar.empty()
            st.session_state["hist_csv"] = H.csv_by_type(results)
        csvs = st.session_state.get("hist_csv")
        if csvs:
            cols = st.columns(len(csvs) + (1 if len(csvs) > 1 else 0))
            for col, (fname, content) in zip(cols, csvs.items()):
                col.download_button(f"⬇ {fname}", content, file_name=fname, mime="text/csv", key=f"hist_dl_{fname}")
            if len(csvs) > 1:
                cols[-1].download_button("⬇ all (zip)", H.zip_files(csvs), file_name="history.zip", mime="application/zip", key="hist_dl_zip")

    st.divider()
    ids = [r["id"] for r in shown]
    by_id = {r["id"]: r for r in shown}
    hid = st.selectbox("Open a document", ids, format_func=lambda i: f"#{i} · {by_id[i]['file']} · {T.TYPE_LABEL.get(by_id[i]['doc_type'], by_id[i]['doc_type'])} · {H.format_when(by_id[i].get('created'))}", key="hist_open_sel")
    item = be.history_item(token, hid)
    img = be.history_image(token, hid) if by_id[hid].get("has_img") else None
    result_view(item["result"], img, item["file"], key=f"h{hid}")
    with st.expander("Delete this document"):
        sure = st.checkbox("Yes, delete it from my history (the stored picture too)", key=f"hist_sure_{hid}")
        if st.button("Delete", key=f"hist_del_{hid}", disabled=not sure):
            be.delete_history(token, hid)
            st.session_state.pop("hist_csv", None)
            st.toast("Deleted")
            st.rerun()


# ------------------------------------------------------------------------------------------------ profile
def profile_view(be) -> None:
    token = st.session_state["token"]
    me, stats = be.me(token), be.stats(token)
    st.session_state["user"] = {**st.session_state.get("user", {}), "name": me["name"], "email": me["email"]}
    st.markdown(T.page_header_html("Profile", "Your account and what you have done with DocuMind.", "👤"), unsafe_allow_html=True)
    st.markdown(f'<div class="dm-card dm-up" style="display:flex;gap:1.3rem;align-items:center">{T.avatar_html(me["name"], 84)}<div>'
                f'<div style="font-size:1.6rem;font-weight:800">{T.escape(me["name"])}</div><div class="dm-muted">{T.escape(me["email"])}</div>'
                f'<div style="margin-top:.4rem">{T.chip_html("Member since " + H.format_when(me.get("created")))}</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
    by_type = stats.get("by_type", {})
    favourite = max(by_type, key=by_type.get) if by_type else None
    st.markdown(T.stat_grid_html([("📚", "Documents read", stats["documents"], ""), ("🎯", "Average confidence", f"{stats['avg_confidence'] * 100:.0f} %", ""),
                                  ("⏱️", "Average time", f"{stats['avg_seconds']:.1f} s", "per document"), ("⚠️", "With warnings", stats["with_warnings"], ""),
                                  ("⭐", "Most read", T.TYPE_LABEL.get(favourite, "-") if favourite else "-", "")]), unsafe_allow_html=True)
    if stats["documents"]:
        c1, c2 = st.columns([2, 1])
        series = H.per_day_series(stats.get("per_day", {}))
        c1.markdown("**Activity, last 30 days**")
        c1.bar_chart(pd.DataFrame({"documents": [n for _, n in series]}, index=[d for d, _ in series]))
        c2.markdown("**By document type**")
        c2.bar_chart(pd.DataFrame({"documents": [by_type[t] for t in by_type]}, index=[T.TYPE_LABEL.get(t, t) for t in by_type]))

    tab_edit, tab_sec, tab_data, tab_danger = st.tabs(["Edit profile", "Security", "My data", "Danger zone"])
    with tab_edit:
        with st.form("rename"):
            name = st.text_input("Your name", value=me["name"], key="pf_name")
            if st.form_submit_button("Save", type="primary"):
                try:
                    st.session_state["user"]["name"] = be.rename(token, name)["name"]
                    st.toast("Profile saved")
                    st.rerun()
                except BackendError as exc:
                    st.error(exc.detail)
    with tab_sec:
        with st.form("password"):
            cur = st.text_input("Current password", type="password", key="pw_cur")
            new = st.text_input("New password (8 characters or more)", type="password", key="pw_new")
            rep = st.text_input("Repeat the new password", type="password", key="pw_rep")
            if st.form_submit_button("Change password", type="primary"):
                if new != rep:
                    st.error("The two new passwords are not the same.")
                else:
                    try:
                        be.change_password(token, cur, new)
                        st.success("Password changed. Your other sessions were signed out.")
                    except BackendError as exc:
                        st.error(exc.detail)
    with tab_data:
        st.caption("Everything you read, as a zip: your pictures, `dataset.jsonl` and one CSV per document type. Only your own documents are in it.")
        if st.button("Prepare my data", key="export_prep"):
            st.session_state["export_zip"] = be.dataset_zip(token)
        if st.session_state.get("export_zip"):
            st.download_button("⬇ documind_dataset.zip", st.session_state["export_zip"], file_name="documind_dataset.zip", mime="application/zip", key="export_dl")
    with tab_danger:
        st.warning("Deleting your account removes your history and your stored pictures for good.")
        pw = st.text_input("Your password", type="password", key="del_pw")
        sure = st.checkbox("I understand, delete my account", key="del_sure")
        if st.button("Delete my account", key="del_go", disabled=not sure or not pw):
            try:
                be.delete_account(token, pw)
                end_session()
                st.session_state["flash"] = "Your account was deleted."
                st.rerun()
            except BackendError as exc:
                st.error(exc.detail)
