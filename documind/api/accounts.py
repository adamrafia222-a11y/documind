"""Accounts, sessions, extraction history and dataset report (SQLite, standard library only)."""
from __future__ import annotations

import csv, hashlib, hmac, io, json, mimetypes, re, secrets, sqlite3, time, zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from ..config import DATA_DIR, REPORTS_DIR

DB = DATA_DIR / "documind.db"
SCANS = DATA_DIR / "scans"          # one folder per account: the original images/PDFs of every scan
router = APIRouter()
_SCHEMA = """CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, salt TEXT, pw TEXT, created REAL);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, user_id INTEGER, file TEXT, created REAL, n_fields INT, conf REAL,
  n_items INT, total TEXT, ms INT, result TEXT);"""


def q(sql: str, args: tuple = ()):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(_SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(history)")}
        if "img" not in cols:
            c.execute("ALTER TABLE history ADD COLUMN img TEXT")
        if "doc_type" not in cols:
            c.execute("ALTER TABLE history ADD COLUMN doc_type TEXT DEFAULT 'invoice'")
        cur = c.execute(sql, args)
        rows = cur.fetchall()
        c.commit()
        return rows, cur.lastrowid
    finally:
        c.close()


def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def _session(uid: int, name: str, email: str) -> dict:
    tok = secrets.token_urlsafe(32)
    q("INSERT INTO sessions VALUES(?,?,?)", (tok, uid, time.time()))
    return {"token": tok, "user": {"id": uid, "name": name, "email": email}}


def user_or_none(authorization: str | None = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    rows, _ = q("SELECT u.id,u.name,u.email FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?", (authorization[7:].strip(),))
    return dict(rows[0]) if rows else None


def current_user(user=Depends(user_or_none)):
    if not user:
        raise HTTPException(401, "Please sign in")
    return user


class Register(BaseModel):
    name: str
    email: str
    password: str


class Login(BaseModel):
    email: str
    password: str


@router.post("/auth/register")
def register(b: Register):
    email, name = b.email.strip().lower(), b.name.strip()
    if "@" not in email or len(name) < 2 or len(b.password) < 8:
        raise HTTPException(422, "Enter a name, a valid email and a password of at least 8 characters")
    salt = secrets.token_hex(16)
    try:
        _, uid = q("INSERT INTO users(email,name,salt,pw,created) VALUES(?,?,?,?,?)", (email, name, salt, _hash(b.password, salt), time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists")
    return _session(uid, name, email)


@router.post("/auth/login")
def login(b: Login):
    rows, _ = q("SELECT * FROM users WHERE email=?", (b.email.strip().lower(),))
    if not rows or not hmac.compare_digest(rows[0]["pw"], _hash(b.password, rows[0]["salt"])):
        raise HTTPException(401, "Wrong email or password")
    u = rows[0]
    return _session(u["id"], u["name"], u["email"])


@router.get("/auth/me")
def me(user=Depends(current_user)):
    rows, _ = q("SELECT created FROM users WHERE id=?", (user["id"],))
    return {**user, "created": rows[0]["created"] if rows else None}


class Rename(BaseModel):
    name: str


class Password(BaseModel):
    current: str
    new: str


class Delete(BaseModel):
    password: str


def _check_password(uid: int, password: str) -> None:
    rows, _ = q("SELECT salt,pw FROM users WHERE id=?", (uid,))
    if not rows or not hmac.compare_digest(rows[0]["pw"], _hash(password, rows[0]["salt"])):
        raise HTTPException(401, "Wrong password")


@router.patch("/auth/me")
def rename(b: Rename, user=Depends(current_user)):
    name = b.name.strip()
    if len(name) < 2:
        raise HTTPException(422, "Enter a name of at least 2 characters")
    q("UPDATE users SET name=? WHERE id=?", (name, user["id"]))
    return me({**user, "name": name})


@router.post("/auth/password")
def change_password(b: Password, authorization: str | None = Header(None), user=Depends(current_user)):
    """Needs the current password. Every OTHER session of the account is closed (a stolen token stops working); the one making the request stays open."""
    _check_password(user["id"], b.current)
    if len(b.new) < 8:
        raise HTTPException(422, "The new password must have at least 8 characters")
    salt = secrets.token_hex(16)
    q("UPDATE users SET salt=?, pw=? WHERE id=?", (salt, _hash(b.new, salt), user["id"]))
    q("DELETE FROM sessions WHERE user_id=? AND token<>?", (user["id"], (authorization or "")[7:].strip()))
    return {"ok": True}


@router.post("/auth/delete")
def delete_account(b: Delete, user=Depends(current_user)):
    """Deletes the account, its sessions, its history and the stored scans. Needs the password."""
    _check_password(user["id"], b.password)
    for r in q("SELECT img FROM history WHERE user_id=?", (user["id"],))[0]:
        Path(r["img"] or "").unlink(missing_ok=True)
    for t in ("history", "sessions"):
        q(f"DELETE FROM {t} WHERE user_id=?", (user["id"],))
    q("DELETE FROM users WHERE id=?", (user["id"],))
    return {"ok": True}


@router.post("/auth/logout")
def logout(authorization: str | None = Header(None), user=Depends(current_user)):
    q("DELETE FROM sessions WHERE token=?", (authorization[7:].strip(),))
    return {"ok": True}


def save_history(user: dict | None, result: dict, ms: int, data: bytes | None = None) -> None:
    """Called by /extract: stores the result AND the original scan for the signed-in user (nothing for anonymous calls)."""
    if not user:
        return
    F = result.get("fields", [])
    total = next((f["text"] for f in F if f["label"] in ("TOTAL", "AMOUNT")), "")           # invoices / receipts: TOTAL, cheques: AMOUNT
    conf = sum(f.get("confidence", 0) for f in F) / len(F) if F else 0.0
    _, hid = q("INSERT INTO history(user_id,file,created,n_fields,conf,n_items,total,ms,result,doc_type) VALUES(?,?,?,?,?,?,?,?,?,?)",
               (user["id"], result.get("file", "document"), time.time(), len(F), conf, len(result.get("items", [])), total, ms, json.dumps(result),
                result.get("doc_type", "invoice")))
    if data:
        ext = re.sub(r"[^a-z0-9.]", "", Path(result.get("file", "")).suffix.lower())[:6] or ".bin"
        path = SCANS / str(user["id"]) / f"{hid}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        q("UPDATE history SET img=? WHERE id=?", (str(path), hid))


@router.get("/history")
def history(user=Depends(current_user)):
    rows, _ = q("SELECT id,file,created,n_fields,conf,n_items,total,ms,COALESCE(doc_type,'invoice') AS doc_type,img IS NOT NULL AS has_img,result FROM history WHERE user_id=? ORDER BY id DESC LIMIT 300", (user["id"],))
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["n_warnings"] = len(json.loads(d.pop("result")).get("warnings", []))
        except (ValueError, TypeError):
            d["n_warnings"] = 0
        out.append(d)
    return out


@router.get("/history/{hid}")
def history_item(hid: int, user=Depends(current_user)):
    rows, _ = q("SELECT * FROM history WHERE id=? AND user_id=?", (hid, user["id"]))
    if not rows:
        raise HTTPException(404, "Not found")
    r = dict(rows[0])
    r["result"] = json.loads(r["result"])
    r["has_img"] = bool(r.pop("img"))
    return r


@router.delete("/history/{hid}")
def history_delete(hid: int, user=Depends(current_user)):
    for r in q("SELECT img FROM history WHERE id=? AND user_id=?", (hid, user["id"]))[0]:
        Path(r["img"] or "").unlink(missing_ok=True)
    q("DELETE FROM history WHERE id=? AND user_id=?", (hid, user["id"]))
    return {"ok": True}


@router.get("/stats")
def stats(user=Depends(current_user)):
    """Everything the profile and dashboard pages draw: totals, documents per day (30 days) and per day and type (90 days), confidence histogram, time and warnings by type,
    the checks that fail most, the fields read with the lowest confidence, a weekday x hour activity map and the latest documents."""
    rows, _ = q("SELECT id,file,created,conf,ms,COALESCE(doc_type,'invoice') AS doc_type,result FROM history WHERE user_id=? ORDER BY id", (user["id"],))
    by_type: dict[str, int] = {}
    per_day: dict[str, int] = {}
    per_day_type: dict[str, dict[str, int]] = {}
    sec_sum: dict[str, float] = {}
    warn_by_type: dict[str, int] = {}
    codes: dict[str, int] = {}
    field_conf: dict[str, list[float]] = {}
    hist = [0] * 10
    heat = [[0] * 24 for _ in range(7)]
    warn = 0
    now = time.time()
    for r in rows:
        t = r["doc_type"]
        by_type[t] = by_type.get(t, 0) + 1
        sec_sum[t] = sec_sum.get(t, 0.0) + (r["ms"] or 0) / 1000
        hist[min(9, int((r["conf"] or 0) * 10))] += 1
        lt = time.localtime(r["created"])
        heat[lt.tm_wday][lt.tm_hour] += 1
        day = time.strftime("%Y-%m-%d", lt)
        if r["created"] >= now - 30 * 86400:
            per_day[day] = per_day.get(day, 0) + 1
        if r["created"] >= now - 90 * 86400:
            per_day_type.setdefault(day, {})[t] = per_day_type.get(day, {}).get(t, 0) + 1
        try:
            res = json.loads(r["result"])
        except (ValueError, TypeError):
            continue
        if res.get("warnings"):
            warn += 1
            warn_by_type[t] = warn_by_type.get(t, 0) + 1
        for w in res.get("warnings", []):
            codes[w.get("code", "?")] = codes.get(w.get("code", "?"), 0) + 1
        for f in res.get("fields", []):
            if f.get("label") and f.get("confidence") is not None:
                field_conf.setdefault(f["label"], []).append(float(f["confidence"]))
    n = len(rows)
    weakest = sorted(({"label": k, "avg": round(sum(v) / len(v), 4), "n": len(v)} for k, v in field_conf.items() if len(v) >= 3), key=lambda d: d["avg"])[:8]
    recent = [{"id": r["id"], "file": r["file"], "doc_type": r["doc_type"], "created": r["created"], "conf": r["conf"] or 0.0} for r in reversed(rows[-6:])]
    return {"documents": n, "by_type": by_type, "avg_confidence": round(sum(r["conf"] or 0 for r in rows) / n, 4) if n else 0.0,
            "avg_seconds": round(sum(r["ms"] or 0 for r in rows) / n / 1000, 2) if n else 0.0, "with_warnings": warn, "per_day": per_day, "per_day_type": per_day_type,
            "confidence_hist": hist, "seconds_by_type": {t: round(sec_sum[t] / by_type[t], 2) for t in by_type}, "warnings_by_type": warn_by_type,
            "warning_codes": sorted(codes.items(), key=lambda kv: -kv[1])[:8], "weakest_fields": weakest, "heatmap": heat, "recent": recent,
            "first": rows[0]["created"] if n else None, "last": rows[-1]["created"] if n else None}


@router.get("/scans/{hid}/image")
def scan_image(hid: int, user=Depends(current_user)):
    rows, _ = q("SELECT img FROM history WHERE id=? AND user_id=?", (hid, user["id"]))
    if not rows or not rows[0]["img"] or not Path(rows[0]["img"]).exists():
        raise HTTPException(404, "No image stored for this scan")
    p = Path(rows[0]["img"])
    return Response(p.read_bytes(), media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")


@router.get("/dataset/export")
def dataset_export(user=Depends(current_user)):
    """The signed-in user's scans as a dataset: images/ + dataset.jsonl (fields, items, doc_type) + ONE csv PER TYPE
    (dataset_invoices.csv, dataset_receipts.csv): the columns of different document types are never mixed in one file."""
    rows, _ = q("SELECT id,file,created,img,result,COALESCE(doc_type,'invoice') AS doc_type FROM history WHERE user_id=? ORDER BY id", (user["id"],))
    buf, tables = io.BytesIO(), {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        lines = []
        for r in rows:
            res, name = json.loads(r["result"]), None
            if r["img"] and Path(r["img"]).exists():
                name = f"images/{r['id']}_{re.sub(r'[^A-Za-z0-9._-]', '_', r['file'])}"
                z.write(r["img"], name)
            lines.append(json.dumps({"id": r["id"], "doc_type": r["doc_type"], "file": r["file"], "image": name, "created": r["created"],
                                     "fields": res.get("fields", []), "items": res.get("items", [])}, ensure_ascii=False))
            tables.setdefault(r["doc_type"], []).append({"id": r["id"], "image": name or "", **res.get("row", {})})
        z.writestr("dataset.jsonl", "\n".join(lines))
        for doc_type, table in tables.items():
            keys = list(dict.fromkeys(k for t in table for k in t))
            out = io.StringIO()
            w = csv.DictWriter(out, keys, extrasaction="ignore")
            w.writeheader(); w.writerows(table)
            z.writestr(f"dataset_{doc_type}s.csv", "\ufeff" + out.getvalue())
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="documind_dataset.zip"'})


@router.get("/dataset")
def dataset():
    """Public: the training report written by documind.training (corpus, scores per field)."""
    p = REPORTS_DIR / "training_report.json"
    if not p.exists():
        raise HTTPException(404, "No training report yet: run python -m documind.training.train")
    r = json.loads(p.read_text(encoding="utf-8"))
    r.pop("details", None)
    return r
