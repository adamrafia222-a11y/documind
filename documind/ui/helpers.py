"""Everything the Streamlit page needs that is not drawing widgets: reading a document (here or through the API), drawing the detected boxes,
tables, and one CSV per document type (never merged)."""
from __future__ import annotations

import colorsys
import hashlib
import io
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..config import PDF_DPI
from ..inference.export import rows_to_csv

TYPES = ("invoice", "receipt", "cheque", "contract")
TYPE_LABEL = {"invoice": "Invoice", "receipt": "Receipt", "cheque": "Cheque", "contract": "Contract"}
CSV_NAMES = {"invoice": "invoices.csv", "receipt": "receipts.csv", "cheque": "cheques.csv", "contract": "contracts.csv"}
TYPE_COLOR = {"invoice": "#3b82f6", "receipt": "#f59e0b", "cheque": "#10b981", "contract": "#8b5cf6"}


# ------------------------------------------------------------------------------------------------ reading a document
def read_local(service, data: bytes, filename: str, doc_type: str = "auto") -> dict:
    """In this process (the models are loaded here). Raises the service's own errors (ModelNotFound, ValueError)."""
    return service.extract(data, filename, doc_type)


def read_api(base_url: str, data: bytes, filename: str, doc_type: str = "auto", timeout: float = 300.0, token: str | None = None) -> dict:
    """Through the FastAPI server (POST /extract). Raises RuntimeError with the server's explanation."""
    import requests
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.post(base_url.rstrip("/") + "/extract", params={"doc_type": doc_type}, files={"file": (filename, data)}, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach the API at {base_url}: {exc}") from exc
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


def api_health(base_url: str, timeout: float = 4.0) -> dict | None:
    import requests
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ the picture with its boxes
def page_image(data: bytes, filename: str, page: int = 1) -> Image.Image:
    """The picture the boxes refer to: a photo with its camera rotation applied, or one page of a PDF at the resolution the reader used."""
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        from ..preprocessing import pdf_to_images
        pages = pdf_to_images(data, PDF_DPI)
        return Image.fromarray(pages[max(0, min(page, len(pages)) - 1)][:, :, ::-1])
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def label_color(label: str) -> str:
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.85)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def draw_boxes(img: Image.Image, fields: list[dict], items: list[dict] | None = None, page: int = 1, max_width: int = 1100) -> Image.Image:
    """The picture (shrunk to `max_width`) with a coloured box and a name on every detected field of the page; item rows in a thin grey box."""
    k = min(1.0, max_width / img.width)
    out = img.convert("RGB").resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
    d = ImageDraw.Draw(out, "RGBA")
    try:
        font = ImageFont.load_default(size=max(11, int(out.width / 70)))
    except TypeError:                                                     # old Pillow without a font size
        font = ImageFont.load_default()
    for it in items or []:
        if it.get("page", 1) == page and it.get("bbox"):
            x0, y0, x1, y1 = (v * k for v in it["bbox"])
            d.rectangle([x0, y0, x1, y1], outline=(120, 120, 120, 200), width=1)
    for f in fields:
        if f.get("page", 1) != page or not f.get("bbox"):
            continue
        c = label_color(f["label"])
        for b in f.get("boxes") or [f["bbox"]]:                          # a value wrapped over two lines has one box per line
            d.rectangle([v * k for v in b], outline=c, width=2, fill=c + "26")
        x0, y0, x1, y1 = (v * k for v in (f.get("boxes") or [f["bbox"]])[0])
        tag = f["label"]
        tw = d.textlength(tag, font=font)
        top = y0 - font.size - 4 if y0 - font.size - 4 > 0 else y1 + 1
        d.rectangle([x0, top, x0 + tw + 6, top + font.size + 3], fill=c)
        d.text((x0 + 3, top + 1), tag, fill="white", font=font)
    return out


# ------------------------------------------------------------------------------------------------ tables
def fields_table(result: dict) -> list[dict]:
    return [{"field": f["label"], "text": f["text"], "confidence": f"{round(f.get('confidence', 0) * 100)} %", "page": f.get("page", 1)}
            for f in sorted(result.get("fields", []), key=lambda f: (f.get("page", 1), f["bbox"][1], f["bbox"][0]))]


def items_table(result: dict) -> list[dict]:
    cols = ["qty", "description", "unit_price", "amount"]
    rows = [{c: it.get(c, "") for c in cols} for it in result.get("items", [])]
    return [{c: v for c, v in r.items() if any(x.get(c) for x in rows)} for r in rows]


def summary_rows(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        r = e.get("result")
        if r is None:
            out.append({"file": e["name"], "type": "-", "fields": 0, "confidence": "-", "warnings": None, "status": "ERROR: " + e.get("error", "")[:80]})
            continue
        conf = [f.get("confidence", 0) for f in r.get("fields", [])]
        out.append({"file": e["name"], "type": TYPE_LABEL.get(r.get("doc_type", "invoice"), r.get("doc_type")), "fields": len(r.get("fields", [])),
                    "confidence": f"{round(100 * sum(conf) / len(conf))} %" if conf else "-", "warnings": len(r.get("warnings", [])),
                    "status": "ok" if not r.get("warnings") else "check"})
    return out


# ------------------------------------------------------------------------------------------------ CSV: one file per document type
def type_of(result: dict) -> str:
    return result.get("doc_type") or "invoice"


def csv_by_type(results: list[dict]) -> dict[str, bytes]:
    """{file name: csv bytes}, one file per document type present; each uses ITS OWN columns (result['columns'])."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(type_of(r), []).append(r)
    return {CSV_NAMES.get(t, f"{t}.csv"): rows_to_csv([r["row"] for r in rs], rs[0].get("columns") or list(rs[0]["row"])) for t, rs in groups.items()}


def zip_files(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ history and profile (pure functions)
def format_when(ts: float | None) -> str:
    import time as _t
    return _t.strftime("%d %b %Y, %H:%M", _t.localtime(ts)) if ts else "-"


def filter_history(rows: list[dict], types: list[str] | None = None, query: str = "", only_warnings: bool = False) -> list[dict]:
    """The history rows (as the backend lists them) that match: one of the types, a file name containing the text, and (optionally) at least one warning."""
    q = (query or "").strip().lower()
    return [r for r in rows if (not types or r.get("doc_type", "invoice") in types) and (not q or q in r.get("file", "").lower()) and (not only_warnings or r.get("n_warnings", 0) > 0)]


def history_frame(rows: list[dict]) -> list[dict]:
    """Rows for the history table (the column names are the headings the person reads)."""
    return [{"ID": r["id"], "File": r["file"], "Type": TYPE_LABEL.get(r.get("doc_type", "invoice"), r.get("doc_type")), "When": format_when(r.get("created")),
             "Fields": r.get("n_fields", 0), "Confidence": round(100 * float(r.get("conf") or 0.0)), "Warnings": r.get("n_warnings", 0), "Total": r.get("total") or "",
             "Seconds": round((r.get("ms") or 0) / 1000, 1)} for r in rows]


def per_day_series(per_day: dict[str, int], days: int = 30, today=None) -> list[tuple[str, int]]:
    """[(date, documents)] for each of the last `days` days, zero where nothing was read (so the chart has no holes)."""
    import datetime as _d
    end = today or _d.date.today()
    out = []
    for k in range(days - 1, -1, -1):
        day = (end - _d.timedelta(days=k)).isoformat()
        out.append((day, int(per_day.get(day, 0))))
    return out


def average_confidence(result: dict) -> float:
    conf = [f.get("confidence", 0) for f in result.get("fields", [])]
    return sum(conf) / len(conf) if conf else 0.0


# ------------------------------------------------------------------------------------------------ dashboard (pure functions)
WARNING_LABELS = {"amount_mismatch": "Amount in figures ≠ amount in words", "amount_not_checked": "Amounts could not be compared", "payment_mismatch": "Monthly payment ≠ loan formula",
                  "iban_length": "IBAN has the wrong length", "micr_mismatch": "Cheque number ≠ MICR line", "date_invalid": "Invalid date", "account_length": "Account number has the wrong length",
                  "no_signature": "No signature found", "no_signature_lender": "Lender's signature missing", "no_signature_borrower": "Borrower's signature missing",
                  "missing_field": "A field was not found", "first_payment_early": "First payment not after the contract date", "signed_before_contract": "Signed before the contract date",
                  "nothing_found": "Nothing detected", "first_page_only": "Only the first page was read"}


def warning_label(code: str) -> str:
    return WARNING_LABELS.get(code) or code.replace("_", " ").capitalize()


def range_summary(stats: dict, days: int, today=None) -> dict:
    """What the dashboard shows for the last `days` days, from stats['per_day_type'] (kept for 90 days):
    {"docs", "prev" (the `days` days before, None when it is not known), "spark" (documents per day), "by_type", "rows" [{"date", "type", "documents"}]}."""
    import datetime as _d
    end = today or _d.date.today()
    per = stats.get("per_day_type", {})
    day = lambda k: (end - _d.timedelta(days=k)).isoformat()
    spark, by_type, rows = [], {}, []
    for k in range(days - 1, -1, -1):
        counts = per.get(day(k), {})
        spark.append(sum(counts.values()))
        for t, n in counts.items():
            by_type[t] = by_type.get(t, 0) + n
            rows.append({"date": day(k), "type": TYPE_LABEL.get(t, t), "documents": n})
    prev = sum(sum(per.get(day(k), {}).values()) for k in range(days, 2 * days)) if 2 * days <= 90 else None
    return {"docs": sum(spark), "prev": prev, "spark": spark, "by_type": by_type, "rows": rows}


def delta_chip(docs: int, prev: int | None, days: int) -> tuple[str, str] | None:
    """('▲ 4 vs the previous 30 days', 'up') / ('▼ 2 ...', 'down') / ('= same ...', 'flat'); None when there is nothing to compare with."""
    if prev is None or (docs == 0 and prev == 0):
        return None
    d = docs - prev
    return (f"{'▲' if d > 0 else '▼' if d < 0 else '='} {abs(d) if d else 'same'} vs the previous {days} days", "up" if d > 0 else "down" if d < 0 else "flat")
