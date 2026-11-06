"""Rule-based layer that complements the LightGBM model.

The model is trained on FATURA's 50 templates and ignores the line-item table (TABLE is background in training).
This module reads, from the OCR words only (no training needed, works on any layout):

* the LINE ITEMS table: header row (Qty / Description / Unit price / Amount ...), then one row per item,
  columns found from the header positions; amounts are cross-checked with  qty x unit price  and with the Subtotal;
* the AMOUNTS by their label (Subtotal, Tax / VAT / GST, Discount, Total, Amount due) and a few key: value pairs
  (invoice number, PO number, dates).

`apply_rules` only fills what the model missed (an amount field without an amount, an empty field): a value the
model already found is kept.
"""
from __future__ import annotations

import copy
import difflib
import itertools
import re
from statistics import median

from ..features.builder import DATE_RE, parse_amount

_NUM = re.compile(r"^[-(]?[$€£]?\d[\d,. ]*\)?[$€£]?$")
_HDR = {
    "desc": re.compile(r"^(descriptions?|items?|details?|products?|particulars?|services?|designation|libell[eé]|article)$"),
    "qty": re.compile(r"^(qty|quantity|qte|quantit[eé]|hrs|hours|units|pcs)$"),
    "price": re.compile(r"^(price|rate|unit|cost|prix|pu)$"),
    "amount": re.compile(r"^(amount|total|montant|ext)$"),
    "tax": re.compile(r"^(tax|vat|gst|tva|taxe)$"),
    "id": re.compile(r"^(id|#|no|n°|nº|ref|sku|code|pos|nr)$"),
}
_STOP = {"subtotal", "sub", "total", "tax", "vat", "gst", "tva", "discount", "balance", "amount", "shipping", "payment",
         "terms", "notes", "note", "thank", "thanks", "sales", "grand", "bank", "net", "paid", "due", "signature"}


_CUR = r"(?:US\$|CA\$|A\$|USD|EUR|GBP|MAD|CAD|AUD|CHF|INR|JPY|DHS?|RS\.?|[$€£¥₹])"
_CUR_PRE = re.compile(rf"^(?:[A-Za-z]{{1,3}}[$€£¥₹]|{_CUR})\s*", re.I)
_CUR_SUF = re.compile(rf"\s*{_CUR}$", re.I)


def _bare(t: str) -> tuple[str, bool]:
    """'US$633.36' -> ('633.36', True): the amount without its currency mark, and whether one was there."""
    t = t.strip()
    b = _CUR_SUF.sub("", _CUR_PRE.sub("", t))
    return b, b != t


def _num(t: str) -> bool:
    b, _ = _bare(t)
    return bool(_NUM.match(b)) and "%" not in b


def _money(t: str) -> bool:
    b, had = _bare(t)
    return _num(t) and (had or bool(re.search(r"[.,]\d{1,2}\)?$", b)))


def _fmt(v: float) -> str:
    return f"{v:.2f}"


_PRE_AMT = re.compile(rf"^[-(]?(?:[A-Za-z]{{1,3}}[$€£¥₹]|{_CUR})?\d{{1,3}}$", re.I)
_POST_AMT = re.compile(r"^\d{3}(?:[.,]\d{1,2})?\)?$")


def _clean(words: list[dict]) -> list[dict]:
    """Drop table-grid noise ("|", "_", "[")."""
    ws = []
    for w in words:
        t = w["text"].strip("|¦_[]{}")                   # grid lines read as characters; "#", ":", "%" are kept
        if t:
            ws.append({**w, "text": t})
    return ws


def _join_thousands(row: list[dict]) -> list[dict]:
    """Within one text line, join amounts split by a thousands space ("$2" "500.00" -> "$2 500.00")."""
    out: list[dict] = []
    for w in row:
        a = out[-1] if out else None
        if (a and _PRE_AMT.match(a["text"]) and _POST_AMT.match(w["text"]) and abs((a["y0"] + a["y1"]) - (w["y0"] + w["y1"])) <= (a["y1"] - a["y0"])
                and 0 <= w["x0"] - a["x1"] <= 0.8 * (a["y1"] - a["y0"])):
            out[-1] = {**a, "text": a["text"] + " " + w["text"], "x1": w["x1"], "y0": min(a["y0"], w["y0"]), "y1": max(a["y1"], w["y1"]),
                       "conf": min(a.get("conf", 100), w.get("conf", 100))}
        else:
            out.append(w)
    return out


def _lines(words: list[dict]) -> list[dict]:
    words = _clean(words)
    if not words:
        return []
    h = median(max(1.0, w["y1"] - w["y0"]) for w in words)
    out: list[dict] = []
    for w in sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2, w["x0"])):
        cy = (w["y0"] + w["y1"]) / 2
        if out and abs(cy - out[-1]["cy"]) <= 0.6 * h:
            out[-1]["words"].append(w)
            out[-1]["cy"] = sum((x["y0"] + x["y1"]) / 2 for x in out[-1]["words"]) / len(out[-1]["words"])
        else:
            out.append({"cy": cy, "words": [w]})
    for ln in out:
        ln["words"] = _join_thousands(sorted(ln["words"], key=lambda w: w["x0"]))
        ln["text"] = " ".join(w["text"] for w in ln["words"])
        ln["x0"] = min(w["x0"] for w in ln["words"]); ln["x1"] = max(w["x1"] for w in ln["words"])
        ln["y0"] = min(w["y0"] for w in ln["words"]); ln["y1"] = max(w["y1"] for w in ln["words"])
    return out


def _box(ws: list[dict]) -> list[float]:
    return [round(min(w["x0"] for w in ws), 1), round(min(w["y0"] for w in ws), 1),
            round(max(w["x1"] for w in ws), 1), round(max(w["y1"] for w in ws), 1)]


# --------------------------------------------------------------------------- line items
_CANON = {"description": "desc", "descriptions": "desc", "particulars": "desc", "quantity": "qty", "unitprice": "price",
          "price": "price", "amount": "amount", "montant": "amount"}


def _cat(t: str) -> str | None:
    t = t.lower().strip(":.,#|_[]")
    hit = next((k for k, rx in _HDR.items() if rx.match(t)), None)
    if hit or len(t) < 5:
        return hit
    close = difflib.get_close_matches(t, list(_CANON), n=1, cutoff=0.8)          # OCR typos: "nitprice", "Descripton"
    return _CANON[close[0]] if close else None


def _find_header(lines: list[dict]) -> int | None:
    for i, ln in enumerate(lines):
        cats = {c for w in ln["words"] if (c := _cat(w["text"])) and c != "id"}
        if ("desc" in cats and (len(cats) >= 3 or cats & {"price", "amount"})) or len(cats - {"desc"}) >= 3:
            return i
    return None


_TAXWORDS = {"cgst", "sgst", "igst", "gst", "tax", "vat", "tva", "taxe"}


def _cell_cat(text: str) -> str:
    """Meaning of a header cell: id, desc, qty, price, amount, tax, taxrate, taxamt, taxable, hsn, discount, other."""
    t = text.lower()
    w = set(re.findall(r"[a-zà-ÿ]+|%", t))
    if re.fullmatch(r"\W*(sr\W*no|sl\W*no|s\W*no|no|id|ref|sku|pos|nr|#|n°)\W*", t):
        return "id"
    if w & {"description", "descriptions", "item", "items", "details", "detail", "particulars", "product", "products",
            "service", "services", "designation", "libellé", "libelle", "article"}:
        return "desc"
    tax = bool(w & _TAXWORDS) or any(re.fullmatch(r"[a-z]{0,2}gs[ti]", x) for x in w)
    if tax and ("%" in w or "rate" in w):
        return "taxrate"
    if tax and (w & {"amount", "amt", "montant", "value"}):
        return "taxamt"
    if "taxable" in w:
        return "taxable"
    if tax:
        return "tax"
    if w & {"quantity", "qty", "qte", "quantite", "quantité", "hrs", "hours", "units", "pcs", "nos"}:
        return "qty"
    if w & {"price", "rate", "cost", "prix", "pu"}:
        return "price"
    if w & {"amount", "total", "montant", "ext", "subtotal"}:
        return "amount"
    if w & {"hsn", "sac"}:
        return "hsn"
    if w & {"discount", "disc", "remise"}:
        return "discount"
    return "other"


_CANON_KEY = {"desc": "description", "qty": "qty", "price": "unit_price", "amount": "amount", "id": "id"}


def _cell_key(text: str, cat: str) -> str:
    base = _CANON_KEY.get(cat)
    if base:
        return base
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text.lower())             # drop the unit marks "(₹)", "(%)", OCR noise "(96)"
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    if cat == "taxrate" and not t.endswith("pct"):
        t += "_pct"
    return t or cat


def _header_block(lines: list[dict], hi: int, width: float) -> tuple[int, int]:
    """Header cells are often 2-3 lines tall ("Unit" / "Price" / "(₹)"): take the lines around the detected one that hold
    only words (no digits) over the same columns."""
    core = lines[hi]
    hh = median(max(1.0, w["y1"] - w["y0"]) for w in core["words"])
    lo, up = core["x0"] - 0.03 * width, core["x1"] + 0.03 * width

    def mark(t: str) -> bool:                        # "(₹)" / "(%)" as the OCR may read them: "(96)", "3)", "(8)", "®"
        return len(t) <= 4 and bool(re.fullmatch(r"[\(\[]?[\d%®©«»]{0,3}[\)\]]?", t)) and bool(re.search(r"[\(\)\[\]®©«»]", t))

    def ok(ln: dict) -> bool:
        return (all(not re.search(r"\d", w["text"]) or mark(w["text"]) for w in ln["words"])
                and sum(lo <= (w["x0"] + w["x1"]) / 2 <= up for w in ln["words"]) >= 0.6 * len(ln["words"]))
    a = z = hi
    while a > 0 and len(lines[a - 1]["words"]) >= 2 and ok(lines[a - 1]) and lines[a]["y0"] - lines[a - 1]["y1"] <= 1.2 * hh:
        a -= 1
    while z + 1 < len(lines) and ok(lines[z + 1]) and lines[z + 1]["y0"] - lines[z]["y1"] <= 1.2 * hh:
        z += 1
    return a, z


def _columns(block: list[dict]) -> list[dict]:
    hh = median(max(1.0, w["y1"] - w["y0"]) for ln in block for w in ln["words"])
    order = {id(w): (i, w["x0"]) for i, ln in enumerate(block) for w in ln["words"]}
    cl: list[dict] = []
    for w in sorted((w for ln in block for w in ln["words"]), key=lambda w: w["x0"]):
        for c in cl:
            ov = min(w["x1"], c["x1"]) - max(w["x0"], c["x0"])
            if ov > 0.35 * min(w["x1"] - w["x0"], c["x1"] - c["x0"]):
                c["w"].append(w); c["x0"] = min(c["x0"], w["x0"]); c["x1"] = max(c["x1"], w["x1"])
                break
        else:
            cl.append({"x0": w["x0"], "x1": w["x1"], "w": [w]})
    cl.sort(key=lambda c: c["x0"])
    merged: list[dict] = []
    for c in cl:                                     # two words of one header cell on one line ("UNIT PRICE")
        if merged and c["x0"] - merged[-1]["x1"] <= 0.8 * hh and {order[id(w)][0] for w in c["w"]} & {order[id(w)][0] for w in merged[-1]["w"]}:
            merged[-1]["w"] += c["w"]; merged[-1]["x1"] = max(merged[-1]["x1"], c["x1"])
        else:
            merged.append(c)
    for c in merged:
        c["text"] = " ".join(w["text"] for w in sorted(c["w"], key=lambda w: order[id(w)]))
        c["cat"] = _cell_cat(c["text"])
    amount = [c for c in merged if c["cat"] == "amount"]
    for c in amount[:-1]:                            # several "total-like" columns: the last one is the line amount
        c["cat"] = "other"
    seen: dict[str, int] = {}
    for c in merged:
        k = _cell_key(c["text"], c["cat"]); seen[k] = seen.get(k, 0) + 1
        c["key"] = k if seen[k] == 1 else f"{k}_{seen[k]}"
    return merged


def _cdist(w: dict, c: dict) -> float:
    return min(abs(w["x1"] - c["x1"]), abs((w["x0"] + w["x1"]) / 2 - (c["x0"] + c["x1"]) / 2), abs(w["x0"] - c["x0"]))


_PCT = re.compile(r"^\(?\d+(?:[.,]\d+)?\s?%\)?$")


def _extract_by_header(words: list[dict], width: float, height: float, page: int = 1) -> tuple[list[dict], int, dict]:
    """-> (items, index of the last table line, {"columns": [...], "footer": {...}}).
    The header may be several lines tall and have any columns (qty, unit price, HSN, tax %, tax amount, total ...):
    every column becomes a key of the item (qty, description, unit_price, amount + the others under their header name)."""
    lines = _lines(words)
    hi = _find_header(lines)
    if hi is None:
        return [], -1, {}
    a, z = _header_block(lines, hi, width)
    cols = _columns(lines[a:z + 1])
    left = min(c["x0"] for c in cols)
    desc_c = next((c for c in cols if c["cat"] == "desc"), None)
    if desc_c is not None and not any(c["cat"] == "qty" for c in cols):      # "QTY" misread by the OCR ("ary")
        odd = [c for c in cols if c["cat"] == "other" and c["x1"] <= desc_c["x0"] and len(c["text"]) <= 6]
        if odd:
            odd[0]["cat"], odd[0]["key"] = "qty", "qty"
    id_c = next((c for c in cols if c["cat"] == "id"), None)
    has_qty = any(c["cat"] == "qty" for c in cols)
    nums = [c for c in cols if c["cat"] not in ("desc", "id")]
    h = median(max(1.0, w["y1"] - w["y0"]) for w in words)
    labels = [{"key": c["key"], "label": c["text"], "cat": c["cat"]} for c in cols]
    for k, lab in (("description", "Description"), ("qty", "Qty"), ("unit_price", "Unit price"), ("amount", "Amount")):
        if not any(c["key"] == k for c in cols):
            labels.append({"key": k, "label": lab})

    def assign(ws: list[dict]) -> tuple[dict, list[dict]]:
        cells: dict[str, dict] = {}
        rest: list[dict] = []
        ws = list(ws)
        if id_c is not None and desc_c is not None and ws:
            lead = ws[0]
            if lead["x1"] <= desc_c["x0"] + 1 and lead["x0"] >= id_c["x0"] - 0.05 * width and len(lead["text"]) <= 5:
                cells[id_c["key"]] = lead; ws = ws[1:]
        for w in ws:
            if _num(w["text"]) or _PCT.match(w["text"]):
                d, c = min(((_cdist(w, c), c) for c in nums), key=lambda x: x[0], default=(1e9, None))
                if c is not None and d <= max(0.7 * (c["x1"] - c["x0"]), 0.04 * width) and c["key"] not in cells:
                    cells[c["key"]] = w; continue
            rest.append(w)
        if not has_qty and "qty" not in cells and len(rest) >= 2 and re.fullmatch(r"\d{1,4}", rest[0]["text"]):
            cells["qty"] = rest.pop(0)
        return cells, rest

    items: list[dict] = []
    footer: dict[str, str] = {}
    footer_bbox: list[float] = []
    last_i, prev_cy = z, lines[z]["cy"]
    for j in range(z + 1, len(lines)):
        ln = lines[j]; ws = ln["words"]
        first = re.sub(r"[^a-z]", "", ws[0]["text"].lower())
        pitch = (items[-1]["_cy"] - items[0]["_cy"]) / (len(items) - 1) if len(items) > 1 else 2.6 * h
        if items and first.startswith(("total", "grand", "subtotal")) and not _num(ws[0]["text"]) and ln["x0"] <= left + 0.2 * width:
            cells, _ = assign(ws[1:])                # the "Total" row printed inside the table
            footer = {k: w["text"] for k, w in cells.items()}
            last_i, footer_bbox = j, _box(ws)
            break
        if first in _STOP and not _num(ws[0]["text"]) and ln["x0"] > left + 0.2 * width:
            break
        if ln["cy"] - prev_cy > 2.2 * max(pitch, 2.2 * h):
            break
        if ln["x0"] > left + 0.3 * width:            # a row starting in the right part of the page = totals block
            break
        right = [w for w in ws if w["x0"] > left + 0.3 * width]
        if right and any(_key_of(" ".join(w["text"] for w in seg)) for seg in _segments(right)):
            break                                    # "Stamp and signature ........ Subtotal 591.00" on one row
        cells, rest = assign(ws)
        desc = " ".join(w["text"] for w in rest).strip()
        if not cells:
            if desc and items and ln["x0"] < left + 0.5 * width:              # wrapped description
                items[-1]["description"] += " " + desc
                bb = items[-1]["bbox"]
                items[-1]["bbox"] = _box([{"x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3]}] + ws)
                last_i, prev_cy = j, ln["cy"]
                continue
            break
        it = {c["key"]: (cells[c["key"]]["text"] if c["key"] in cells else "") for c in cols if c["cat"] != "desc"}
        it["description"] = desc
        for k in ("qty", "unit_price", "amount"):
            it.setdefault(k, cells[k]["text"] if k in cells else "")
        if "qty" in cells and "qty" not in it:
            it["qty"] = cells["qty"]["text"]
        it = {c["key"]: it.get(c["key"], "") for c in labels if c["key"] in it} | {k: v for k, v in it.items()}
        it.update({"bbox": _box(ws), "page": page, "_cy": ln["cy"]})
        items.append(it); last_i, prev_cy = j, ln["cy"]
    return items, last_i, {"columns": labels, "footer": footer, "footer_bbox": footer_bbox}


# --------------------------------------------------------------------------- line items read from the data itself
# The header-driven reader above needs the header row to be read correctly and the columns to sit under their header.
# This one does not: it finds the block of rows that end with numbers, clusters those numbers into columns, and finds
# which column is the quantity, the unit price and the amount by arithmetic (qty x price = amount), whatever the header says.
_ISNUM = lambda t: _num(t) or bool(_PCT.match(t))
_TOTAL_WORDS = {"subtotal", "sub", "total", "grand", "balance", "net", "discount", "shipping", "tax", "vat", "gst", "tva", "amount"}


def _val(t: str) -> float | None:
    b, _ = _bare(t)
    s = re.sub(r"[^\d.,\-()]", "", b)
    neg = s.startswith(("(", "-"))
    s = s.strip("()-")
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


_CUR_ONLY = re.compile(rf"^{_CUR}$", re.I)


def _merge_currency(ws: list[dict]) -> list[dict]:
    """"421.08" "MAD" -> "421.08 MAD" and "$" "464.75" -> "$ 464.75": a currency mark printed apart from its amount belongs to it."""
    out: list[dict] = []
    for w in ws:
        if out and _CUR_ONLY.match(w["text"]) and _num(out[-1]["text"]) and not _CUR_ONLY.match(out[-1]["text"]):
            a = out.pop(); out.append({**a, "text": a["text"] + " " + w["text"], "x1": max(a["x1"], w["x1"])})
        elif out and _CUR_ONLY.match(out[-1]["text"]) and _num(w["text"]):
            a = out.pop(); out.append({**w, "text": a["text"] + w["text"], "x0": min(a["x0"], w["x0"])})
        else:
            out.append(w)
    return out


def _junk(t: str) -> bool:
    """A cell the OCR could not read (a "9%" read as "Ho" or "vay"): short and not a number."""
    return len(t) <= 3 and not _ISNUM(t)


_STRAY = re.compile(r"^(?:[\u2014\u2013~_|'\"`]+-?|--+)(?=[$€£¥₹(]?\d)|(?<=\d)[\u2014\u2013~_|'\"`]+$")


def _unpunct(w: dict) -> dict:
    """Specks the OCR glued to a number (\u2014-18,360.33, 12.50~) are not part of it."""
    t = _STRAY.sub("", w["text"])
    return {**w, "text": t} if t != w["text"] and t else w


def _tail(ws: list[dict]) -> tuple[list[dict], list[dict]]:
    """The run of numbers at the end of a text line. One or two unreadable cells inside the run (see _junk) do not end it."""
    k = len(ws)
    while k > 0:
        if _ISNUM(ws[k - 1]["text"]):
            k -= 1
            continue
        j = k
        while j > 0 and k - j < 2 and _junk(ws[j - 1]["text"]):
            j -= 1
        if j < k and j > 0 and _ISNUM(ws[j - 1]["text"]) and k < len(ws):
            k = j                                                    # junk sits between two numbers: part of the number run
            continue
        break
    return ws[:k], ws[k:]


def _is_total_line(head: list[dict], tail: list[dict]) -> bool:
    if not head:
        return False
    first = re.sub(r"[^a-z]", "", head[0]["text"].lower())
    if first in ("total", "subtotal", "grandtotal") and len(head) <= 3:
        return True                                         # a "Total" row closes the table, however many columns it fills
    if first not in _TOTAL_WORDS or len(tail) > 2 or len(head) > 3:
        return False
    return "total" in first or len(head) == 1 or any(c in w["text"] for w in head for c in ":%(")


def _score_items(items: list[dict]) -> float:
    s = 0.0
    for it in items:
        a, q, p = _val(it.get("amount", "")), _val(it.get("qty", "")), _val(it.get("unit_price", ""))
        s += 1 if a is not None else -1
        if None not in (a, q, p) and abs(q * p - a) <= max(0.011, 0.006 * abs(a)):
            s += 2
        if re.search(r"\d[.,]\d{2}\b", it.get("description", "")):
            s -= 2
    return s


def _extract_by_data(words: list[dict], width: float, height: float, page: int = 1) -> tuple[list[dict], int, dict]:
    lines = _lines(words)
    if len(lines) < 3:
        return [], -1, {}
    h = median(max(1.0, w["y1"] - w["y0"]) for w in words)
    info = []
    for ln in lines:
        ln["words"] = _merge_currency([_unpunct(w) for w in ln["words"]])
        head, tail = _tail(ln["words"])
        info.append((head, tail, _is_total_line(head, tail)))
    # 1. blocks of consecutive rows that end with at least two numbers (wrapped description lines may sit between them)
    blocks, cur, prev_cy = [], None, None
    for i, (head, tail, tot) in enumerate(info):
        ln = lines[i]
        row = (not tot) and bool(head) and len(tail) >= 2
        # how far apart table rows may be: from the pitch seen so far (rows with wrapped cells are much taller than the text)
        lim = max(3.8 * h, 1.7 * median(cur["gaps"])) if cur is not None and cur.get("gaps") else 6.0 * h
        near = prev_cy is not None and ln["cy"] - prev_cy <= lim
        if row and cur is not None and near:
            cur.setdefault("gaps", []).append(ln["cy"] - prev_cy); cur["rows"].append(i); cur["last"] = i; prev_cy = ln["cy"]
        elif row:
            cur = {"rows": [i], "last": i}; blocks.append(cur); prev_cy = ln["cy"]
        elif cur is not None and near and not tot and head and not tail and ln["x0"] < lines[cur["rows"][0]]["x0"] + 0.5 * width:
            cur.setdefault("wraps", {}).setdefault(cur["rows"][-1], []).append(i); cur["last"] = i; prev_cy = ln["cy"]
        else:
            cur = None
    good = []
    for b in blocks:
        n = len(b["rows"])
        after_total = any(info[j][2] for j in range(b["last"] + 1, min(len(lines), b["last"] + 9)))
        money = sum(any(_money(w["text"]) for w in info[i][1]) for i in b["rows"])
        if (n >= 2 or after_total) and (after_total or n >= 3 or money >= 0.5 * n) and money >= 1:
            good.append(b)
    if not good:
        return [], -1, {}
    blk = max(good, key=lambda b: (len(b["rows"]), b["rows"][0] * -1))
    rows = blk["rows"]; n = len(rows)
    # 2. the leading number of each row: a running index (id) or a quantity printed before the description
    lead_re = re.compile(r"\d{1,3}[.)]?")
    lead_rows = [i for i in rows if lead_re.fullmatch(info[i][0][0]["text"]) and len(info[i][0]) >= 2]
    lead_ok = len(lead_rows) >= max(1, -(-n * 6 // 10))           # most rows start with a small number (a row may have lost it to OCR)
    lead_vals = [int(re.sub(r"\D", "", info[i][0][0]["text"])) for i in lead_rows]
    seq = lead_ok and lead_vals == sorted(lead_vals)
    id_ws = {i: info[i][0][0] for i in lead_rows} if (lead_ok and seq) else {}
    qty_lead = {i: info[i][0][0] for i in lead_rows} if (lead_ok and not seq) else {}
    # 3. cluster the numbers at the end of the rows into columns
    gap = 0.012 * width
    toks = sorted(((i, w) for i in rows for w in info[i][1]), key=lambda t: (t[1]["x0"] + t[1]["x1"]) / 2)
    cols: list[dict] = []
    for i, w in toks:
        for c in cols:
            if w["x0"] <= c["x1"] + gap and w["x1"] >= c["x0"] - gap and all(i != r for r, _ in c["t"]):
                c["x0"], c["x1"] = min(c["x0"], w["x0"]), max(c["x1"], w["x1"]); c["t"].append((i, w)); break
        else:
            cols.append({"x0": w["x0"], "x1": w["x1"], "t": [(i, w)]})
    need = 1 if n == 1 else -(-n * 5 // 10)
    keep = sorted((c for c in cols if len(c["t"]) >= need), key=lambda c: c["x0"])
    if not keep:
        return [], -1, {}
    dropped = {id(w) for c in cols if c not in keep for _, w in c["t"]}
    for c in keep:
        c["v"] = {i: w for i, w in c["t"]}
        c["pct"] = sum(bool(_PCT.match(w["text"])) for _, w in c["t"]) >= 0.5 * len(c["t"])
    # 4. roles: qty x unit price = amount, found by arithmetic
    numeric = lambda c: sum(_val(w["text"]) is not None for _, w in c["t"]) >= 0.5 * len(c["t"])
    money_cols = [k for k, c in enumerate(keep) if not c["pct"] and numeric(c)]
    role: dict[int, str] = {}
    best, bscore = None, (0,)
    for q in money_cols:
        for p in money_cols:
            for a in money_cols:
                if len({q, p, a}) < 3:
                    continue
                sat = 0
                for i in rows:
                    vq, vp, va = (_val(keep[k]["v"][i]["text"]) if i in keep[k]["v"] else None for k in (q, p, a))
                    if None not in (vq, vp, va) and abs(vq * vp - va) <= max(0.011, 0.006 * abs(va)):
                        sat += 1
                sc = (sat, q < p < a, a)
                if sat >= (n if n <= 2 else -(-n * 6 // 10)) and sc > bscore:
                    best, bscore = (q, p, a), sc
    if best:
        role[best[0]], role[best[1]], role[best[2]] = "qty", "unit_price", "line"
    elif money_cols:
        role[money_cols[-1]] = "line"
        if len(money_cols) >= 2:
            role[money_cols[-2]] = "unit_price"
        if len(money_cols) >= 3:
            role[money_cols[-3]] = "qty"
    last_money = money_cols[-1] if money_cols else None
    amount_k = last_money if last_money is not None else None
    # 5. names for the other columns come from the header lines just above the first row (when they were read)
    pitch = median(blk["gaps"]) if blk.get("gaps") else 3.0 * h
    hdr, j = [], rows[0] - 1
    while j >= max(0, rows[0] - 3) and (not hdr or lines[j + 1]["cy"] - lines[j]["cy"] <= max(1.9 * h, 0.7 * pitch)) and (hdr or lines[rows[0]]["cy"] - lines[j]["cy"] <= max(3.2 * h, 1.4 * pitch)):
        hdr += lines[j]["words"]; j -= 1
    heads: dict[int, list[dict]] = {k: [] for k in range(len(keep))}
    for w in hdr:
        cx = (w["x0"] + w["x1"]) / 2
        k, d = min(((k, abs(cx - (c["x0"] + c["x1"]) / 2)) for k, c in enumerate(keep)), key=lambda t: t[1])
        if d <= 0.07 * width:
            heads[k].append(w)
    keys: dict[int, str] = {}
    for k in range(len(keep)):
        txt = " ".join(w["text"] for w in sorted(heads[k], key=lambda w: (round(w["y0"] / h), w["x0"])))
        keys[k] = re.sub(r"[^a-z0-9]+", "_", re.sub(r"\(.*?\)", "", txt.lower())).strip("_") or f"col_{k + 1}"
    names: dict[int, str] = {}
    for k in range(len(keep)):
        r = role.get(k)
        names[k] = "qty" if r == "qty" else "unit_price" if r == "unit_price" else "amount" if k == amount_k else keys[k]
    used = set()
    for k in names:
        base = names[k]
        while names[k] in used:
            names[k] += "_2"
        used.add(names[k])
    # 6. build the items
    items = []
    first_col_x = keep[0]["x0"]
    for i in rows:
        head = info[i][0]
        cells = {}
        for k, c in enumerate(keep):
            cells[names[k]] = c["v"][i]["text"] if i in c["v"] else ""
        desc_ws = [w for w in head if w is not id_ws.get(i) and w is not qty_lead.get(i)] + [w for _, w in toks if _ == i and id(w) in dropped and w["x1"] <= first_col_x]
        extra = [lines[j] for j in blk.get("wraps", {}).get(i, [])]
        desc = " ".join(w["text"] for w in sorted(desc_ws, key=lambda w: w["x0"])) + "".join(" " + ln["text"] for ln in extra)
        it = {"id": id_ws[i]["text"] if i in id_ws else "", "description": desc.strip()}
        it["qty"] = cells.pop("qty", qty_lead[i]["text"] if i in qty_lead else "")
        it["unit_price"] = cells.pop("unit_price", "")
        it["amount"] = cells.pop("amount", "")
        it.update(cells)
        allw = list(head) + [w for c in keep for ww in [c["v"].get(i)] if ww for w in [ww]] + [w for ln in extra for w in ln["words"]]
        it.update({"bbox": _box(allw), "page": page, "_cy": lines[i]["cy"]})
        items.append(it)
    labels = [{"key": k, "label": k.replace("_", " ").title(), "cat": "other"} for k in [*names.values()]]
    for k, lab in (("description", "Description"), ("qty", "Qty"), ("unit_price", "Unit price"), ("amount", "Amount")):
        if not any(x["key"] == k for x in labels):
            labels.append({"key": k, "label": lab})
    return items, blk["last"], {"columns": labels, "footer": {}, "footer_bbox": [], "method": "data"}


def extract_line_items(words: list[dict], width: float, height: float, page: int = 1) -> tuple[list[dict], int, dict]:
    """Line items from two independent readers: the header-driven one and the data-driven one (see above). The result with the
    better internal consistency wins (every row has an amount, qty x price = amount, no amount stuck in a description)."""
    old = _extract_by_header(words, width, height, page)
    try:
        new = _extract_by_data(words, width, height, page)
    except Exception:                                   # a bad reading must never break the extraction
        new = ([], -1, {})
    return new if _score_items(new[0]) > _score_items(old[0]) else old


def _check_items(items: list[dict], subtotal: float | None, total: float | None = None) -> dict:
    """Repair amounts that OCR damaged, using qty x unit price, and compare the sum with the printed Subtotal.
    No repair when the table has tax / discount columns: there amount != qty x price is normal; the rows are then
    checked on the taxable-value column (against the subtotal) or on the amounts (against the grand total)."""
    if any(re.search(r"tax|gst|vat|tva|disc|remise", k) for it in items for k in it):
        for it in items:
            it["corrected"] = False
        sm = lambda k: [parse_amount(i[k]) if i.get(k) else None for i in items]
        amt = sm("amount"); s_amt = round(sum(v for v in amt if v is not None), 2)
        tv = sm("taxable_value") if any("taxable_value" in i for i in items) else None
        if tv is not None and subtotal is not None and None not in tv:
            ok = abs(sum(tv) - subtotal) <= 0.01
        elif amt and None not in amt and (subtotal is not None or total is not None):
            ok = any(v is not None and abs(s_amt - v) <= 0.01 for v in (subtotal, total))     # amounts before or after tax
        else:
            ok = None
        return {"items_sum": s_amt, "matches_subtotal": ok}
    def val(s): return parse_amount(s) if s else None
    calc = []
    for it in items:
        q, u = val(it["qty"]), val(it["unit_price"])
        calc.append(round(q * u, 2) if q is not None and u is not None else None)
    raw = [val(it["amount"]) for it in items]
    fixed = [False] * len(items)
    def total(vs): return round(sum(v for v in vs if v is not None), 2)
    use_calc = [c if c is not None else r for c, r in zip(calc, raw)]
    if subtotal is not None:
        if abs(total(raw) - subtotal) > 0.01 and abs(total(use_calc) - subtotal) <= 0.01:
            for i, (c, r) in enumerate(zip(calc, raw)):
                if c is not None and (r is None or abs(c - r) > 0.01):
                    items[i]["amount"], fixed[i] = _fmt(c), True
    else:
        for i, (c, r) in enumerate(zip(calc, raw)):
            if c is not None and (r is None or r == 0):
                items[i]["amount"], fixed[i] = _fmt(c), True
    for it, f in zip(items, fixed):
        it["corrected"] = f
    s = total([val(it["amount"]) for it in items])
    return {"items_sum": s, "matches_subtotal": None if subtotal is None else abs(s - subtotal) <= 0.01}


# --------------------------------------------------------------------------- totals and key: value pairs
def _key_of(text: str) -> str | None:
    t = text.lower()
    if re.search(r"sub[\s_-]?total\b", t): return "SUB_TOTAL"
    if re.search(r"\b(amount|balance)\s+due\b|\bdue\s+amount\b|\bbalance\b", t): return "AMOUNT_DUE"
    if re.search(r"\bdiscount\b", t): return "DISCOUNT"
    has_tax = re.search(r"\b(tax|vat|gst|tva|igst|cgst|sgst)\b", t)
    if re.search(r"\btotal\b", t) and not re.search(r"\b(qty|quantity|words)\b", t) and (not has_tax or re.search(r"incl|including", t)):
        return "TOTAL"
    if has_tax: return "TAX"
    return None


_TRIGGER = re.compile(r"^(sub[_-]?total|subtotal|total|tax|tax:vat|vat|gst|tva|igst|cgst|sgst|discount|balance|amount|grand|net|shipping)\b", re.I)
_GLUE = {"grand", "sub", "balance", "amount", "net", "total"}


def _segments(words: list[dict]) -> list[list[dict]]:
    """Split a row where several keys sit side by side ("SUB_TOTAL : 176.20   TAX:VAT (4.2%): 41.55")."""
    segs: list[list[dict]] = []
    for i, w in enumerate(words):
        prev = re.sub(r"[^a-z]", "", words[i - 1]["text"].lower()) if i else ""
        if _TRIGGER.match(w["text"]) and not (segs and prev in _GLUE and len(segs[-1]) == 1):
            segs.append([w])
        elif segs:
            segs[-1].append(w)
    return segs


def extract_totals(lines: list[dict], start: int = 0, page: int = 1) -> dict[str, dict]:
    found: dict[str, list[tuple[float, dict]]] = {}
    for ln in lines[start:]:
        for seg in _segments(ln["words"]):
            key = _key_of(" ".join(w["text"] for w in seg))
            if not key:
                continue
            toks = [w for w in seg if _money(w["text"])] or [w for w in seg if _num(w["text"])][-1:]
            if not toks:
                continue
            w = max(toks, key=lambda w: w["x1"])
            v = parse_amount(w["text"])
            if v is not None:
                found.setdefault(key, []).append((v, {"text": w["text"].strip("()"), "bbox": _box([w]), "page": page,
                                                      "cy": (w["y0"] + w["y1"]) / 2}))
    pick = {"SUB_TOTAL": lambda c: c[0], "TAX": lambda c: c[0], "DISCOUNT": lambda c: c[0],
            "AMOUNT_DUE": lambda c: c[-1], "TOTAL": lambda c: max(c, key=lambda x: x[0])}
    out = {k: pick[k](c)[1] | {"value": pick[k](c)[0]} for k, c in found.items()}
    if len(found.get("TAX", [])) > 1 and "SUB_TOTAL" in out and "TOTAL" in out:      # "Sales Tax 6%" + "Tax 2%" -> both count
        lo, hi = out["SUB_TOTAL"]["cy"] - 2, out["TOTAL"]["cy"] + 2
        grp = [x for x in found["TAX"] if lo <= x[1]["cy"] <= hi]
        if len(grp) > 1:
            tot = round(sum(v for v, _ in grp), 2)
            bb = [i["bbox"] for _, i in grp]
            out["TAX"] = {"text": _fmt(tot), "value": tot, "page": page, "cy": grp[0][1]["cy"],
                          "bbox": [min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb)]}
    return out


_KV = {
    "DUE_DATE": (re.compile(r"\bdue\s*date\b|\bpayment\s+due\b|\bdate\s+due\b"), "date"),
    "DATE": (re.compile(r"(?<!due )\b(?:invoice\s+|issue\s+|issued\s+|billing\s+)?date\b(?!\s+due)"), "date"),
    "NUMBER": (re.compile(r"\binv(?:oice)?\s*(?:#|no\.?|number|num\.?|id\b)"), "id"),
    "PO_NUMBER": (re.compile(r"\bp\.?o\.?\s*(?:#|no\.?|number)?(?![a-z])|\bpurchase\s+order\b"), "id"),
}


def extract_pairs(lines: list[dict], page: int = 1) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for idx, ln in enumerate(lines):
        text, pos, offs = ln["text"], 0, []
        for w in ln["words"]:
            offs.append(pos); pos += len(w["text"]) + 1
        low = text.lower()
        for label, (rx, kind) in _KV.items():
            if label in out:
                continue
            m = rx.search(low)
            if not m:
                continue
            def good(w):
                return bool(DATE_RE.match(w["text"])) if kind == "date" else (any(c.isdigit() for c in w["text"]) and len(w["text"]) >= 2 and not DATE_RE.match(w["text"]))
            same = [w for w, o in zip(ln["words"], offs) if o >= m.end() and re.search(r"\w", w["text"])]
            if same:                                                  # value right after the key, same row
                if good(same[0]):
                    out[label] = {"text": same[0]["text"].lstrip(":#"), "bbox": _box([same[0]]), "page": page}
                continue
            key_x1 = max(w["x1"] for w, o in zip(ln["words"], offs) if o < m.end())
            for nb in (lines[idx + 1] if idx + 1 < len(lines) else None, lines[idx - 1] if idx else None):   # OCR split the row in two
                if nb and abs(nb["cy"] - ln["cy"]) <= 1.6 * max(1.0, ln["y1"] - ln["y0"]):
                    cand = [w for w in nb["words"] if w["x0"] >= key_x1 - 2 and good(w)]
                    if cand:
                        out[label] = {"text": cand[0]["text"].lstrip(":#"), "bbox": _box([cand[0]]), "page": page}
                        break
    return out


# --------------------------------------------------------------------------- glue
def _has(label: str, fields: list[dict]) -> bool:
    texts = [f["text"] for f in fields if f["label"] == label]
    if label in ("SUB_TOTAL", "TAX", "TOTAL", "DISCOUNT", "AMOUNT_DUE"):
        return any(_money(t) or (_num(t) and "." in t) for tx in texts for t in tx.split())
    if label in ("DATE", "DUE_DATE"):
        return any(DATE_RE.match(t.strip(":")) for tx in texts for t in tx.split())
    return any(any(c.isdigit() for c in t) for tx in texts for t in tx.split())


def _consistent(t: dict) -> bool:
    if "SUB_TOTAL" not in t or "TOTAL" not in t:
        return False
    v = lambda k: t[k]["value"] if k in t else 0.0
    return abs(v("SUB_TOTAL") + v("TAX") - v("DISCOUNT") - t["TOTAL"]["value"]) <= 0.01


def _reconcile(readings: list[dict]) -> dict:
    """Merge the totals of several OCR readings. When their numbers disagree, keep the combination where
    subtotal + tax - discount = total (an OCR slip such as 14.76 -> 1476 cannot satisfy it)."""
    keys = sorted({k for r in readings for k in r})
    cands = {k: [] for k in keys}
    for r in readings:
        for k in keys:
            if k in r and all(abs(r[k]["value"] - c["value"]) > 1e-9 for c in cands[k]):
                cands[k].append(r[k])
    merged = {k: cands[k][0] for k in keys}
    if _consistent(merged) or not all(k in cands for k in ("SUB_TOTAL", "TOTAL")):
        return merged
    for combo in itertools.product(*(cands[k] for k in keys)):
        t = dict(zip(keys, combo))
        if _consistent(t):
            return t
    return merged


def _from_footer(meta: dict, page: int) -> dict[str, dict]:
    """SUB_TOTAL / TAX / TOTAL from the "Total" row printed inside the table (taxable value, tax amounts, total)."""
    f, bb = meta.get("footer") or {}, meta.get("footer_bbox") or [0, 0, 0, 0]
    cats = {c["key"]: c.get("cat") for c in meta.get("columns", [])}
    out: dict[str, dict] = {}
    def put(label, v):
        out[label] = {"text": _fmt(v), "value": round(v, 2), "bbox": bb, "page": page}
    val = lambda k: parse_amount(f[k]) if k in f else None
    if val("amount") is not None:
        put("TOTAL", val("amount"))
    if val("taxable_value") is not None:
        put("SUB_TOTAL", val("taxable_value"))
    tax = [val(k) for k in f if cats.get(k) in ("taxamt", "tax") and val(k) is not None]
    if tax:
        put("TAX", sum(tax))
    return out


def apply_rules(words: list[dict], fields: list[dict], width: float, height: float, page: int = 1,
                alt_words: list[dict] | None = None, alt_readings: list[list[dict]] | None = None) -> dict:
    """-> {"fields": merged fields, "items": [...], "items_check": {...}}.
    `alt_words` / `alt_readings` = other OCR readings of the same page (different modes / zoom): the OCR is fragile on
    tables, so every reading is tried and the one whose items add up to the printed subtotal wins."""
    readings = [words] + list(alt_readings or []) + ([alt_words] if alt_words else [])
    per = []
    for rw in readings:
        ls = _lines(rw)
        its, end, meta = extract_line_items(rw, width, height, page)
        per.append((its, extract_totals(ls, start=end + 1 if end >= 0 else 0, page=page), extract_pairs(ls, page), meta))
    totals = _reconcile([r[1] for r in per])
    pairs: dict[str, dict] = {}
    for r in per:
        pairs = {**r[2], **pairs}
    sub = totals.get("SUB_TOTAL", {}).get("value")
    items, meta = [], {}
    for r in per:                                    # the reading whose items add up to the printed subtotal wins
        cand = copy.deepcopy(r[0])
        if cand and (not items or (_check_items(cand, sub).get("matches_subtotal") and not _check_items(copy.deepcopy(items), sub).get("matches_subtotal"))):
            items, meta = cand, r[3]
    if "TAX" in totals and "SUB_TOTAL" in totals and "TOTAL" in totals and not _consistent(totals):
        gap = round(totals["TOTAL"]["value"] - totals["SUB_TOTAL"]["value"] + (totals["DISCOUNT"]["value"] if "DISCOUNT" in totals else 0), 2)
        if gap > 0:                                  # a tax line was read, but its digits are damaged: use total - subtotal
            totals["TAX"] = {**totals["TAX"], "text": _fmt(gap), "value": gap, "computed": True}
    foot = _from_footer(meta, page)
    for k, v in foot.items():                              # "Total" row inside the table (GST invoices)
        totals.setdefault(k, v)
    amounts = [parse_amount(i["amount"]) for i in items if i.get("amount")]
    if items and amounts and None not in amounts:
        s_amt = round(sum(amounts), 2)
        for c in [r[1]["TOTAL"] for r in per if "TOTAL" in r[1]] + ([foot["TOTAL"]] if "TOTAL" in foot else []):
            if abs(c["value"] - s_amt) <= 0.01:
                totals["TOTAL"] = c
                break
    sub = totals.get("SUB_TOTAL", {}).get("value")
    check = {**_check_items(items, sub, totals.get("TOTAL", {}).get("value")), **meta} if items else {}
    new = []
    for label, r in {**pairs, **totals}.items():
        if not _has(label, fields):
            new.append({"label": label, "text": r["text"], "confidence": 0.7, "bbox": r["bbox"], "page": page,
                        "source": "computed" if r.get("computed") else "rule"})
    replaced = {f["label"] for f in new}
    merged = [f for f in fields if f["label"] not in replaced] + new
    for it in items:
        it.pop("_cy", None)
    foot_amt = parse_amount(meta["footer"]["amount"]) if meta.get("footer", {}).get("amount") else None
    s_items = [parse_amount(i["amount"]) for i in items if i.get("amount")]
    foot_ok = foot_amt is not None and s_items and None not in s_items and abs(sum(s_items) - foot_amt) <= 0.01
    return {"fields": merged, "items": items, "items_check": check,
            "confident": bool(items) and (check.get("matches_subtotal") is True or bool(foot_ok)),
            "table_seen": any(r[3].get("columns") for r in per)}


def items_to_text(items: list[dict]) -> str:
    out = []
    for it in items:
        s = (f"{it['qty']} x " if it["qty"] else "") + it["description"]
        if it["unit_price"]: s += f" @ {it['unit_price']}"
        if it["amount"]: s += f" = {it['amount']}"
        out.append(s)
    return " ; ".join(out)
