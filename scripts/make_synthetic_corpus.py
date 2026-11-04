"""Builds a synthetic multi-template invoice CORPUS directly (no images, no OCR, takes seconds).

    python scripts/make_synthetic_corpus.py --templates 40 --per-template 25 --out data/processed/corpus_synth.jsonl

Why: it lets you test the training code (and the notebook) in a minute, and it behaves like FATURA in the one
way that matters for your problem: every template has its own layout, key wording and field positions, so a
split BY TEMPLATE measures real generalisation. The corpus has the same format as data/processed/corpus.jsonl.
It is a proxy, not a replacement for FATURA: never report its scores as your project result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

W, H = 1240.0, 1754.0

KEYS = {
    "NUMBER": ["Invoice No", "Invoice #", "Ref", "No.", "Facture No", "Document", "Invoice Number"],
    "DATE": ["Date", "Invoice Date", "Issued", "Dated", "Issue Date", "Created"],
    "DUE_DATE": ["Due Date", "Payment Due", "Due", "Pay By", "Due On"],
    "BILL_TO": ["Bill To", "Billed To", "Sold To", "Customer", "Client", "Invoice To"],
    "SUB_TOTAL": ["Subtotal", "Sub Total", "Net Amount", "Total before tax"],
    "TAX": ["Tax", "VAT", "Sales Tax", "GST"],
    "DISCOUNT": ["Discount", "Rebate", "Promo"],
    "TOTAL": ["Total", "Grand Total", "Amount Due", "Total Due", "Balance", "Net Total"],
}
COMPANIES = ["Northwind Traders", "Blue Harbor Ltd", "Atlas Supplies", "Orion Systems", "Sunrise Foods", "Kappa Tools",
             "Delta Logistics", "Maple Leaf Co", "Cedar Works", "Pioneer Print", "Lumen Labs", "Zenith Parts"]
PEOPLE = ["Alice Martin", "Omar Benali", "Sofia Rossi", "Daniel Kim", "Lina Haddad", "Peter Novak", "Yuki Tanaka"]
STREETS = ["Main Street", "Oak Avenue", "Rue Victor Hugo", "Elm Road", "Park Lane", "Market Square", "Hill Drive"]
CITIES = ["Springfield", "Marrakech", "Lyon", "Austin", "Leeds", "Porto", "Turin"]
ITEMS = ["Widget", "Service fee", "Consulting", "Cable set", "License", "Maintenance", "Shipping", "Printer toner"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
UNITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FREE = ["payment", "within", "days", "thank", "you", "for", "your", "business", "please", "remit", "bank", "account",
        "goods", "remain", "property", "until", "paid", "late", "fees", "apply", "contact", "us", "questions",
        "terms", "conditions", "transfer", "reference", "invoice", "number", "kindly", "settle", "balance"]
EXTRA_KEYS = {"SEND_TO": ["Ship To", "Deliver To", "Send To", "Shipping Address"],
              "TOTAL_WORDS": ["In words", "Amount in words", "Total in words", "Say"],
              "PAYMENT_DETAILS": ["Payment Details", "Bank Details", "Payment Info", "How to pay"],
              "NOTE": ["Notes", "Remarks", "Comments", "Note"],
              "CONDITIONS": ["Terms", "Conditions", "Terms & Conditions", "Terms of sale"]}
KEYS.update(EXTRA_KEYS)


def money(v: float, cur: str) -> str:
    return f"{cur}{v:,.2f}"


class Page:
    def __init__(self, fh: float):
        self.words: list[dict] = []
        self.fh = fh

    def put(self, x: float, y: float, text: str, label: str = "O", scale: float = 1.0, rnd=None) -> float:
        """Place whitespace-separated words on one row starting at (x, y); returns the x after the row."""
        h = self.fh * scale
        for tok in text.split():
            w = 0.55 * h * len(tok) + 2
            j = rnd.uniform(-1.5, 1.5) if rnd else 0.0
            self.words.append({"text": tok, "conf": 90.0 + (rnd.random() * 9 if rnd else 0),
                               "x0": x, "y0": y + j, "x1": x + w, "y1": y + h + j, "label": label})
            x += w + 0.45 * h
        return x


def make_template(t: int) -> dict:
    r = random.Random(1000 + t)
    pick = lambda k: r.choice(KEYS[k])
    return {
        "fh": r.choice([18, 20, 22, 24, 26]), "ml": r.randint(50, 130), "right_x": r.randint(600, 800),
        "title_pos": r.choice(["left", "center", "right"]), "seller_side": r.choice(["left", "right"]),
        "meta_side": r.choice(["left", "right"]), "buyer_side": r.choice(["left", "right"]),
        "key_mode": r.choice(["left", "above"]), "colon": r.random() < 0.6, "upper_keys": r.random() < 0.3,
        "keys": {k: pick(k) for k in KEYS}, "cur": r.choice(["$", "€", "£", ""]),
        "has_email": r.random() < 0.7, "has_discount": r.random() < 0.5, "has_due": r.random() < 0.75,
        "has_words": r.random() < 0.5, "has_payment": r.random() < 0.7, "has_note": r.random() < 0.5,
        "totals_side": r.choice(["left", "right"]), "table_cols": r.randint(3, 5), "gap": r.uniform(1.0, 1.8),
        "seller_first": r.random() < 0.6, "y_jitter": r.randint(0, 40), "date_fmt": r.choice(["slash", "dash", "text"]),
        "total_key_pos": r.choice(["same_row", "same_row", "above"]),
        "has_send": r.random() < 0.5, "has_cond": r.random() < 0.5, "hdr_prob": r.uniform(0.7, 1.0),
        "free_lines": r.randint(1, 3), "noise": r.uniform(0.0, 0.05),
    }


def make_doc(t: int, k: int, tpl: dict, rnd: random.Random) -> dict:
    pg = Page(tpl["fh"] * rnd.uniform(0.97, 1.03))
    fh, ml, rx, cur = pg.fh, tpl["ml"], tpl["right_x"], tpl["cur"]
    colx = lambda side: ml if side == "left" else rx
    line = fh * 1.6
    key = lambda name: (tpl["keys"][name].upper() if tpl["upper_keys"] else tpl["keys"][name]) + (":" if tpl["colon"] else "")

    def date():
        d, m, y = rnd.randint(1, 28), rnd.randint(1, 12), rnd.randint(2019, 2025)
        return {"slash": f"{d:02d}/{m:02d}/{y}", "dash": f"{y}-{m:02d}-{d:02d}", "text": f"{MONTHS[m - 1]} {d}, {y}"}[tpl["date_fmt"]]

    def kv(x, y, name, value, label):
        """key/value pair; returns the y after it."""
        if tpl["key_mode"] == "left":
            nx = pg.put(x, y, key(name), "O", rnd=rnd)
            pg.put(nx + 6, y, value, label, rnd=rnd)
            return y + line
        pg.put(x, y, key(name), "O", 0.85, rnd=rnd)
        pg.put(x, y + line * 0.95, value, label, rnd=rnd)
        return y + line * 2.0

    y0 = 60 + rnd.randint(0, tpl["y_jitter"])
    tx = {"left": ml, "center": 420, "right": rx + 100}[tpl["title_pos"]]
    pg.put(tx, y0, rnd.choice(["INVOICE", "INVOICE", "TAX INVOICE", "Facture"]), "TITLE", 1.8, rnd)
    if rnd.random() < 0.5:   # a logo = a company-like word, same look as a seller name
        pg.put(ml if tpl["seller_side"] == "right" else rx, y0, rnd.choice(COMPANIES).split()[0], "LOGO", 1.4, rnd)

    def party(x, y, lab_name, lab_addr):
        """name + 2 address lines: IDENTICAL look for seller, buyer and ship-to (only context tells them apart)."""
        pg.put(x, y, rnd.choice(COMPANIES + PEOPLE), lab_name, 1.0, rnd)
        pg.put(x, y + line, f"{rnd.randint(1, 250)} {rnd.choice(STREETS)}", lab_addr, 1.0, rnd)
        pg.put(x, y + line * 2, f"{rnd.choice(CITIES)} {rnd.randint(10000, 99999)}", lab_addr, 1.0, rnd)

    # seller block (no header: only the position tells it is the seller)
    sy = y0 + fh * 3.5 + (0 if tpl["seller_first"] else 330)
    sx = colx(tpl["seller_side"])
    party(sx, sy, "SELLER_NAME", "SELLER_ADDRESS")
    if tpl["has_email"]:
        pg.put(sx, sy + line * 3.1, f"info{rnd.randint(1, 99)}@{rnd.choice(['shop', 'corp', 'mail'])}.com", "SELLER_EMAIL", 1.0, rnd)

    # meta block (number / dates)
    my = y0 + fh * 3.5 + (330 if tpl["seller_first"] else 0)
    mx = colx(tpl["meta_side"])
    inv_no = f"{rnd.choice(['INV', 'FA', 'SO'])}-{rnd.randint(10000, 99999)}"
    my = kv(mx, my, "NUMBER", inv_no, "NUMBER")
    my = kv(mx, my, "DATE", date(), "DATE")
    if tpl["has_due"]:
        my = kv(mx, my, "DUE_DATE", date(), "DUE_DATE")

    # buyer (+ ship-to) blocks: header key = BILL_TO / SEND_TO, value block = BUYER / SEND_TO
    by = max(sy + line * 5, my) + 40
    bx = colx(tpl["buyer_side"])
    pg.put(bx, by, key("BILL_TO"), "BILL_TO", 1.0, rnd)
    party(bx, by + line, "BUYER", "BUYER")
    if tpl["has_send"]:
        sx2 = rx if bx == ml else ml
        pg.put(sx2, by, key("SEND_TO"), "SEND_TO", 1.0, rnd)
        party(sx2, by + line, "SEND_TO", "SEND_TO")

    # table (all background, but full of amounts: hard negatives)
    ty = by + line * 5.5
    heads = ["Description", "Qty", "Price", "Amount", "Tax"][: tpl["table_cols"]]
    xs = [ml + i * (W - 2 * ml) / tpl["table_cols"] for i in range(tpl["table_cols"])]
    for hx, hd in zip(xs, heads):
        pg.put(hx, ty, hd, "O", 0.95, rnd)
    total_val = 0.0
    for r in range(rnd.randint(3, 8)):
        ry = ty + line * (r + 1) * tpl["gap"]
        qty, price = rnd.randint(1, 9), rnd.uniform(5, 600)
        total_val += qty * price
        cells = [rnd.choice(ITEMS), str(qty), money(price, cur), money(qty * price, cur), f"{rnd.choice([5, 10, 20])}%"][: tpl["table_cols"]]
        for cx, c in zip(xs, cells):
            pg.put(cx, ry, c, "O", rnd=rnd)
    ey = ty + line * (8 + 1) * tpl["gap"] + 20

    # totals block
    tx0 = colx(tpl["totals_side"])
    disc = total_val * rnd.choice([0.05, 0.1]) if tpl["has_discount"] else 0.0
    tax = (total_val - disc) * 0.2
    rows = [("SUB_TOTAL", total_val)] + ([("DISCOUNT", disc)] if tpl["has_discount"] else []) + [("TAX", tax), ("TOTAL", total_val - disc + tax)]
    for name, v in rows:
        if tpl["total_key_pos"] == "same_row" or name != "TOTAL":
            nx = pg.put(tx0, ey, key(name), "O", 1.0, rnd)
            pg.put(nx + 8, ey, money(v, cur), name, 1.0, rnd)
            ey += line
        else:
            pg.put(tx0, ey, key(name), "O", 1.0, rnd)
            pg.put(tx0, ey + line * 0.95, money(v, cur), name, 1.3, rnd)
            ey += line * 2.2

    def free_block(name, flag, n_lines):
        """free text lines drawn from ONE shared vocabulary; a header key tells which field it is."""
        nonlocal ey
        if not flag:
            return
        ey += 30
        if rnd.random() < tpl["hdr_prob"]:
            pg.put(ml, ey, key(name), "O", 1.0, rnd)
            ey += line
        for _ in range(n_lines):
            pg.put(ml, ey, " ".join(rnd.choice(FREE) for _ in range(rnd.randint(3, 7))), name, 1.0, rnd)
            ey += line

    free_block("TOTAL_WORDS", tpl["has_words"], 1)
    free_block("PAYMENT_DETAILS", tpl["has_payment"], tpl["free_lines"])
    free_block("CONDITIONS", tpl["has_cond"], tpl["free_lines"])
    free_block("NOTE", tpl["has_note"], 1)
    if rnd.random() < 0.5:   # stray unlabeled text with a date (hard negative)
        pg.put(ml, H - 120, f"Printed on {date()} page 1 of 1", "O", 0.8, rnd)

    # OCR-like noise on the text
    for w in pg.words:
        if rnd.random() < tpl["noise"]:
            txt = w["text"]
            i = rnd.randrange(len(txt))
            w["text"] = txt[:i] + rnd.choice("0Oo1lI5S,.") + txt[i + 1:]
    words = sorted(pg.words, key=lambda w: (round(w["y0"] / 8), w["x0"]))
    return {"stem": f"Template{t}_Instance{k}", "template": f"template{t}", "width": W, "height": H, "words": words,
            "n_annotations": len(words), "align": "synthetic", "align_source": None, "align_fit": 1.0, "ann_errors": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=int, default=40)
    ap.add_argument("--per-template", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/processed/corpus_synth.jsonl")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    docs = []
    for t in range(1, a.templates + 1):
        tpl = make_template(t)
        docs += [make_doc(t, k, tpl, rnd) for k in range(a.per_template)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    meta = {"source": "synthetic (make_synthetic_corpus.py)", "created_at": "synthetic", "n_documents": len(docs),
            "ocr_lang": "eng", "ocr_psm": 6}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    n = sum(len(d["words"]) for d in docs)
    print(f"{len(docs)} docs, {n} words -> {out}")


if __name__ == "__main__":
    main()
