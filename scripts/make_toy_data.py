"""Generates a tiny FATURA-like dataset (6 templates) to test the whole pipeline in minutes.
It is NOT for real training - use the real FATURA dataset for that.

    python scripts/make_toy_data.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]
COMPANIES = ["Northwind Traders", "Blue Harbor Ltd", "Atlas Supplies", "Orion Systems", "Sunrise Foods", "Kappa Tools"]
PEOPLE = ["Alice Martin", "Omar Benali", "Sofia Rossi", "Daniel Kim", "Lina Haddad", "Peter Novak"]
CATEGORIES = ["TOTAL", "DATE", "INVOICE_NUMBER", "VENDOR", "BUYER"]

# template -> key texts and positions (x, y) of vendor / invoice no / date / buyer / total
TEMPLATES = [
    dict(inv="Invoice No:", date="Date:", buyer="Bill To:", total="Total:", pos=[(80, 80), (80, 300), (700, 300), (80, 450), (700, 1200)]),
    dict(inv="Invoice #", date="Issued:", buyer="Customer:", total="Amount Due:", pos=[(700, 80), (80, 220), (80, 300), (80, 520), (80, 1300)]),
    dict(inv="Ref:", date="Invoice Date:", buyer="Sold To:", total="Grand Total:", pos=[(80, 150), (700, 150), (700, 230), (80, 400), (600, 1400)]),
    dict(inv="No.", date="Dated", buyer="Client", total="TOTAL DUE", pos=[(400, 60), (80, 260), (400, 260), (80, 600), (700, 1100)]),
    dict(inv="Facture No:", date="Date:", buyer="Billed To:", total="Net Total:", pos=[(80, 60), (500, 200), (500, 280), (80, 700), (80, 1500)]),
    dict(inv="Document:", date="Created:", buyer="Attn:", total="Balance:", pos=[(600, 700), (80, 100), (80, 180), (80, 300), (600, 1250)]),
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def encode_box(box, convention: str, size) -> list:
    """Write a pixel box (x0, y0, x1, y1) the way different datasets do."""
    x0, y0, x1, y1 = box
    W, H = size
    if convention == "xywh":
        return [x0, y0, x1 - x0, y1 - y0]
    if convention == "xyxy":
        return [x0, y0, x1, y1]
    if convention == "xyxy_n1000":
        return [round(x0 * 1000 / W), round(y0 * 1000 / H), round(x1 * 1000 / W), round(y1 * 1000 / H)]
    return [round(x0 / W, 5), round(y0 / H, 5), round(x1 / W, 5), round(y1 / H, 5)]       # xyxy_n1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "toy"))
    ap.add_argument("--per-template", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--convention", default="xywh", choices=["xywh", "xyxy", "xyxy_n1000", "xyxy_n1"],
                    help="how the COCO-style boxes are written (real datasets differ): tests the auto-detection")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "Annotations" / "COCO_compatible_format").mkdir(parents=True, exist_ok=True)
    f_big, f = font(44), font(30)

    for t_i, t in enumerate(TEMPLATES, start=1):
        for k in range(args.per_template):
            img = Image.new("RGB", (1240, 1754), (255, 255, 255))
            d = ImageDraw.Draw(img)
            vendor = rnd.choice(COMPANIES)
            inv = f"{rnd.choice(['INV', 'FA', 'SO'])}-{rnd.randint(10000, 99999)}"
            date = f"{rnd.randint(1, 28):02d}/{rnd.randint(1, 12):02d}/{rnd.randint(2019, 2025)}"
            buyer = rnd.choice(PEOPLE)
            total = f"${rnd.randint(50, 9000)}.{rnd.randint(0, 99):02d}"
            values = [("VENDOR", vendor), ("INVOICE_NUMBER", inv), ("DATE", date), ("BUYER", buyer), ("TOTAL", total)]
            keys = [None, t["inv"], t["date"], t["buyer"], t["total"]]
            anns = []
            for (cat, val), key, (x, y) in zip(values, keys, t["pos"]):
                font_ = f_big if cat == "VENDOR" else f
                if key:
                    d.text((x, y), key, fill=(30, 30, 30), font=font_)
                    x += int(d.textlength(key + " ", font=font_))
                d.text((x, y), val, fill=(0, 0, 0), font=font_)
                x0, y0, x1, y1 = d.textbbox((x, y), val, font=font_)
                anns.append({"bbox": encode_box((x0, y0, x1, y1), args.convention, img.size),
                             "category_id": CATEGORIES.index(cat) + 1})
            for r in range(5):   # a few unlabeled table lines
                d.text((80, 800 + r * 60), f"Item {r + 1}   {rnd.randint(1, 9)} x ${rnd.randint(5, 300)}.00", fill=(40, 40, 40), font=f)
            stem = f"Template{t_i}_Instance{k}"
            img.save(out / "images" / f"{stem}.jpg", quality=92)
            coco = {"images": [{"id": 1, "file_name": f"{stem}.jpg", "width": 1240, "height": 1754}],
                    "annotations": [{"id": i + 1, "image_id": 1, **a} for i, a in enumerate(anns)],
                    "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CATEGORIES)]}
            (out / "Annotations" / "COCO_compatible_format" / f"{stem}.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"Toy dataset written to {out} ({len(TEMPLATES) * args.per_template} images)")


if __name__ == "__main__":
    main()
