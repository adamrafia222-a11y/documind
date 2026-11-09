#!/usr/bin/env python3
"""
Synthetic receipt (ticket de caisse) dataset generator: images + labels.

For every receipt it writes:
  - a JPEG photo/scan (random store, items, thermal-paper look, torn edges,
    shadow, rotation, blur, noise, JPEG compression)
  - YOLO labels (.txt)         -> images/<split>/, labels/<split>/
  - COCO annotations (.json)   -> annotations/<split>.json
  - ground truth (.jsonl)      -> ground_truth.jsonl (store, date, items, totals...)
  - data.yaml for Ultralytics YOLO

Classes (13): store_name, address, phone, date, time, receipt_number,
              item_name, item_qty, item_price, subtotal, tax, total, payment_method

Requirements:  pip install pillow numpy
Usage:
  python generate_receipts.py --n 10000 --out receipt_dataset
  python generate_receipts.py --n 10000 --currency EUR
  python generate_receipts.py --n 50 --preview 10      # quick test with box previews
"""
import argparse
import json
import math
import os
import random
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CLASSES = [
    "store_name", "address", "phone", "date", "time", "receipt_number",
    "item_name", "item_qty", "item_price", "subtotal", "tax", "total", "payment_method",
]

# ----------------------------------------------------------------------------
# Random content
# ----------------------------------------------------------------------------
STORES = ["Supermarché Atlas", "Marché Central", "Épicerie du Coin", "Hyper Horizon",
          "Super Médina", "Mini Market Al Baraka", "Bazar Océan", "Libre Service Sahara",
          "Comptoir du Nord", "Superette Riad", "Drugstore Delta", "Boulangerie Soleil"]
STREETS = ["Mohammed V", "Hassan II", "de la Liberté", "des Écoles", "Al Massira",
           "de la République", "Victor Hugo", "du Commerce", "des Fleurs", "Ibn Sina"]
CITIES = ["Marrakech", "Casablanca", "Rabat", "Fès", "Agadir", "Tanger", "Meknès",
          "Lyon", "Paris", "Lille", "Toulouse", "Nantes"]
PRODUCTS = [
    "Lait demi-écrémé 1L", "Pain complet", "Eau minérale 1,5L", "Yaourt nature x4",
    "Fromage râpé 200g", "Œufs x12", "Riz basmati 1kg", "Pâtes spaghetti 500g",
    "Huile d'olive 1L", "Sucre en poudre 1kg", "Café moulu 250g", "Thé vert 25 sachets",
    "Tomates 1kg", "Pommes de terre 2kg", "Poulet entier", "Jus d'orange 1L",
    "Biscuits chocolat", "Savon liquide", "Dentifrice", "Papier toilette x6",
    "Lessive 2L", "Bananes 1kg", "Oignons 1kg", "Beurre 250g", "Confiture fraise",
    "Farine 1kg", "Sel fin 1kg", "Thon en boîte", "Eau gazeuse 1L", "Chocolat noir 100g",
    "Baguette", "Croissant", "Concombre", "Carottes 1kg", "Citrons 500g", "Miel 500g",
]
PAYMENTS = ["Espèces", "Carte bancaire", "CB sans contact", "Chèque", "Espèces", "Carte bancaire"]
FOOTERS = ["Merci de votre visite", "À bientôt !", "Merci et à bientôt", "Conservez votre ticket"]


def money(v):
    return f"{v:.2f}".replace(".", ",")


def digits(rng, k):
    return "".join(rng.choice("0123456789") for _ in range(k))


# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


PRINT = _first(["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                "C:/Windows/Fonts/arial.ttf"])
BOLD = _first(["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
               "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]) or PRINT
MONO = _first(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
               "/System/Library/Fonts/Supplemental/Courier New.ttf",
               "C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/consola.ttf"]) or PRINT
MONOB = _first(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
                "C:/Windows/Fonts/courbd.ttf", "C:/Windows/Fonts/consolab.ttf"]) or MONO


@lru_cache(maxsize=256)
def get_font(path, size):
    size = int(size)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Receipt drawing
# ----------------------------------------------------------------------------
class Pen:
    """Small helper: draws text with alignment and returns tight boxes."""

    def __init__(self, d, ink):
        self.d, self.ink = d, ink

    def put(self, text, x, y, font, anchor="l", fill=None):
        w = font.getlength(text)
        x0 = x if anchor == "l" else x - w if anchor == "r" else x - w / 2
        self.d.text((x0, y), text, font=font, fill=fill or self.ink)
        b = self.d.textbbox((x0, y), text, font=font)
        return [b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2]

    def pair(self, label, value, x, y, font, anchor="l"):
        """Draw label+value as one string, return the box of the value only."""
        total = font.getlength(label + value)
        x0 = x if anchor == "l" else x - total if anchor == "r" else x - total / 2
        self.d.text((x0, y), label, font=font, fill=self.ink)
        vx = x0 + font.getlength(label)
        self.d.text((vx, y), value, font=font, fill=self.ink)
        b = self.d.textbbox((vx, y), value, font=font)
        return [b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2]


def make_receipt(rng, cfg):
    cur = cfg["currency"]
    suffix = " DH" if cur == "MAD" else " €"
    rw = rng.randint(380, 480)
    m = 22
    mono = rng.random() < 0.8
    fp, fpb = (MONO, MONOB) if mono else (PRINT, BOLD)
    fs = rng.randint(19, 24)
    f, fb = get_font(fp, fs), get_font(fpb, fs)
    fbig = get_font(fpb, fs + rng.randint(8, 14))
    fsm = get_font(fp, max(14, fs - 4))
    lh = int(fs * 1.5)

    paper = tuple(min(255, rng.randint(228, 252) + d) for d in (0, 0, rng.randint(-6, 3)))
    g = rng.randint(15, 80)
    ink = (g, g, min(255, g + rng.randint(0, 25)))
    canvas = Image.new("RGB", (rw, 3200), paper)
    d = ImageDraw.Draw(canvas)
    pen = Pen(d, ink)
    boxes = []

    sep_char = rng.choice(["-", "=", "*", "- "])

    def sep(y):
        ch = sep_char
        n = int((rw - 2 * m) / f.getlength(ch))
        d.text((m, y), ch * n, font=f, fill=ink)

    y = rng.randint(14, 30)
    truth = {}

    # ---- header
    store = rng.choice(STORES)
    truth["store_name"] = store
    boxes.append(("store_name", pen.put(store.upper() if rng.random() < 0.5 else store,
                                         rw / 2, y, fbig, "c")))
    y += int(fbig.size * 1.4)
    street = f"{rng.randint(1, 250)} {rng.choice(['Av.', 'Rue', 'Bd'])} {rng.choice(STREETS)}"
    city = f"{digits(rng, 5)} {rng.choice(CITIES)}"
    for line in (street, city):
        boxes.append(("address", pen.put(line, rw / 2, y, f, "c")))
        y += lh
    truth["address"] = f"{street}, {city}"
    phone = "0" + rng.choice("5678") + " " + " ".join(digits(rng, 2) for _ in range(4))
    boxes.append(("phone", pen.pair(rng.choice(["Tél: ", "Tel : ", "Tél. "]), phone, rw / 2, y, f, "c")))
    truth["phone"] = phone
    y += lh
    if rng.random() < 0.5:  # unannotated fiscal line
        d.text((m, y), "ICE: " + digits(rng, 15), font=fsm, fill=ink)
        y += int(lh * 0.9)
    y += 4
    sep(y)
    y += lh

    # ---- date / time / ticket number
    dt = rng.choice(["%02d/%02d/%d", "%02d-%02d-%d", "%02d.%02d.%d"]) % (
        rng.randint(1, 28), rng.randint(1, 12), rng.randint(2022, 2026))
    tm = "%02d:%02d" % (rng.randint(7, 22), rng.randint(0, 59))
    if rng.random() < 0.3:
        tm += ":%02d" % rng.randint(0, 59)
    no = digits(rng, rng.randint(5, 8))
    boxes.append(("date", pen.pair(rng.choice(["Date: ", "Le ", "Date "]), dt, m, y, f)))
    boxes.append(("time", pen.pair(rng.choice(["Heure: ", "à ", ""]), tm, rw - m, y, f, "r")))
    y += lh
    boxes.append(("receipt_number", pen.pair(rng.choice(["Ticket N° ", "N° ", "Ticket: ", "Facture "]), no, m, y, f)))
    if rng.random() < 0.6:
        d.text((rw - m - f.getlength("Caisse 01"), y), "Caisse %02d" % rng.randint(1, 9), font=f, fill=ink)
    y += lh
    truth.update({"date": dt, "time": tm, "receipt_number": no})
    sep(y)
    y += lh

    # ---- items
    qty_style = rng.choice(["col", "prefix"])
    lo, hi = (3, 90) if cur == "MAD" else (0.6, 20)
    items, total = [], 0.0
    x_price = rw - m
    x_qty = rw * 0.76
    if qty_style == "col" and rng.random() < 0.6:
        d.text((m, y), "Article", font=fsm, fill=ink)
        d.text((x_qty - fsm.getlength("Qté"), y), "Qté", font=fsm, fill=ink)
        d.text((x_price - fsm.getlength("Prix"), y), "Prix", font=fsm, fill=ink)
        y += int(lh * 0.95)
    for name in rng.sample(PRODUCTS, rng.randint(2, 11)):
        qty = rng.choice([1, 1, 1, 2, 2, 3, 4])
        unit = round(rng.uniform(lo, hi) * 20) / 20
        line_total = round(qty * unit, 2)
        total += line_total
        shown = name.upper() if rng.random() < 0.4 else name
        if qty_style == "col":
            max_w = x_qty - m - 36
            x_name = m
        else:
            qtxt = f"{qty} x "
            boxes.append(("item_qty", pen.put(str(qty), m, y, f)))
            d.text((m + f.getlength(str(qty)), y), " x", font=f, fill=ink)
            x_name = m + f.getlength(qtxt)
            max_w = x_price - x_name - 90
        while f.getlength(shown) > max_w and " " in shown:   # drop trailing words first
            shown = shown.rsplit(" ", 1)[0]
        while f.getlength(shown) > max_w and len(shown) > 4:
            shown = shown[:-1]
        boxes.append(("item_name", pen.put(shown, x_name, y, f)))
        if qty_style == "col":
            boxes.append(("item_qty", pen.put(str(qty), x_qty, y, f, "r")))
        boxes.append(("item_price", pen.put(money(line_total), x_price, y, f, "r")))
        items.append({"name": shown, "qty": qty, "unit_price": money(unit), "price": money(line_total)})
        y += lh
    truth["items"] = items
    sep(y)
    y += lh

    # ---- totals (prices are tax-inclusive, tax is included in the total)
    total = round(total, 2)
    rate = 0.20 if cur == "MAD" else rng.choice([0.055, 0.10, 0.20])
    tax = round(total * rate / (1 + rate), 2)
    sub = round(total - tax, 2)
    rate_txt = f"{rate * 100:g}".replace(".", ",")
    d.text((m, y), "Total HT", font=f, fill=ink)
    boxes.append(("subtotal", pen.put(money(sub), x_price, y, f, "r")))
    y += lh
    d.text((m, y), f"TVA {rate_txt}%", font=f, fill=ink)
    boxes.append(("tax", pen.put(money(tax), x_price, y, f, "r")))
    y += lh
    total_txt = money(total) + (suffix if rng.random() < 0.6 else "")
    d.text((m, y), rng.choice(["TOTAL TTC", "TOTAL", "NET À PAYER"]), font=fb, fill=ink)
    boxes.append(("total", pen.put(total_txt, x_price, y, fb, "r")))
    y += int(lh * 1.3)
    truth.update({"subtotal": money(sub), "tax_rate": rate_txt, "tax": money(tax),
                  "total": total_txt, "total_value": total, "currency": cur})

    # ---- payment
    pay = rng.choice(PAYMENTS)
    d.text((m, y), "Paiement:", font=f, fill=ink)
    boxes.append(("payment_method", pen.put(pay, x_price, y, f, "r")))
    truth["payment_method"] = pay
    y += lh
    if pay.startswith("Esp") and rng.random() < 0.7:
        given = math.ceil(total / 10) * 10 + rng.choice([0, 0, 10, 50])
        d.text((m, y), "Rendu", font=f, fill=ink)
        d.text((x_price - f.getlength(money(given - total)), y), money(given - total), font=f, fill=ink)
        y += lh

    # ---- footer + barcode
    y += 6
    foot = rng.choice(FOOTERS)
    d.text((rw / 2 - f.getlength(foot) / 2, y), foot, font=f, fill=ink)
    y += int(lh * 1.4)
    if rng.random() < 0.7:
        x = m + rng.randint(0, 30)
        top, bot = y, y + rng.randint(45, 70)
        while x < rw - m - 30:
            bw = rng.choice([1, 1, 2, 3])
            d.rectangle([x, top, x + bw - 1, bot], fill=ink)
            x += bw + rng.choice([1, 2, 2, 3])
        y = bot + 8
        d.text((rw / 2 - fsm.getlength(digits(rng, 12)) / 2, y), digits(rng, 12), font=fsm, fill=ink)
        y += int(lh * 0.9)
    H = y + rng.randint(14, 30)
    canvas = canvas.crop((0, 0, rw, H))

    # faded thermal bands
    if rng.random() < 0.4:
        ov = ImageDraw.Draw(canvas, "RGBA")
        for _ in range(rng.randint(1, 4)):
            yy = rng.randint(0, H)
            ov.rectangle([0, yy, rw, yy + rng.randint(4, 40)], fill=paper + (rng.randint(40, 120),))

    # torn edges -> alpha mask
    mask = Image.new("L", (rw, H), 255)
    md = ImageDraw.Draw(mask)
    if rng.random() < 0.7:
        step = rng.choice([6, 8, 10])
        for x in range(0, rw, step):
            md.polygon([(x, 0), (x + step / 2, rng.randint(3, 6)), (x + step, 0)], fill=0)
            md.polygon([(x, H - 1), (x + step / 2, H - 1 - rng.randint(3, 6)), (x + step, H - 1)], fill=0)
    rec = canvas.convert("RGBA")
    rec.putalpha(mask)
    return rec, boxes, truth


# ----------------------------------------------------------------------------
# Compose on a background + augmentation (boxes follow every transform)
# ----------------------------------------------------------------------------
def compose(rec, boxes, rng, nrng):
    w, h = rec.size
    angle = rng.uniform(-4, 4)
    rot = rec.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    nw, nh = rot.size
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    mm = rng.randint(30, 110)
    ox, oy = rng.randint(-10, 10), rng.randint(-10, 10)
    W, H = nw + 2 * mm, nh + 2 * mm
    new_boxes = []
    for cls, (x0, y0, x1, y1) in boxes:
        pts = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            dx, dy = x - w / 2, y - h / 2
            pts.append((nw / 2 + dx * c + dy * s + mm + ox, nh / 2 - dx * s + dy * c + mm + oy))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        new_boxes.append((cls, [min(xs), min(ys), max(xs), max(ys)]))

    kind = rng.random()
    if kind < 0.35:
        base = tuple([rng.randint(215, 255)] * 3)       # scanner / white desk
    elif kind < 0.7:
        base = tuple(rng.randint(90, 200) for _ in range(3))   # table
    else:
        base = tuple(rng.randint(15, 70) for _ in range(3))    # dark surface
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = base
    bg += nrng.normal(0, rng.uniform(2, 10), (H, W, 1))
    bg_img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)).convert("RGBA")

    shadow = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    shadow.putalpha(rot.getchannel("A").point(lambda a: int(a * 0.4)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    bg_img.alpha_composite(shadow, (mm + ox + 6, mm + oy + 9))
    bg_img.alpha_composite(rot, (mm + ox, mm + oy))
    return bg_img.convert("RGB"), new_boxes


def augment(img, boxes, rng, nrng):
    f = rng.uniform(0.6, 1.0)
    if f < 0.97:
        img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
        boxes = [(k, [v * f for v in b]) for k, b in boxes]
    if rng.random() < 0.25:
        img = img.convert("L").convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.12))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    r = rng.uniform(0, 1.2)
    if r > 0.25:
        img = img.filter(ImageFilter.GaussianBlur(r))
    arr = np.asarray(img).astype(np.float32)
    arr += nrng.normal(0, rng.uniform(0, 8), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    W, H = img.size
    out = [(k, [max(0, b[0]), max(0, b[1]), min(W - 1, b[2]), min(H - 1, b[3])]) for k, b in boxes]
    return img, out


# ----------------------------------------------------------------------------
# Worker + main
# ----------------------------------------------------------------------------
def worker(job):
    idx, split, cfg = job
    seed = cfg["seed"] * 1_000_003 + idx
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    rec, boxes, truth = make_receipt(rng, cfg)
    img, boxes = compose(rec, boxes, rng, nrng)
    img, boxes = augment(img, boxes, rng, nrng)
    name = f"receipt_{idx:06d}"
    root = cfg["out"]
    img.save(os.path.join(root, "images", split, name + ".jpg"), quality=rng.randint(45, 92))
    W, H = img.size
    lines = ["%d %.6f %.6f %.6f %.6f" % (
        CLASSES.index(c), (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H)
        for c, b in boxes]
    with open(os.path.join(root, "labels", split, name + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"idx": idx, "split": split, "file": f"images/{split}/{name}.jpg",
            "w": W, "h": H, "boxes": boxes, "truth": truth}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="number of receipts")
    ap.add_argument("--out", default="receipt_dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--currency", choices=["MAD", "EUR"], default="MAD")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--preview", type=int, default=0, help="save N previews with boxes drawn")
    a = ap.parse_args()
    cfg = {"out": a.out, "seed": a.seed, "currency": a.currency}

    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(a.out, sub, sp), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    if a.preview:
        os.makedirs(os.path.join(a.out, "preview"), exist_ok=True)

    n_train, n_val = int(a.n * 0.8), int(a.n * 0.1)
    jobs = [(i, "train" if i < n_train else "val" if i < n_train + n_val else "test", cfg)
            for i in range(a.n)]
    records = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(worker, jobs, chunksize=16), 1):
            records.append(r)
            if k % 500 == 0 or k == a.n:
                print(f"{k}/{a.n}")

    coco = {sp: {"images": [], "annotations": [],
                 "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
            for sp in ("train", "val", "test")}
    ann_id = 1
    with open(os.path.join(a.out, "ground_truth.jsonl"), "w", encoding="utf-8") as gt:
        for r in records:
            c = coco[r["split"]]
            c["images"].append({"id": r["idx"], "file_name": os.path.basename(r["file"]),
                                "width": r["w"], "height": r["h"]})
            for cls, (x0, y0, x1, y1) in r["boxes"]:
                c["annotations"].append({
                    "id": ann_id, "image_id": r["idx"], "category_id": CLASSES.index(cls) + 1,
                    "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                    "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0})
                ann_id += 1
            gt.write(json.dumps({"file_name": r["file"], "split": r["split"],
                                 "ground_truth": r["truth"]}, ensure_ascii=False) + "\n")
    for sp, data in coco.items():
        with open(os.path.join(a.out, "annotations", f"{sp}.json"), "w") as fh:
            json.dump(data, fh)

    with open(os.path.join(a.out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        fh.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

    for r in records[:a.preview]:
        im = Image.open(os.path.join(a.out, r["file"])).convert("RGB")
        dr = ImageDraw.Draw(im)
        for cls, (x0, y0, x1, y1) in r["boxes"]:
            dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        im.save(os.path.join(a.out, "preview", os.path.basename(r["file"])))

    print(f"Done: {a.n} receipts in '{a.out}' ({n_train} train / {n_val} val / {a.n - n_train - n_val} test)")


if __name__ == "__main__":
    main()
