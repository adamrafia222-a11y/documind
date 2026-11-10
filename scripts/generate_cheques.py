#!/usr/bin/env python3
"""
Synthetic bank-cheque dataset generator (FATURA-style): images + labels.

For every cheque it writes:
  - a JPEG image (random layout, background, handwriting, signature, scan noise)
  - YOLO labels (.txt)            -> images/<split>/, labels/<split>/
  - COCO annotations (.json)      -> annotations/<split>.json
  - ground truth text (.jsonl)    -> ground_truth.jsonl  (for OCR / Donut / LayoutLM)
  - data.yaml for Ultralytics YOLO

Classes (10): bank_name, cheque_number, amount_words, payee, amount_digits,
              place, date, signature, account_number, micr_line

Requirements:  pip install pillow numpy
Usage:
  python generate_cheques.py --n 10000 --out cheque_dataset
  python generate_cheques.py --n 10000 --currency EUR --fonts-dir ./handwriting_fonts
  python generate_cheques.py --n 50 --preview 10      # quick test with box previews

Tip: put free handwriting fonts (.ttf) in --fonts-dir for much more realistic
handwriting (e.g. Google Fonts: Caveat, Homemade Apple, Reenie Beanie,
Nanum Pen Script, Shadows Into Light). Without them a slanted sans font with
per-character jitter is used.
"""
import argparse
import glob
import json
import math
import os
import random
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CLASSES = [
    "bank_name", "cheque_number", "amount_words", "payee", "amount_digits",
    "place", "date", "signature", "account_number", "micr_line",
]

# ----------------------------------------------------------------------------
# Random content
# ----------------------------------------------------------------------------
BANKS = ["Banque Atlas", "Crédit du Souss", "Banque Horizon", "Banque Al Amal",
         "Caisse Centrale d'Épargne", "Banque Medina", "Crédit Régional du Nord",
         "Banque Nationale du Sud", "Union Bancaire Maghreb", "Banque Océane"]
FIRST = ["Mohamed", "Fatima", "Youssef", "Khadija", "Ahmed", "Amina", "Omar", "Salma",
         "Hassan", "Zineb", "Karim", "Imane", "Rachid", "Nadia", "Pierre", "Marie",
         "Jean", "Sophie", "Lucas", "Camille", "Thomas", "Julie", "Nicolas", "Laura"]
LAST = ["Alaoui", "Bennani", "El Idrissi", "Tazi", "Berrada", "Lahlou", "Chraibi",
        "Benjelloun", "Fassi", "Amrani", "Martin", "Bernard", "Dubois", "Moreau",
        "Laurent", "Simon", "Michel", "Lefebvre", "Garcia", "Roux"]
COMPANY = ["Atlas Distribution", "Sahara Équipements", "Maroc Services", "Nord Matériaux",
           "Horizon Informatique", "Médina Textile", "Océan Logistique", "Al Baraka Négoce",
           "Delta Fournitures", "Riad Immobilier"]
CITIES_MA = ["Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir", "Meknès",
             "Oujda", "Kénitra", "Tétouan"]
CITIES_FR = ["Paris", "Lyon", "Marseille", "Lille", "Toulouse", "Bordeaux", "Nantes", "Nice"]

UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
         "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept",
         "dix-huit", "dix-neuf"]
TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}


def _lt100(n, final):
    if n < 20:
        return UNITS[n]
    if n < 70:
        t, u = n - n % 10, n % 10
        if u == 0:
            return TENS[t]
        return TENS[t] + (" et un" if u == 1 else "-" + UNITS[u])
    if n < 80:
        return "soixante et onze" if n == 71 else "soixante-" + UNITS[n - 60]
    if n == 80:
        return "quatre-vingts" if final else "quatre-vingt"
    return "quatre-vingt-" + UNITS[n - 80]


def _lt1000(n, final):
    h, r = divmod(n, 100)
    if h == 0:
        return _lt100(r, final)
    head = "cent" if h == 1 else UNITS[h] + " cent"
    if r == 0:
        return head + ("s" if h > 1 and final else "")
    return head + " " + _lt100(r, final)


def fr_words(n):
    """French number to words, 0 <= n <= 999 999."""
    if n == 0:
        return "zéro"
    th, r = divmod(n, 1000)
    parts = []
    if th:
        parts.append("mille" if th == 1 else _lt1000(th, False) + " mille")
    if r:
        parts.append(_lt1000(r, True))
    return " ".join(parts)


def amount_to_words(units, cents, currency):
    name = "dirham" if currency == "MAD" else "euro"
    s = f"{fr_words(units)} {name}{'s' if units > 1 else ''}"
    if cents:
        s += f" et {fr_words(cents)} centime{'s' if cents > 1 else ''}"
    return s


def digits(rng, k):
    return "".join(rng.choice("0123456789") for _ in range(k))


def group(s, n=4):
    return " ".join(s[i:i + n] for i in range(0, len(s), n))


# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


PRINT = _first_existing([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf"])
BOLD = _first_existing([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf"]) or PRINT
MONO = _first_existing([
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "C:/Windows/Fonts/cour.ttf"]) or PRINT
SLANT = _first_existing([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "C:/Windows/Fonts/ariali.ttf"]) or PRINT


@lru_cache(maxsize=512)
def get_font(path, size):
    size = int(size)
    try:
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def text_w(font, s):
    return font.getlength(s)


# ----------------------------------------------------------------------------
# Drawing helpers (all return a tight bbox [x0, y0, x1, y1])
# ----------------------------------------------------------------------------
def blend(c, tint, a):
    return tuple(int(c[k] * (1 - a) + tint[k] * a) for k in range(3))


def draw_printed(d, text, xy, path, size, fill):
    font = get_font(path, size)
    d.text(xy, text, font=font, fill=fill)
    return list(d.textbbox(xy, text, font=font))


def draw_hand(d, text, x, y, path, size, ink, rng, max_w):
    """Handwriting imitation: per-character baseline/size/spacing jitter."""
    size = int(size)
    while size > 16 and sum(text_w(get_font(path, size), c) for c in text) * 1.05 > max_w:
        size -= 2
    cx = x
    box = [1e9, 1e9, -1e9, -1e9]
    for ch in text:
        s = max(12, int(size * rng.uniform(0.92, 1.08)))
        font = get_font(path, s)
        dy = rng.uniform(-0.07, 0.07) * size
        d.text((cx, y + dy), ch, font=font, fill=ink)
        if not ch.isspace():
            b = d.textbbox((cx, y + dy), ch, font=font)
            box = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        cx += text_w(font, ch) * rng.uniform(0.95, 1.07)
    return box


def draw_signature(d, region, ink, rng):
    """Random scribble: uneven strokes, small loops, hook and optional underline."""
    x0, y0, x1, y1 = region
    n = rng.randint(9, 15)
    weights = [rng.uniform(0.3, 1.7) for _ in range(n)]
    total, acc, pts = sum(weights), 0.0, []
    h = y1 - y0
    for w in weights:
        acc += w
        px = x0 + (x1 - x0) * (acc / total) * rng.uniform(0.85, 1.0)
        # tall "capital" strokes at the start, smaller wiggles afterwards
        amp = 1.0 if len(pts) < 3 else rng.uniform(0.15, 0.6)
        py = y0 + h * (0.5 + rng.uniform(-0.5, 0.5) * amp)
        pts.append((px, py))
        if rng.random() < 0.22:  # small loop
            r = rng.uniform(6, 14)
            for a in (0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6):
                pts.append((px + r * math.cos(a) * 0.9, py + r * math.sin(a)))
    pts.insert(0, (pts[0][0] - rng.uniform(10, 35), pts[0][1] + rng.uniform(-30, 30)))
    pts.append((pts[-1][0] + rng.uniform(15, 60), pts[-1][1] + rng.uniform(-5, 25)))
    width = rng.randint(3, 5)
    curve = []
    P = [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in [k / 10 for k in range(10)]:
            t2, t3 = t * t, t * t * t
            curve.append(tuple(
                0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                       + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                       + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    d.line(curve, fill=ink, width=width, joint="curve")
    ux0 = min(p[0] for p in curve)
    ux1 = max(p[0] for p in curve)
    uy = max(p[1] for p in curve) + rng.randint(2, 12)
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    if rng.random() < 0.6:
        d.line([(ux0, uy), (ux1 + 20, uy - rng.randint(0, 8))], fill=ink, width=max(2, width - 1))
        xs.append(ux1 + 20)
        ys.append(uy)
    return [min(xs) - width, min(ys) - width, max(xs) + width, max(ys) + width]


def pad(box, p=4):
    return [box[0] - p, box[1] - p, box[2] + p, box[3] + p]


# ----------------------------------------------------------------------------
# One cheque
# ----------------------------------------------------------------------------
def make_cheque(rng, cfg):
    W, H = 1600, 700
    bg = tuple(rng.randint(222, 252) for _ in range(3))
    tint = rng.choice([(60, 90, 160), (150, 70, 70), (70, 140, 90), (130, 100, 60), (90, 90, 90)])
    dark = blend((0, 0, 0), tint, 0.35)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # background: header band + guilloche-like waves + border
    if rng.random() < 0.6:
        d.rectangle([8, 8, W - 9, rng.randint(105, 135)], fill=blend(bg, tint, 0.14))
    for _ in range(rng.randint(8, 16)):
        amp, freq = rng.uniform(8, 40), rng.uniform(0.004, 0.02)
        ph, y0 = rng.uniform(0, 6.28), rng.uniform(0, H)
        col = blend(bg, tint, rng.uniform(0.08, 0.22))
        d.line([(x, y0 + amp * math.sin(freq * x + ph)) for x in range(0, W, 6)], fill=col, width=1)
    d.rectangle([8, 8, W - 9, H - 9], outline=blend(bg, tint, 0.55), width=3)

    ink = rng.choice([(20, 30, 110), (10, 10, 10), (30, 40, 140), (15, 15, 60)])
    hand = rng.choice(cfg["hand_fonts"])
    label_col = blend((0, 0, 0), tint, 0.5)
    boxes, truth = {}, {}

    # ---- content
    currency = cfg["currency"]
    units = int(round(10 ** rng.uniform(1.5, 5.3)))
    cents = 0 if rng.random() < 0.5 else rng.choice([rng.randint(1, 99), 50, 25, 75])
    amount_digits = f"{units:,}".replace(",", " ") + "," + f"{cents:02d}"
    words = amount_to_words(units, cents, currency)
    if rng.random() < 0.5:
        words = words.capitalize()
    cities = CITIES_MA if (currency == "MAD" or rng.random() < 0.2) else CITIES_FR
    city = rng.choice(cities)
    year = rng.randint(2018, 2026)
    date = rng.choice(["%02d/%02d/%d", "%02d-%02d-%d", "%02d.%02d.%d"]) % (
        rng.randint(1, 28), rng.randint(1, 12), year)
    payee = (f"{rng.choice(FIRST)} {rng.choice(LAST)}" if rng.random() < 0.7
             else rng.choice(COMPANY) + rng.choice(["", " SARL", " SA"]))
    bank = rng.choice(BANKS)
    cheque_no = digits(rng, 7)
    acc = digits(rng, 24 if currency == "MAD" else 23)
    acc_txt = group(acc)

    # ---- printed header
    top = rng.randint(28, 48)
    boxes["bank_name"] = pad(draw_printed(d, bank, (50, top), BOLD, rng.randint(40, 52), dark))
    boxes["cheque_number"] = pad(draw_printed(
        d, f"N° {cheque_no}", (1250 + rng.randint(-30, 20), top + 8), PRINT, 30, dark))

    # ---- amount in words
    variant_right = rng.random() < 0.65
    y_words = 200 + rng.randint(-8, 14)
    draw_printed(d, "Payez contre ce chèque non endossable", (50, y_words - 42), PRINT, 24, label_col)
    max_w_words = 1150 if variant_right else 1480
    ws = rng.randint(44, 60)
    boxes["amount_words"] = pad(draw_hand(d, words, 60 + rng.randint(0, 20), y_words, hand, ws, ink, rng, max_w_words))
    d.line([(50, y_words + 64), (50 + max_w_words, y_words + 64)], fill=blend(bg, tint, 0.45), width=1)

    # ---- payee
    y_pay = 335 + rng.randint(-10, 10)
    draw_printed(d, "à l'ordre de", (50, y_pay + 12), PRINT, 26, label_col)
    max_w_pay = 950 if not variant_right else 1250
    boxes["payee"] = pad(draw_hand(d, payee, 270 + rng.randint(0, 15), y_pay, hand, rng.randint(46, 58), ink, rng, max_w_pay))
    d.line([(265, y_pay + 62), (265 + max_w_pay, y_pay + 62)], fill=blend(bg, tint, 0.45), width=1)

    # ---- amount box (digits)
    by = (y_words - 20) if variant_right else (y_pay - 12)
    bx0, bx1 = 1250 + rng.randint(-20, 10), 1550
    d.rectangle([bx0, by, bx1, by + 92], outline=dark, width=2, fill=blend(bg, (255, 255, 255), 0.5))
    draw_printed(d, "Montant", (bx0 + 6, by - 24), PRINT, 20, label_col)
    fsz = 50
    while text_w(get_font(hand, fsz), amount_digits) > (bx1 - bx0 - 28) and fsz > 20:
        fsz -= 2
    tw = text_w(get_font(hand, fsz), amount_digits)
    boxes["amount_digits"] = pad(draw_hand(d, amount_digits, bx1 - 14 - tw, by + 16, hand, fsz, ink, rng, tw + 40))

    # ---- place + date
    y_d = 455 + rng.randint(-10, 10)
    draw_printed(d, "À", (700, y_d + 14), PRINT, 26, label_col)
    boxes["place"] = pad(draw_hand(d, city, 745, y_d, hand, rng.randint(44, 54), ink, rng, 340))
    draw_printed(d, "le", (1115, y_d + 14), PRINT, 26, label_col)
    boxes["date"] = pad(draw_hand(d, date, 1160, y_d, hand, rng.randint(44, 52), ink, rng, 330))

    # ---- signature
    sig_region = (1030 + rng.randint(0, 40), 515, 1480, 630)
    draw_printed(d, "Signature", (1030, 490), PRINT, 20, label_col)
    d.rectangle([1020, 486, 1540, 655], outline=blend(bg, tint, 0.35), width=1)
    boxes["signature"] = pad(draw_signature(d, sig_region, ink, rng), 2)

    # ---- account number + MICR
    ya = 505 + rng.randint(-10, 10)
    lab = "RIB" if currency == "MAD" else "IBAN"
    draw_printed(d, lab, (50, ya + 2), BOLD, 24, label_col)
    boxes["account_number"] = pad(draw_printed(d, acc_txt, (50 + 90, ya), PRINT, 28, dark))
    bank_code = digits(rng, 3)
    micr_txt = f"<{cheque_no}<  {bank_code}{digits(rng, 5)}:  {acc[:12]}<{digits(rng, 2)}"
    boxes["micr_line"] = pad(draw_printed(d, micr_txt, (50, 632 + rng.randint(-6, 6)), MONO, 32, (25, 25, 25)))

    truth = {
        "bank_name": bank, "cheque_number": cheque_no, "amount_words": words,
        "payee": payee, "amount_digits": amount_digits,
        "amount_value": units + cents / 100.0, "currency": currency,
        "place": city, "date": date, "account_number": acc, "micr_line": micr_txt,
        "signature": True,
    }
    return img, boxes, truth, bg


# ----------------------------------------------------------------------------
# Scan / photo augmentation (boxes are transformed along)
# ----------------------------------------------------------------------------
def rotate_with_boxes(img, boxes, angle, fill):
    w, h = img.size
    out = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    nw, nh = out.size
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    new = {}
    for k, (x0, y0, x1, y1) in boxes.items():
        pts = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            dx, dy = x - w / 2, y - h / 2
            pts.append((nw / 2 + dx * c + dy * s, nh / 2 - dx * s + dy * c))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        new[k] = [min(xs), min(ys), max(xs), max(ys)]
    return out, new


def augment(img, boxes, bg, rng, nrng):
    img, boxes = rotate_with_boxes(img, boxes, rng.uniform(-2.5, 2.5), bg)
    f = rng.uniform(0.6, 1.0)
    if f < 0.97:
        nw, nh = int(img.width * f), int(img.height * f)
        img = img.resize((nw, nh), Image.LANCZOS)
        boxes = {k: [v * f for v in b] for k, b in boxes.items()}
    if rng.random() < 0.3:
        img = img.convert("L").convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.1))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.15))
    r = rng.uniform(0, 1.1)
    if r > 0.2:
        img = img.filter(ImageFilter.GaussianBlur(r))
    arr = np.asarray(img).astype(np.float32)
    arr += nrng.normal(0, rng.uniform(0, 9), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    W, H = img.size
    clipped = {}
    for k, (x0, y0, x1, y1) in boxes.items():
        clipped[k] = [max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)]
    return img, clipped


# ----------------------------------------------------------------------------
# Worker + main
# ----------------------------------------------------------------------------
def worker(job):
    idx, split, cfg = job
    seed = cfg["seed"] * 1_000_003 + idx
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    img, boxes, truth, bg = make_cheque(rng, cfg)
    img, boxes = augment(img, boxes, bg, rng, nrng)
    name = f"cheque_{idx:06d}"
    root = cfg["out"]
    img_path = os.path.join(root, "images", split, name + ".jpg")
    img.save(img_path, quality=rng.randint(45, 92))
    W, H = img.size
    lines = []
    for cls, (x0, y0, x1, y1) in boxes.items():
        lines.append("%d %.6f %.6f %.6f %.6f" % (
            CLASSES.index(cls), (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    with open(os.path.join(root, "labels", split, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"idx": idx, "split": split, "file": f"images/{split}/{name}.jpg",
            "w": W, "h": H, "boxes": boxes, "truth": truth}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="number of cheques")
    ap.add_argument("--out", default="cheque_dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--currency", choices=["MAD", "EUR"], default="MAD")
    ap.add_argument("--fonts-dir", default=None, help="folder with handwriting .ttf/.otf fonts")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--preview", type=int, default=0, help="save N previews with boxes drawn")
    a = ap.parse_args()

    hand_fonts = []
    if a.fonts_dir:
        hand_fonts = sorted(glob.glob(os.path.join(a.fonts_dir, "*.ttf")) +
                            glob.glob(os.path.join(a.fonts_dir, "*.otf")))
    if not hand_fonts:
        hand_fonts = [SLANT]
        print("No handwriting fonts found: using a slanted fallback font (see --fonts-dir).")
    cfg = {"out": a.out, "seed": a.seed, "currency": a.currency, "hand_fonts": hand_fonts}

    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(a.out, sub, sp), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    if a.preview:
        os.makedirs(os.path.join(a.out, "preview"), exist_ok=True)

    n_train, n_val = int(a.n * 0.8), int(a.n * 0.1)
    jobs = [(i, "train" if i < n_train else "val" if i < n_train + n_val else "test", cfg)
            for i in range(a.n)]
    with Pool(a.workers) as pool:
        records = []
        for k, r in enumerate(pool.imap(worker, jobs, chunksize=16), 1):
            records.append(r)
            if k % 500 == 0 or k == a.n:
                print(f"{k}/{a.n}")

    # COCO per split + ground-truth JSONL
    coco = {sp: {"images": [], "annotations": [],
                 "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
            for sp in ("train", "val", "test")}
    ann_id = 1
    with open(os.path.join(a.out, "ground_truth.jsonl"), "w", encoding="utf-8") as gt:
        for r in records:
            c = coco[r["split"]]
            c["images"].append({"id": r["idx"], "file_name": os.path.basename(r["file"]),
                                "width": r["w"], "height": r["h"]})
            for cls, (x0, y0, x1, y1) in r["boxes"].items():
                c["annotations"].append({
                    "id": ann_id, "image_id": r["idx"], "category_id": CLASSES.index(cls) + 1,
                    "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                    "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0})
                ann_id += 1
            gt.write(json.dumps({"file_name": r["file"], "split": r["split"],
                                 "ground_truth": r["truth"]}, ensure_ascii=False) + "\n")
    for sp, data in coco.items():
        with open(os.path.join(a.out, "annotations", f"{sp}.json"), "w") as f:
            json.dump(data, f)

    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

    for r in records[:a.preview]:
        im = Image.open(os.path.join(a.out, r["file"])).convert("RGB")
        dr = ImageDraw.Draw(im)
        for cls, (x0, y0, x1, y1) in r["boxes"].items():
            dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            dr.text((x0, max(0, y0 - 12)), cls, fill=(255, 0, 0))
        im.save(os.path.join(a.out, "preview", os.path.basename(r["file"])))

    print(f"Done: {a.n} cheques in '{a.out}' ({n_train} train / {n_val} val / {a.n - n_train - n_val} test)")


if __name__ == "__main__":
    main()
