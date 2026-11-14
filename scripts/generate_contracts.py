#!/usr/bin/env python3
"""
Synthetic FINANCIAL CONTRACT dataset generator (French): images + labels.

Contract types: personal loan, car loan, mortgage ("prêt immobilier").
Every image is one scanned A4 page: letterhead, title, parties, articles with the key
terms (amount, rate, duration, monthly payment, IBAN...), signature block, stamp.
Layouts vary (fonts, margins, header style, numbering, justified text, bold values...).

For every contract it writes:
  - a JPEG scan (rotation, shadow gradient, blur, noise, JPEG compression)
  - YOLO labels (.txt)          -> images/<split>/, labels/<split>/
  - COCO annotations (.json)    -> annotations/<split>.json
  - ground truth (.jsonl)       -> ground_truth.jsonl (all the field values)
  - data.yaml for Ultralytics YOLO

Classes (19): title, contract_number, contract_date, lender_name, borrower_name,
  borrower_id, borrower_address, amount, amount_words, interest_rate, duration,
  monthly_payment, first_payment_date, iban, place, signing_date,
  signature_lender, signature_borrower, stamp

Requirements:  pip install pillow numpy
Usage:
  python generate_contracts.py --n 10000 --out contract_dataset
  python generate_contracts.py --n 10000 --currency EUR --type mortgage
  python generate_contracts.py --n 30 --preview 10        # quick test with box previews
"""
import argparse
import json
import math
import os
import random
import re
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CLASSES = [
    "title", "contract_number", "contract_date", "lender_name", "borrower_name",
    "borrower_id", "borrower_address", "amount", "amount_words", "interest_rate",
    "duration", "monthly_payment", "first_payment_date", "iban", "place",
    "signing_date", "signature_lender", "signature_borrower", "stamp",
]

# ----------------------------------------------------------------------------
# Random content
# ----------------------------------------------------------------------------
BANKS = ["Banque Atlas", "Crédit du Souss", "Banque Horizon", "Banque Al Amal",
         "Caisse Centrale d'Épargne", "Banque Medina", "Crédit Régional du Nord",
         "Banque Nationale du Sud", "Union Bancaire Maghreb", "Banque Océane",
         "Société Financière Atlas", "Crédit Maghreb Finance"]
FIRST = ["Mohamed", "Fatima", "Youssef", "Khadija", "Ahmed", "Amina", "Omar", "Salma",
         "Hassan", "Zineb", "Karim", "Imane", "Rachid", "Nadia", "Pierre", "Marie",
         "Jean", "Sophie", "Lucas", "Camille", "Thomas", "Julie", "Nicolas", "Laura"]
LAST = ["Alaoui", "Bennani", "El Idrissi", "Tazi", "Berrada", "Lahlou", "Chraibi",
        "Benjelloun", "Fassi", "Amrani", "Martin", "Bernard", "Dubois", "Moreau",
        "Laurent", "Simon", "Michel", "Lefebvre", "Garcia", "Roux"]
STREETS = ["Mohammed V", "Hassan II", "de la Liberté", "des Écoles", "Al Massira",
           "de la République", "Victor Hugo", "du Commerce", "des Fleurs", "Ibn Sina"]
CITIES_MA = ["Marrakech", "Casablanca", "Rabat", "Fès", "Agadir", "Tanger", "Meknès", "Oujda"]
CITIES_FR = ["Paris", "Lyon", "Marseille", "Lille", "Toulouse", "Bordeaux", "Nantes", "Nice"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre"]

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


def _lt1m(n, final):
    th, r = divmod(n, 1000)
    parts = []
    if th:
        parts.append("mille" if th == 1 else _lt1000(th, False) + " mille")
    if r:
        parts.append(_lt1000(r, final))
    return " ".join(parts)


def fr_words(n):
    """French number to words, 0 <= n <= 999 999 999."""
    if n == 0:
        return "zéro"
    m, r = divmod(n, 1_000_000)
    parts = []
    if m:
        parts.append("un million" if m == 1 else _lt1000(m, False) + " millions")
    if r:
        parts.append(_lt1m(r, True))
    return " ".join(parts)


def amount_words(units, cents, cur_name):
    de = "de " if units % 1_000_000 == 0 and units else ""
    s = f"{fr_words(units)} {de}{cur_name}{'s' if units > 1 else ''}"
    if cents:
        s += f" et {fr_words(cents)} centime{'s' if cents > 1 else ''}"
    return s


def digits(rng, k):
    return "".join(rng.choice("0123456789") for _ in range(k))


def group(s, n=4):
    return " ".join(s[i:i + n] for i in range(0, len(s), n))


def fmt_num(v):
    return f"{v:,.2f}".replace(",", " ").replace(".", ",")


# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
FONT_PAIRS = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
    ("C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/georgiab.ttf"),
    ("C:/Windows/Fonts/cambria.ttc", "C:/Windows/Fonts/cambriab.ttf"),
    ("/Library/Fonts/Times New Roman.ttf", "/Library/Fonts/Times New Roman Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]
PAIRS = [p for p in FONT_PAIRS if os.path.exists(p[0]) and os.path.exists(p[1])] or [(None, None)]


@lru_cache(maxsize=256)
def get_font(path, size):
    size = int(size)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Text flow engine: word-wrapping with per-field bounding boxes
# ----------------------------------------------------------------------------
def tokenize(segs):
    toks, pend = [], False
    for text, field, bold in segs:
        for m in re.finditer(r"\S+|\s+", text):
            t = m.group()
            if t.isspace():
                pend = True
            else:
                toks.append((t, field, bold, pend))
                pend = False
    return toks


def flow(d, segs, x0, x1, y, fr, fb, lh, justify, boxes, indent=0, draw=True):
    """Lay out [(text, field, bold), ...] between x0 and x1. Returns the y after the paragraph."""
    sp = fr.getlength(" ")
    lines, cur, x = [], [], x0 + indent
    for text, field, bold, space in tokenize(segs):
        f = fb if bold else fr
        w = f.getlength(text)
        gap = 1 if (cur and space) else 0
        if cur and x + gap * sp + w > x1:
            if gap:                                  # normal break between two words
                lines.append(cur)
                cur, x, gap = [], x0, 0
            else:                                    # glued to the previous word, e.g. "(" + word: break before the whole chunk
                k = max([i for i, t in enumerate(cur) if t[4] == 1] or [0])
                if k >= 1:
                    lines.append(cur[:k])
                    cur = cur[k:]
                    cur[0][4] = 0
                    x = x0 + sum(t[3] for t in cur) + sum(t[4] for t in cur) * sp
        cur.append([text, field, f, w, gap])
        x += gap * sp + w
    if cur:
        lines.append(cur)
    if not draw:
        return y + len(lines) * lh
    for li, line in enumerate(lines):
        start = x0 + (indent if li == 0 else 0)
        nsp = sum(w[4] for w in line)
        natural = sum(w[3] for w in line) + nsp * sp
        extra = 0.0
        if justify and li < len(lines) - 1 and nsp:
            extra = max(0.0, (x1 - start - natural) / nsp)
        x, run = start, None
        for text, field, f, w, gap in line:
            x += gap * (sp + extra)
            d.text((x, y), text, font=f, fill=INK[0])
            if field:
                b = f.getbbox(text)
                x_l, y_t, x_r, y_b = x, y + b[1], x + w, y + b[3]
                if run and run[0] == field:
                    run[1] = min(run[1], x_l)
                    run[2] = min(run[2], y_t)
                    run[3] = max(run[3], x_r)
                    run[4] = max(run[4], y_b)
                else:
                    if run:
                        boxes.append((run[0], [run[1] - 2, run[2] - 2, run[3] + 2, run[4] + 2]))
                    run = [field, x_l, y_t, x_r, y_b]
            else:
                if run:
                    boxes.append((run[0], [run[1] - 2, run[2] - 2, run[3] + 2, run[4] + 2]))
                    run = None
            x += w
        if run:
            boxes.append((run[0], [run[1] - 2, run[2] - 2, run[3] + 2, run[4] + 2]))
        y += lh
    return y


INK = [(20, 20, 20)]   # current page ink (set per page)


def put(d, text, x, y, font, anchor="l", fill=None):
    w = font.getlength(text)
    x0 = x if anchor == "l" else x - w if anchor == "r" else x - w / 2
    d.text((x0, y), text, font=font, fill=fill or INK[0])
    b = d.textbbox((x0, y), text, font=font)
    return [b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2]


def put_pair(d, label, value, x, y, font, anchor="l"):
    total = font.getlength(label + value)
    x0 = x if anchor == "l" else x - total if anchor == "r" else x - total / 2
    d.text((x0, y), label, font=font, fill=INK[0])
    vx = x0 + font.getlength(label)
    d.text((vx, y), value, font=font, fill=INK[0])
    b = d.textbbox((vx, y), value, font=font)
    return [b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2]


# ----------------------------------------------------------------------------
# Signature and stamp
# ----------------------------------------------------------------------------
def draw_signature(d, region, ink, rng):
    x0, y0, x1, y1 = region
    n = rng.randint(9, 15)
    weights = [rng.uniform(0.3, 1.7) for _ in range(n)]
    total, acc, pts = sum(weights), 0.0, []
    h = y1 - y0
    for w in weights:
        acc += w
        px = x0 + (x1 - x0) * (acc / total) * rng.uniform(0.85, 1.0)
        amp = 1.0 if len(pts) < 3 else rng.uniform(0.15, 0.6)
        py = y0 + h * (0.5 + rng.uniform(-0.5, 0.5) * amp)
        pts.append((px, py))
        if rng.random() < 0.22:
            r = rng.uniform(5, 11)
            for a in (0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6):
                pts.append((px + r * math.cos(a) * 0.9 - r * 0.9, py + r * math.sin(a)))
    pts.insert(0, (pts[0][0] - rng.uniform(10, 35), pts[0][1] + rng.uniform(-30, 30)))
    pts.append((pts[-1][0] + rng.uniform(15, 60), pts[-1][1] + rng.uniform(-5, 25)))
    width = rng.randint(2, 4)
    curve, P = [], [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in [k / 10 for k in range(10)]:
            t2, t3 = t * t, t * t * t
            curve.append(tuple(
                0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                       + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                       + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    d.line(curve, fill=ink, width=width, joint="curve")
    xs, ys = [p[0] for p in curve], [p[1] for p in curve]
    if rng.random() < 0.6:
        ux0, ux1, uy = min(xs), max(xs), max(ys) + rng.randint(2, 12)
        d.line([(ux0, uy), (ux1 + 20, uy - rng.randint(0, 8))], fill=ink, width=max(2, width - 1))
        xs.append(ux1 + 20)
        ys.append(uy)
    return [min(xs) - width, min(ys) - width, max(xs) + width, max(ys) + width]


def draw_stamp(page, cx, cy, r, lines, color, rng, font_path):
    size = 2 * r + 24
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dd = ImageDraw.Draw(layer)
    c = size // 2
    col = color + (205,)
    dd.ellipse([c - r, c - r, c + r, c + r], outline=col, width=4)
    dd.ellipse([c - r + 9, c - r + 9, c + r - 9, c + r - 9], outline=col, width=2)
    fs = max(11, r // 4)
    y = c - len(lines) * fs * 0.6
    for i, ln in enumerate(lines):
        f = get_font(font_path, fs)
        while f.getlength(ln) > 1.5 * r and fs > 9:
            fs -= 1
            f = get_font(font_path, fs)
        dd.text((c - f.getlength(ln) / 2, y + i * fs * 1.25), ln, font=f, fill=col)
    layer = layer.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC)
    page.paste(layer.convert("RGB"), (cx - c, cy - c), layer.getchannel("A").point(lambda a: int(a * 0.85)))
    return [cx - r, cy - r, cx + r, cy + r]


# ----------------------------------------------------------------------------
# Contract content
# ----------------------------------------------------------------------------
TYPES = {
    # key: (title, number prefix, amount range MAD, amount range EUR, rate range, months choices)
    "personal": ("CONTRAT DE PRÊT PERSONNEL", "CP", (10_000, 300_000), (1_000, 40_000), (5.5, 12.5), (12, 24, 36, 48, 60, 72, 84)),
    "car": ("CONTRAT DE CRÉDIT AUTOMOBILE", "CA", (60_000, 400_000), (5_000, 50_000), (4.5, 9.5), (24, 36, 48, 60, 72, 84)),
    "mortgage": ("CONTRAT DE PRÊT IMMOBILIER", "CI", (300_000, 1_500_000), (80_000, 500_000), (3.0, 6.5), (120, 180, 240, 300)),
}

OBJECT = {
    "personal": "un prêt personnel d'un montant en principal de {A} ({W}), destiné au financement des besoins personnels de l'Emprunteur.",
    "car": "un crédit d'un montant en principal de {A} ({W}), destiné à l'acquisition d'un véhicule automobile.",
    "mortgage": "un prêt immobilier d'un montant en principal de {A} ({W}), destiné à l'acquisition d'un bien immobilier à usage d'habitation.",
}

FILLERS = [
    ("Retard de paiement", "Toute somme non payée à son échéance portera de plein droit intérêt à un taux majoré de trois points, sans mise en demeure préalable. Les frais de recouvrement éventuels restent à la charge de l'Emprunteur."),
    ("Remboursement anticipé", "L'Emprunteur peut à tout moment rembourser tout ou partie du capital restant dû, après en avoir informé le Prêteur par écrit trente jours à l'avance. Une indemnité égale à un pour cent du capital remboursé par anticipation pourra être perçue."),
    ("Déchéance du terme", "Le Prêteur pourra exiger le remboursement immédiat du capital restant dû, des intérêts échus et des accessoires en cas de non-paiement d'une échéance, de déclaration inexacte ou de cessation de paiement de l'Emprunteur."),
    ("Assurance", "L'Emprunteur déclare avoir souscrit une assurance décès-invalidité couvrant le montant du prêt pendant toute sa durée. Le Prêteur est désigné bénéficiaire de cette assurance à hauteur du capital restant dû."),
    ("Garanties", "En garantie du remboursement du prêt, l'Emprunteur constitue au profit du Prêteur les sûretés convenues entre les parties. Il s'engage à ne pas céder ni donner en garantie à un tiers les biens concernés sans l'accord écrit du Prêteur."),
    ("Protection des données", "Les données personnelles de l'Emprunteur sont traitées par le Prêteur pour la gestion du présent contrat, dans le respect de la législation en vigueur. L'Emprunteur dispose d'un droit d'accès et de rectification."),
    ("Réclamations", "Toute réclamation relative à l'exécution du présent contrat doit être adressée par écrit au service clientèle du Prêteur, qui s'engage à répondre dans un délai de quinze jours ouvrables."),
    ("Notifications", "Toute notification entre les parties est valablement faite par lettre recommandée avec accusé de réception à l'adresse indiquée en tête des présentes. Tout changement d'adresse doit être notifié sans délai."),
    ("Droit de rétractation", "L'Emprunteur dispose d'un délai de rétractation de sept jours calendaires à compter de la signature du présent contrat, qu'il peut exercer sans avoir à justifier de motif."),
    ("Droit applicable", "Le présent contrat est soumis au droit applicable. Tout litige relatif à son interprétation ou à son exécution relève de la compétence exclusive des tribunaux du siège social du Prêteur."),
]


def make_content(rng, cfg):
    cur = cfg["currency"]
    ctype = cfg["type"] if cfg["type"] != "random" else rng.choice(list(TYPES))
    title, prefix, (lo_m, hi_m), (lo_e, hi_e), (rlo, rhi), months_ch = TYPES[ctype]
    lo, hi = (lo_m, hi_m) if cur == "MAD" else (lo_e, hi_e)
    amount = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))) / 100)) * 100
    cents = 0
    rate = round(rng.uniform(rlo, rhi) * 20) / 20
    n = rng.choice(months_ch)
    r = rate / 1200
    pay = amount * r / (1 - (1 + r) ** -n)
    cur_name = "dirham" if cur == "MAD" else "euro"
    unit = rng.choice(["dirhams", "MAD", "DH"]) if cur == "MAD" else rng.choice(["euros", "EUR", "€"])
    year = rng.randint(2018, 2026)
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    long_dates = rng.random() < 0.5

    def fdate(dy, mo, yr):
        return f"{dy} {MONTHS[mo - 1]} {yr}" if long_dates else f"{dy:02d}/{mo:02d}/{yr}"

    fm = month % 12 + 1
    fy = year + (1 if month == 12 else 0)
    cities = CITIES_MA if cur == "MAD" else CITIES_FR
    zipc = lambda: digits(rng, 5)
    bank = rng.choice(BANKS)
    sep = rng.choice(["-", "/", ""])
    cin = rng.choice(["", "A", "B", "AB", "BE", "JA", "K"]) + digits(rng, rng.choice([5, 6]))
    iban = ("MA64 " if cur == "MAD" else "FR76 ") + group(digits(rng, 24 if cur == "MAD" else 23))
    civ = rng.choice(["Monsieur", "Madame"])
    return {
        "type": ctype, "title": title, "currency": cur,
        "contract_number": f"{prefix}{sep}{year}{sep}{digits(rng, 6)}",
        "contract_date": fdate(day, month, year),
        "first_payment_date": fdate(rng.randint(1, 28), fm, fy),
        "lender_name": bank, "legal_form": rng.choice([("société anonyme", "e"), ("société financière", "e"), ("établissement de crédit", "")]),
        "capital": f"{rng.randint(50, 900)} 000 000 {unit}",
        "rc": digits(rng, 6),
        "lender_address": f"{rng.randint(1, 250)} Bd {rng.choice(STREETS)}, {zipc()} {rng.choice(cities)}",
        "borrower_name": f"{rng.choice(FIRST)} {rng.choice(LAST).upper() if rng.random() < 0.5 else rng.choice(LAST)}",
        "civ": civ, "birth": fdate(rng.randint(1, 28), rng.randint(1, 12), rng.randint(1955, 2000)),
        "borrower_id": cin,
        "borrower_address": f"{rng.randint(1, 250)} {rng.choice(['Rue', 'Av.', 'Bd'])} {rng.choice(STREETS)}, {zipc()} {rng.choice(cities)}",
        "amount": f"{fmt_num(amount)} {unit}",
        "amount_words": amount_words(amount, cents, cur_name),
        "interest_rate": f"{fmt_num(rate)} %",
        "duration": f"{n} mois" if (n % 12 or rng.random() < 0.5) else f"{n // 12} ans",
        "monthly_payment": f"{fmt_num(pay)} {unit}",
        "iban": iban, "place": rng.choice(cities), "signing_date": fdate(day, month, year),
        "amount_value": amount, "rate_value": rate, "months": n, "monthly_value": round(pay, 2),
    }


# ----------------------------------------------------------------------------
# Page rendering
# ----------------------------------------------------------------------------
class Overflow(Exception):
    pass


def build_page(rng, c, fs):
    W, H = 1240, 1754
    mx = rng.randint(90, 130)
    x0, x1 = mx, W - mx
    fpath, bpath = rng.choice(PAIRS)
    fr, fb = get_font(fpath, fs), get_font(bpath, fs)
    lh = int(fs * rng.uniform(1.4, 1.55))
    justify = rng.random() < 0.6
    vbold = rng.random() < 0.5
    indent = rng.choice([0, 0, 30])
    paper = tuple(min(255, rng.randint(240, 255) - rng.randint(0, 10) * k) for k in (0, 0, 1))
    g = rng.randint(10, 45)
    INK[0] = (g, g, min(255, g + rng.randint(0, 20)))
    accent = rng.choice([(30, 60, 130), (120, 30, 30), (30, 100, 60), (60, 60, 60)])
    page = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(page)
    boxes = []

    def V(field, text):
        return (text, field, vbold)

    def T(text, bold=False):
        return (text, None, bold)

    # ---- letterhead
    top = rng.randint(70, 105)
    style = rng.choice(["left", "center"])
    big = get_font(bpath, fs + rng.randint(8, 14))
    small = get_font(fpath, max(13, fs - 6))
    if style == "left":
        nx = x0
        if rng.random() < 0.6:     # logo block
            ls = int(big.size * 1.6)
            if rng.random() < 0.5:
                d.rectangle([x0, top, x0 + ls, top + ls], fill=accent)
            else:
                d.ellipse([x0, top, x0 + ls, top + ls], fill=accent)
            ini = "".join(w[0] for w in c["lender_name"].split()[:2]).upper()
            d.text((x0 + ls / 2 - big.getlength(ini) / 2, top + ls / 2 - big.size / 2), ini, font=big, fill=(255, 255, 255))
            nx = x0 + ls + 18
        boxes.append(("lender_name", put(d, c["lender_name"], nx, top, big, fill=accent)))
        d.text((nx, top + int(big.size * 1.35)), c["lender_address"], font=small, fill=INK[0])
        rx, ry = x1, top
        boxes.append(("contract_number", put_pair(d, "Contrat n° ", c["contract_number"], rx, ry, fr, "r")))
        boxes.append(("contract_date", put_pair(d, "Date : ", c["contract_date"], rx, ry + lh, fr, "r")))
        y = top + max(int(big.size * 1.35) + small.size, 2 * lh) + 30
    else:
        boxes.append(("lender_name", put(d, c["lender_name"], W / 2, top, big, "c", accent)))
        d.text((W / 2 - small.getlength(c["lender_address"]) / 2, top + int(big.size * 1.35)), c["lender_address"], font=small, fill=INK[0])
        y = top + int(big.size * 1.35) + small.size + 22
        boxes.append(("contract_number", put_pair(d, "Contrat n° ", c["contract_number"], x0, y, fr)))
        boxes.append(("contract_date", put_pair(d, "Date : ", c["contract_date"], x1, y, fr, "r")))
        y += lh + 14
    d.line([(x0, y), (x1, y)], fill=accent, width=3)
    y += rng.randint(35, 60)

    # ---- title
    tf = get_font(bpath, fs + rng.randint(6, 12))
    tb = put(d, c["title"], W / 2, y, tf, "c")
    boxes.append(("title", tb))
    ts = rng.choice(["plain", "box", "underline"])
    if ts == "box":
        d.rectangle([tb[0] - 18, tb[1] - 10, tb[2] + 18, tb[3] + 10], outline=INK[0], width=2)
    elif ts == "underline":
        d.line([(tb[0], tb[3] + 6), (tb[2], tb[3] + 6)], fill=INK[0], width=2)
    y = tb[3] + rng.randint(38, 56)

    # ---- parties
    def para(segs, gap=int(lh * 0.35), ind=None):
        nonlocal y
        y = flow(d, segs, x0, x1, y, fr, fb, lh, justify, boxes, indent if ind is None else ind) + gap

    para([T("ENTRE LES SOUSSIGNÉS", True)], gap=int(lh * 0.2), ind=0)
    lf, e = c["legal_form"]
    para([V("lender_name", c["lender_name"]),
          T(f", {lf} au capital de {c['capital']}, immatriculé{e} au registre du commerce sous le n° {c['rc']}, "
            f"dont le siège social est situé {c['lender_address']}, représenté{e} par le directeur de son agence, "
            f"ci-après dénommé{e} « le Prêteur »,")])
    para([T("d'une part, et")], ind=0)
    para([T(f"{c['civ']} "), V("borrower_name", c["borrower_name"]),
          T(f", né(e) le {c['birth']}, titulaire de la carte nationale d'identité n° "), V("borrower_id", c["borrower_id"]),
          T(", demeurant à "), V("borrower_address", c["borrower_address"]),
          T(", ci-après dénommé(e) « l'Emprunteur », d'autre part.")])
    para([T("IL A ÉTÉ CONVENU ET ARRÊTÉ CE QUI SUIT :", True)], gap=int(lh * 0.5), ind=0)

    # ---- articles
    num = rng.choice(["Article {n} – {t}", "ARTICLE {n} : {t}", "{n}. {t}"])
    art_no = [0]

    def article(title, segs, check_limit=None):
        nonlocal y
        art_no[0] += 1
        head = num.format(n=art_no[0], t=title.upper() if "ARTICLE" in num else title)
        h_end = flow(d, [T(head, True)], x0, x1, y, fr, fb, lh, False, boxes, 0, draw=False)
        b_end = flow(d, segs, x0, x1, h_end, fr, fb, lh, justify, boxes, indent, draw=False)
        if check_limit is not None and b_end + 14 > check_limit:
            art_no[0] -= 1
            return False
        y = flow(d, [T(head, True)], x0, x1, y, fr, fb, lh, False, boxes, 0)
        y = flow(d, segs, x0, x1, y, fr, fb, lh, justify, boxes, indent) + int(lh * 0.45)
        return True

    SIG_H = 330
    sig_limit = H - 130 - SIG_H                      # last y where articles may end
    obj = OBJECT[c["type"]].split("{A}")
    a_pre, rest = obj[0], obj[1].split("{W}")
    segs1 = [T("Le Prêteur consent à l'Emprunteur, qui l'accepte, " + a_pre), V("amount", c["amount"]), T(rest[0]),
             V("amount_words", c["amount_words"]), T(rest[1])]
    essential = [
        ("Objet du contrat", segs1),
        ("Durée", [T("Le contrat est conclu pour une durée de "), V("duration", c["duration"]),
                   T(", à compter de la date de mise à disposition des fonds.")]),
        ("Taux d'intérêt", [T("Le taux annuel effectif global (TAEG) est fixe et s'élève à "), V("interest_rate", c["interest_rate"]),
                            T(" pendant toute la durée du contrat.")]),
        ("Remboursement", [T("L'Emprunteur s'engage à rembourser le capital et les intérêts en mensualités constantes de "),
                           V("monthly_payment", c["monthly_payment"]), T(", la première échéance étant fixée au "),
                           V("first_payment_date", c["first_payment_date"]),
                           T(". Les échéances sont prélevées sur le compte de l'Emprunteur n° "), V("iban", c["iban"]), T(".")]),
    ]
    for t, s in essential:
        if not article(t, s, check_limit=sig_limit):
            raise Overflow()
    fill = FILLERS[:]
    rng.shuffle(fill)
    last = [f for f in fill if f[0] == "Droit applicable"]
    fill = [f for f in fill if f[0] != "Droit applicable"][:rng.randint(2, 6)] + last[:1]
    for t, s in fill:
        article(t, [T(s)], check_limit=sig_limit)

    # ---- signature block
    y_sig = y + 20 if rng.random() < 0.5 else sig_limit + 10
    y_sig = min(y_sig, H - 130 - SIG_H + 10)
    ph = [T("Fait à "), V("place", c["place"]), T(", le "), V("signing_date", c["signing_date"]),
          T(", en deux exemplaires originaux.")]
    y2 = flow(d, ph, x0, x1, y_sig, fr, fb, lh, False, boxes, 0)
    colw = (x1 - x0) // 2
    yc = y2 + int(lh * 0.8)
    d.text((x0 + 20, yc), "Le Prêteur", font=fb, fill=INK[0])
    d.text((x0 + 20, yc + lh), "(cachet et signature)", font=small, fill=INK[0])
    d.text((x0 + colw + 20, yc), "L'Emprunteur", font=fb, fill=INK[0])
    d.text((x0 + colw + 20, yc + lh), "(« Lu et approuvé »)", font=small, fill=INK[0])
    ink_sig = rng.choice([(20, 30, 110), (10, 10, 10), (30, 40, 140)])
    reg_y = yc + int(lh * 2.1)
    lbox = draw_signature(d, (x0 + 40, reg_y, x0 + colw - 60, reg_y + 110), ink_sig, rng)
    rbox = draw_signature(d, (x0 + colw + 40, reg_y, x1 - 60, reg_y + 110), ink_sig, rng)
    boxes.append(("signature_lender", lbox))
    boxes.append(("signature_borrower", rbox))
    if rng.random() < 0.85:
        r = rng.randint(58, 80)
        cx = int((lbox[0] + lbox[2]) / 2 + rng.randint(-40, 60))
        cy = int(reg_y + 55 + rng.randint(-20, 25))
        scol = rng.choice([(30, 60, 170), (170, 30, 40), (60, 60, 60)])
        short = c["lender_name"].upper()
        boxes.append(("stamp", draw_stamp(page, cx, cy, r, [short, c["place"].upper()], scol, rng, bpath)))

    # ---- footer
    d.line([(x0, H - 100), (x1, H - 100)], fill=blend(paper, (0, 0, 0), 0.25), width=1)
    d.text((x0, H - 88), f"{c['lender_name']} – Contrat n° {c['contract_number']}", font=small, fill=INK[0])
    d.text((x1 - small.getlength("Page 1 / 1"), H - 88), "Page 1 / 1", font=small, fill=INK[0])
    return page, boxes, paper


def blend(c, tint, a):
    return tuple(int(c[k] * (1 - a) + tint[k] * a) for k in range(3))


def make_contract(rng, cfg):
    c = make_content(rng, cfg)
    fs = rng.randint(20, 24)
    while fs >= 15:
        try:
            page, boxes, paper = build_page(rng, c, fs)
            return page, boxes, c
        except Overflow:
            fs -= 1
    raise RuntimeError("page overflow")


# ----------------------------------------------------------------------------
# Scan augmentation (boxes follow every transform)
# ----------------------------------------------------------------------------
def augment(img, boxes, rng, nrng):
    w, h = img.size
    angle = rng.uniform(-1.8, 1.8)
    bedv = rng.choice([rng.randint(235, 255), rng.randint(120, 200), rng.randint(20, 60)])
    out = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(bedv, bedv, bedv))
    nw, nh = out.size
    th = math.radians(angle)
    cs, sn = math.cos(th), math.sin(th)
    nb = []
    for cls, (x0, y0, x1, y1) in boxes:
        pts = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            dx, dy = x - w / 2, y - h / 2
            pts.append((nw / 2 + dx * cs + dy * sn, nh / 2 - dx * sn + dy * cs))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        nb.append((cls, [min(xs), min(ys), max(xs), max(ys)]))
    boxes = nb
    img = out
    f = rng.uniform(0.6, 1.0)
    if f < 0.97:
        img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
        boxes = [(k, [v * f for v in b]) for k, b in boxes]
    if rng.random() < 0.35:
        img = img.convert("L").convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.1))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    r = rng.uniform(0, 1.0)
    if r > 0.2:
        img = img.filter(ImageFilter.GaussianBlur(r))
    arr = np.asarray(img).astype(np.float32)
    # soft illumination gradient (scanner / phone light)
    gh, gw = arr.shape[:2]
    gx = np.linspace(rng.uniform(-14, 0), rng.uniform(0, 14), gw)[None, :, None]
    gy = np.linspace(rng.uniform(-14, 0), rng.uniform(0, 14), gh)[:, None, None]
    arr += gx + gy
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
    page, boxes, content = make_contract(rng, cfg)
    img, boxes = augment(page, boxes, rng, nrng)
    name = f"contract_{idx:06d}"
    root = cfg["out"]
    img.save(os.path.join(root, "images", split, name + ".jpg"), quality=rng.randint(50, 92))
    W, H = img.size
    lines = ["%d %.6f %.6f %.6f %.6f" % (
        CLASSES.index(c), (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H)
        for c, b in boxes]
    with open(os.path.join(root, "labels", split, name + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    keep = ("type", "title", "currency", "contract_number", "contract_date", "lender_name", "borrower_name",
            "borrower_id", "borrower_address", "amount", "amount_words", "interest_rate", "duration",
            "monthly_payment", "first_payment_date", "iban", "place", "signing_date",
            "amount_value", "rate_value", "months", "monthly_value")
    return {"idx": idx, "split": split, "file": f"images/{split}/{name}.jpg", "w": W, "h": H,
            "boxes": boxes, "truth": {k: content[k] for k in keep}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="number of contracts")
    ap.add_argument("--out", default="contract_dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--currency", choices=["MAD", "EUR"], default="MAD")
    ap.add_argument("--type", choices=["random", "personal", "car", "mortgage"], default="random")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--preview", type=int, default=0, help="save N previews with boxes drawn")
    a = ap.parse_args()
    cfg = {"out": a.out, "seed": a.seed, "currency": a.currency, "type": a.type}

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
        for k, r in enumerate(pool.imap(worker, jobs, chunksize=8), 1):
            records.append(r)
            if k % 250 == 0 or k == a.n:
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
            dr.text((x0, max(0, y0 - 11)), cls, fill=(255, 0, 0))
        im.save(os.path.join(a.out, "preview", os.path.basename(r["file"])))

    print(f"Done: {a.n} contracts in '{a.out}' ({n_train} train / {n_val} val / {a.n - n_train - n_val} test)")


if __name__ == "__main__":
    main()
