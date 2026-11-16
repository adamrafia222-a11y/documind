"""Find the handwritten fields of a cheque from the PAGE LAYOUT - the safety net behind the YOLO detector of `cheques.py`.

The detector is trained on a few cheque layouts. On another one (a British cheque, a bank it has never seen) it misses boxes: no date, no payee, no amount
in figures - or it puts the "amount in words" label on the payee line. Here the whole page is read once, as sparse text, and the fields are located by what
they ARE, not by where one layout puts them:

  date              a date (22/01/2006) next to the word "Date" or, when the digits were not read, the space right of the label "Date"
  amount in figures the large digits next to a currency sign (£ $ € or MAD / DH), or the largest number in the right half of the cheque
  amount in words   the line made of number words ("One Hundred Pounds Only", "Mille deux cents dirhams"), English or French
  payee             the handwritten line right above the amount in words, or right of the printed label (Pay / Payez / à l'ordre de)

Every field becomes a box; the box is read by `ocr_box` (Tesseract + the handwriting recognizer of documind.ocr.handwriting), so a cursive date or name is
read by the model made for handwriting. `cheques.extract_cheque_bytes` uses this for the fields the detector missed or read as something unusable.
"""
from __future__ import annotations

import difflib
import re

import cv2
import numpy as np

from ..ocr import run_ocr
from .cheques import _CENTIME, _CURRENCY, _UNITS, parse_amount, parse_date, words_to_number
from .dates import find_dates
from .contract_text import _ONES, _SCALE, _TENS, _Flat, english_words_to_number, fold
from .postprocess import _lines
from .receipts import ocr_box

DATE_RE = re.compile(r"(?<!\d)\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*(?:\d{4}|\d{2})(?!\d)")
SYMBOL_RE = re.compile(r"[£$€]|(?<![A-Za-z])(?:MAD|DHS?|EUR|GBP|USD)(?![A-Za-z])")
LABEL_PAY = {"pay", "payez", "payer", "ordre", "order", "to", "a"}
_FUNCTION = {"and", "et", "de", "only", "of", "seulement", "zero"}
_NUMBER_WORDS = (set(_ONES) | set(_TENS) | set(_SCALE) | {"hundred", "pound", "pounds", "pence", "penny", "dollar", "dollars", "cents", "cent", "euro", "euros", "dirham", "dirhams"}
                 | set(_UNITS) | {"cent", "cents", "mille", "million", "millions"} | set(_CURRENCY) | set(_CENTIME)) - {"zero"}
_STRONG = {"hundred", "thousand", "million", "mille", "cent", "cents", "pound", "pounds", "dirham", "dirhams", "euro", "euros", "dollar", "dollars"}
WL = {"date": "0123456789/-. ", "amount_digits": "0123456789,.- "}
PARAMS = dict(pad=6, scale=1.0)                                                # handwriting is big: Tesseract reads it best at its own size (see cheques.OCR_PARAMS)


def _h(w): return w["y1"] - w["y0"]
def _cy(w): return (w["y0"] + w["y1"]) / 2
def _bbox(ws): return [min(w["x0"] for w in ws), min(w["y0"] for w in ws), max(w["x1"] for w in ws), max(w["y1"] for w in ws)]


def _numeric_word(w: dict) -> bool:
    """A word made of digits and separators, a junk character inside allowed ("{00" is a "100" read badly)."""
    keep = re.sub(r"[^\d.,:\-—–_/]", "", w["text"].replace("O", "0"))
    return bool(re.search(r"\d", keep)) and len(keep) >= 0.5 * len(w["text"])


def _clean(t: str) -> str:
    return fold(t).strip(".,;:-—–_'\"()[]|")


def _is_number_word(t: str) -> bool:
    return _clean(t) in _NUMBER_WORDS


def _clip(box, W: float, H: float) -> list[float]:
    x0, y0, x1, y1 = box
    return [round(float(max(0, x0)), 1), round(float(max(0, y0)), 1), round(float(min(W, x1)), 1), round(float(min(H, y1)), 1)]


_PRINTED = {"payez", "contre", "ce", "cheque", "non", "endossable", "sauf", "au", "profit", "ordre", "de", "pay", "to", "the", "order", "of", "only", "not", "negotiable", "payable", "en",
            "france", "votre", "numero", "compte", "nom", "prenom", "adresse", "a", "l'ordre", "le", "la", "du", "remplissez", "et", "un"}


def _printed_phrase(ws: list[dict]) -> bool:
    """'contre ce chèque non endossable sauf au profit' is the printed sentence of the cheque, not a name written after a label."""
    toks = [_clean(w["text"]) for w in ws if _clean(w["text"])]
    return bool(toks) and sum(t in _PRINTED for t in toks) / len(toks) >= 0.6


def _inside(w: dict, b, pad: float = 4.0) -> bool:
    cx, cy = (w["x0"] + w["x1"]) / 2, _cy(w)
    return b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad


def locate_fields(words: list[dict], W: float, H: float, hints: dict | None = None) -> dict:
    """OCR words (sparse reading of the whole cheque) -> {field: {"box": [x0, y0, x1, y1], "candidate": text read on the page ('' if none), "conf": 0..1}}.
    `hints` = boxes the detector already found ({"amount_words": [x0, y0, x1, y1], "signature": ...}): they say where the OTHER fields are not."""
    hints = hints or {}
    out: dict[str, dict] = {}
    if not words:
        return out
    sure = [w for w in words if w["conf"] >= 60] or words
    lh = float(np.median([_h(w) for w in sure]))
    fl = _Flat(words)

    # ---- currency sign (the amount in figures sits right of it). Handwriting is sometimes read as a sign ("Only" -> "£0"): a sign counts only when numbers follow it
    num = _numeric_word
    syms = [w for w in words if SYMBOL_RE.search(w["text"]) and w["x0"] > 0.3 * W and _cy(w) < 0.8 * H]
    followed = lambda s_: [w for w in words if num(w) and s_["x1"] - 6 <= w["x0"] <= s_["x1"] + 0.15 * W and abs(_cy(w) - _cy(s_)) < 1.6 * max(_h(s_), _h(w))]
    syms = [s_ for s_ in syms if followed(s_)]
    sym = max(syms, key=lambda s_: sum(_h(w) for w in followed(s_))) if syms else None

    # ---- date
    labels = [w for w in words if _cy(w) < 0.55 * H and 3 <= len(_clean(w["text"])) <= 5 and difflib.SequenceMatcher(None, _clean(w["text"]), "date").ratio() >= 0.74]
    best = None
    for f in find_dates(fl.text):                                                                  # 22/01/2006 and also "1 decembre 2005", "December 1st, 2005"
        ws = fl.boxes(f.start, f.end)
        text = f.text
        if not ws or not parse_date(text)[1]:
            continue
        year4 = f.kind == "named" or bool(re.search(r"\d{4}$", text))
        if not year4 and "/" not in text and "." not in text and not labels:       # 01-02-03 is a sort code, not a date
            continue
        b = _bbox(ws)
        near = any(abs(_cy(l) - (b[1] + b[3]) / 2) < 2.5 * _h(l) + 12 and 0 <= b[0] - l["x1"] < 0.25 * W for l in labels)
        score = 2 * near + (_cy(ws[0]) < 0.5 * H) + year4 + np.mean([w["conf"] for w in ws]) / 200
        if best is None or score > best[0]:
            best = (score, b, text, float(np.mean([w["conf"] for w in ws])) / 100)
    if best is None:                                                                               # "Le 27052074": the dots of a handwritten date are often not read at all
        for l in labels + [w for w in words if _clean(w["text"]) in ("le", "du", "date") and _cy(w) < 0.8 * H]:
            nxt = sorted([w for w in words if w["x0"] >= l["x1"] - 2 and w["x0"] - l["x1"] < 0.3 * W and abs(_cy(w) - _cy(l)) < 1.2 * max(_h(l), _h(w))], key=lambda w: w["x0"])
            for n in range(1, min(4, len(nxt)) + 1):
                grp, txt = nxt[:n], "".join(w["text"] for w in nxt[:n])
                if re.fullmatch(r"[\d.,/\-\s]{6,12}", txt) and 6 <= sum(c.isdigit() for c in txt) <= 8 and parse_date(txt)[1]:
                    b = _bbox(grp)
                    best = (1.0, b, txt, float(np.mean([w["conf"] for w in grp])) / 100)
                    break
            if best:
                break
    if best:
        out["date"] = {"box": _clip([best[1][0] - 4, best[1][1] - 3, best[1][2] + 4, best[1][3] + 3], W, H), "candidate": best[2], "conf": round(max(best[3], 0.5), 2)}
    elif labels:
        l = max(labels, key=lambda w: w["conf"])
        out["date"] = {"box": _clip([l["x1"] + 2, l["y0"] - 2.2 * _h(l), l["x1"] + 0.28 * W, l["y1"] + 0.6 * _h(l)], W, H), "candidate": "", "conf": 0.3}

    # ---- amount in figures
    group = []
    if sym is not None:
        hs = max(_h(sym), 1.0)
        group = sorted([w for w in words if num(w) and w["x0"] >= sym["x1"] - 6 and abs(_cy(w) - _cy(sym)) < 1.6 * max(hs, _h(w))], key=lambda w: w["x0"])
        if group:
            b = _bbox(group)
            left = sym["x1"] + 2 if b[0] - sym["x1"] < 0.12 * W else b[0] - 6                  # the box starts right of the sign: a first digit read as junk is still inside
            out["amount_digits"] = {"box": _clip([left, b[1] - 4, b[2] + 8, b[3] + 4], W, H), "candidate": "".join(re.sub(r"[^\d.,:\-—–_/]", "", w["text"]) for w in group), "conf": round(float(np.mean([w["conf"] for w in group])) / 100, 2)}
        else:
            out["amount_digits"] = {"box": _clip([sym["x1"] + 2, _cy(sym) - 1.3 * hs, sym["x1"] + 0.35 * W, _cy(sym) + 1.3 * hs], W, H), "candidate": "", "conf": 0.3}
    else:
        date_ws = {(w["x0"], w["y0"]) for f in find_dates(fl.text) for w in fl.boxes(f.start, f.end)}        # (words are copied by _lines: compare positions, not objects)
        tall = [w for w in words if num(w) and (w["x0"], w["y0"]) not in date_ws and w["x0"] > 0.4 * W and _cy(w) < 0.7 * H and _h(w) >= 1.5 * lh]
        if tall:
            ref = max(tall, key=lambda w: _h(w) * (w["x1"] - w["x0"]))
            g = sorted([w for w in tall if abs(_cy(w) - _cy(ref)) < 0.8 * _h(ref)], key=lambda w: w["x0"])
            i = g.index(ref)
            lo = hi = i                                                                            # only words that touch each other: a far-away number on the line is another thing
            while lo > 0 and g[lo]["x0"] - g[lo - 1]["x1"] < 2.5 * _h(ref):
                lo -= 1
            while hi < len(g) - 1 and g[hi + 1]["x0"] - g[hi]["x1"] < 2.5 * _h(ref):
                hi += 1
            g = g[lo:hi + 1]
            b = _bbox(g)
            out["amount_digits"] = {"box": _clip([b[0] - 6, b[1] - 4, b[2] + 8, b[3] + 4], W, H), "candidate": "".join(w["text"] for w in sorted(g, key=lambda w: w["x0"])), "conf": round(float(np.mean([w["conf"] for w in g])) / 100, 2)}

    # ---- amount in words: the line with the most number words
    best_line = None
    for ln in _lines(words):
        n = [w for w in ln["words"] if _is_number_word(w["text"]) and _clean(w["text"]) not in _FUNCTION]
        strong = len(n) == 1 and _clean(n[0]["text"]) in _STRONG and any(_clean(w["text"]) in _STRONG | _FUNCTION for w in ln["words"] if w is not n[0])   # "hundred ... and": the rest is handwriting
        if (len(n) >= 2 or strong) and (best_line is None or len(n) > best_line[0]):
            best_line = (len(n), ln["words"], n)
    right_limit = 0.82 * W
    if sym is not None:
        right_limit = min(right_limit, sym["x0"] - 4)
    if "amount_digits" in out:
        right_limit = min(right_limit, out["amount_digits"]["box"][0] - 8)
    aw, aw_text_top, aw_h = None, 0.0, 0.0
    if best_line:
        _, lw, n = best_line
        cy = float(np.mean([_cy(w) for w in n]))
        h = max(_h(w) for w in n)
        band = [w for w in words if abs(_cy(w) - cy) < 0.6 * h]
        x0 = min(w["x0"] for w in band)
        x1 = max(max(w["x1"] for w in band) + 0.08 * W, 0)
        x1 = min(x1, right_limit) if right_limit > x0 + 3 * h else x1
        aw = [x0 - 4, min(w["y0"] for w in n) - 0.25 * h, x1, max(w["y1"] for w in n) + 0.25 * h]
        aw_text_top, aw_h = min(w["y0"] for w in n), h
        out["amount_words"] = {"box": _clip(aw, W, H), "candidate": " ".join(w["text"] for w in sorted(band, key=lambda w: w["x0"]) if x0 - 1 <= w["x0"] <= x1), "conf": round(float(np.mean([w["conf"] for w in n])) / 100, 2)}

    # ---- payee: right of the printed label (Pay / Payez à l'ordre de), else the line above the amount in words, else a line of handwriting below it
    done = False
    lab = [w for w in words if _clean(w["text"]) in ("pay", "payez", "payer", "ordre", "order") and 0.08 * H < _cy(w) < 0.75 * H and w["x0"] < 0.5 * W
           and (aw is None or _cy(w) < aw_text_top - 0.3 * aw_h)]
    if lab:
        l = min(lab, key=lambda w: (w["y0"], w["x0"]))
        end = l["x1"]
        for w in sorted([w for w in words if w["x0"] >= l["x1"] and abs(_cy(w) - _cy(l)) < 0.8 * _h(l)], key=lambda w: w["x0"]):       # "Payez à l'ordre de": the label has several words
            if w["x0"] - end < 1.2 * _h(l) and _clean(w["text"]) in ("a", "l'ordre", "lordre", "ordre", "de", "d", "of", "the", "to", "order", "pay", "to", "the", "ordre"):
                end = w["x1"]
        rest = [w for w in words if w["x0"] >= end - 2 and abs(_cy(w) - _cy(l)) < 0.8 * _h(l) + 4 and w["x0"] < right_limit]
        if not (rest and _printed_phrase(rest)):                                           # "Payez contre ce chèque non endossable ...": the printed sentence, not a label + a name
            hh = max([_h(l)] + [_h(w) for w in rest])
            if rest:
                b = _bbox(rest)
                box = [end + 2, min(b[1], l["y0"]) - 0.3 * hh, right_limit, max(b[3], l["y1"]) + 0.3 * hh]
            else:
                box = [end + 2, l["y0"] - 1.2 * hh, right_limit, l["y1"] + 0.8 * hh]
            out["payee"] = {"box": _clip(box, W, H), "candidate": " ".join(w["text"] for w in sorted(rest, key=lambda w: w["x0"])), "conf": 0.4}
            done = True
    if not done and aw is not None:
        h = aw_h                                                                             # height of the handwriting itself, not of the padded box
        above = [w for w in words if aw[0] - 40 <= w["x0"] and w["x1"] <= aw[2] + 20 and aw_text_top - 1.3 * h < _cy(w) < aw_text_top + 0.1 * h
                 and not (w["conf"] >= 85 and len(_clean(w["text"])) >= 4 and w["text"].isupper())]
        if above:
            b = _bbox(above)
            top, bottom = max(b[1] - 0.25 * h, aw_text_top - 1.3 * h), min(b[3] + 0.25 * h, aw_text_top + 0.2 * h)
            out["payee"] = {"box": _clip([min(b[0], aw[0]) - 4, top, aw[2], bottom], W, H), "candidate": " ".join(w["text"] for w in sorted(above, key=lambda w: w["x0"])), "conf": 0.4}
            done = True
    if not done:                                                                             # handwriting = runs of low-confidence words (what it looks like to Tesseract)
        from ..ocr.handwriting import _segments
        aw_ref = aw or hints.get("amount_words")
        known = [b for b in (hints.get(k) for k in ("amount_words", "amount_digits", "signature", "date", "payee")) if b] + ([aw] if aw else []) + [out[k]["box"] for k in ("date", "amount_digits") if k in out]
        low = [w for w in words if w["conf"] < 70 and _h(w) >= 0.8 * lh and not any(_inside(w, b) for b in known)]
        segs = [sg for sg in _segments(low, lh) if min(w["x0"] for w in sg) < 0.45 * W and max(w["x1"] for w in sg) - min(w["x0"] for w in sg) >= 3 * lh]
        if aw_ref is not None:                                                               # the line just below the amount in words
            segs = [sg for sg in segs if min(w["y0"] for w in sg) > aw_ref[3] - 0.4 * (aw_ref[3] - aw_ref[1])]
            pick = min(segs, key=lambda sg: min(w["y0"] for w in sg)) if segs else None
        else:
            pick = max(segs, key=lambda sg: max(w["x1"] for w in sg) - min(w["x0"] for w in sg)) if segs else None
        if pick:
            b = _bbox(pick)
            hh = b[3] - b[1]
            out["payee"] = {"box": _clip([b[0] - 6, b[1] - 0.2 * hh, b[2] + 0.08 * W, b[3] + 0.2 * hh], W, H), "candidate": " ".join(w["text"] for w in sorted(pick, key=lambda w: w["x0"])), "conf": 0.35}
            done = True
    if not done and aw is not None:
        h = aw_h
        out["payee"] = {"box": _clip([aw[0] - 4, aw_text_top - 1.2 * h, aw[2], aw_text_top + 0.15 * h], W, H), "candidate": "", "conf": 0.3}
    return out


_PAYEE_LABEL = re.compile(r"^\s*(?:(?:pay(?:ez)?|payer|à|[àa]\s+l.?ordre\s+de|l.?ordre\s+de|order\s+of|to\s+the\s+order\s+of|to|the|ordre\s+de)\b[\s:.\-_]*)+", re.I)


_LOOKALIKE = str.maketrans({"l": "1", "I": "1", "|": "1", "!": "1", "O": "0", "o": "0", "Q": "0", "S": "5", "B": "8"})


def numeric_text(text: str, keep: str = ".,-/ ") -> str:
    """Handwritten digits read as letters ('l00' = 100, 'O5' = 05): lookalikes mapped, everything that is not a digit or a separator dropped.
    The whitelist of Tesseract is NOT used for handwriting: it drops a '1' it sees as 'l' instead of mapping it (tested: '100-00' became '00-00')."""
    t = (text or "").replace("—", "-").replace("–", "-").translate(_LOOKALIKE)
    t = "".join(c for c in t if c.isdigit() or c in keep)
    return re.sub(r"\s+", " ", t).strip(" .,:;/-")


def amountish(text: str) -> bool:
    """Digits, separators, a currency sign or code - not words: a reading such as '2 May 2025' in the amount box is another field's text."""
    t = re.sub(r"MAD|DHS?|EUR|GBP|USD|[£$€]", "", text or "")
    t = t.translate(_LOOKALIKE)
    return sum(1 for c in t if c.isalpha()) <= 1


def looks_like_name(text: str) -> bool:
    """A payee is a name: letters, a few separators. 'Que Nationwide hundred = and' is the amount line + the bank name read as one, not a payee."""
    from .cheques import payee_ok
    return payee_ok(text) and not any(_is_number_word(t) for t in text.split())


def _enough_number_words(text: str) -> bool:
    """Two number words at least ('One Hundred'): 'hundred' alone is a fragment, not an amount."""
    return sum(1 for t in text.split() if _is_number_word(t) and _clean(t) not in _FUNCTION) >= 2


def clean_payee(text: str) -> str:
    """'PAY Wikimedia Foundation' -> 'Wikimedia Foundation' (the printed label sometimes falls inside the crop)."""
    return _PAYEE_LABEL.sub("", (text or "").strip(" |_;:,.-—")).strip(" |_;:,.-—\"'“”")


def read_cheque_layout(img, lang: str | None = None, need: set | frozenset | None = None, hint_amount: float | None = None, hints: dict | None = None) -> dict:
    """PIL image -> {field: {"value", "raw_text", "box_confidence", "bbox", "n_boxes", "source": "layout"}} for the fields in `need` (all four when None).
    `hint_amount` = the amount in words, as a number, when it is known (the figures must agree with it: a cross-check between the two readings)."""
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    k = float(np.clip(1600.0 / max(W, 1), 1.0, 2.5))
    big = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1.05 else gray
    words = run_ocr(big, scale=k if k > 1.05 else 1.0, lang=lang, psm=11)
    loc = locate_fields(words, float(W), float(H), hints)
    rgb = img.convert("RGB")
    out = {}
    for name in ("date", "amount_digits", "amount_words", "payee"):
        L = loc.get(name)
        if L is None or (need is not None and name not in need):
            continue
        box, cand = L["box"], L["candidate"]
        if box[2] - box[0] < 8 or box[3] - box[1] < 6:
            continue
        readings = None
        if name == "date":                                                                           # readings of the box vote for the date; the page reading votes too
            from .date_reader import read_date_box
            vote = read_date_box(rgb, box, lang, extra={"page": cand} if cand else None)
            raw, readings = vote["text"], vote["readings"]
            if not vote["date"] and cand:
                raw = cand
            if raw.strip():
                out[name] = {"value": raw, "raw_text": raw, "box_confidence": round(float(L["conf"]), 2), "bbox": box, "n_boxes": 1, "source": "layout", "readings": readings}
            continue
        reading = ocr_box(rgb, box, psm=7, lang=lang, field=name, **PARAMS)                     # no whitelist (see numeric_text)
        if name == "amount_digits":
            raws = [reading, ocr_box(rgb, box, psm=7, lang=lang, whitelist=WL["amount_digits"], field=name, **PARAMS), cand]
            alts = [numeric_text(r) for r in raws if r and amountish(r)]                             # words are not an amount ("2 May 2025")
            alts = [t for t in alts if t and (parse_amount(t) or 0) > 0]
            if not alts:                                                                               # located, but only zeros / nothing readable: say so, do not invent 0.00
                out[name] = {"value": "", "raw_text": "", "box_confidence": round(float(L["conf"]), 2), "bbox": box, "n_boxes": 1, "source": "layout", "unread": True}
                continue
            agree = [t for t in alts if hint_amount is not None and abs(parse_amount(t) - hint_amount) < 0.005]
            raw = (agree or sorted(alts, key=lambda t: -sum(c.isdigit() for c in t)) or [cand or reading])[0]          # agrees with the words, else the reading with most digits
        elif name == "amount_words":
            ok = lambda t: bool(t and (words_to_number(t) or english_words_to_number(t)))
            raw = (reading if ok(reading) and _enough_number_words(reading) else cand if ok(cand) and _enough_number_words(cand) else "").strip(" |_;:,—-")
            if not raw:                                                                                # a fragment ("hundred ... and"): handwriting that only the handwriting model reads
                out[name] = {"value": "", "raw_text": "", "box_confidence": round(float(L["conf"]), 2), "bbox": box, "n_boxes": 1, "source": "layout", "unread": True}
                continue
        else:
            raw = clean_payee(reading or cand)
            if not looks_like_name(raw):                                                              # located, but nothing readable (or not a name): cursive that only a handwriting model reads
                out[name] = {"value": "", "raw_text": "", "box_confidence": round(float(L["conf"]), 2), "bbox": box, "n_boxes": 1, "source": "layout", "unread": True}
                continue
        if raw.strip():
            out[name] = {"value": raw, "raw_text": raw, "box_confidence": round(float(L["conf"]), 2), "bbox": box, "n_boxes": 1, "source": "layout"}
    return out
