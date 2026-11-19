"""Read a loan contract from its TEXT (English or French) - the safety net behind the YOLO detector of `contracts.py`.

The detector is trained on French contracts of one layout: on an English contract, or on another layout, it finds only a few boxes and the page
came back with "not found: amount, interest_rate, duration ..." and "no signature". Here the whole page is read once by the OCR and the terms are
found by what the sentence says ("the sum of 20,000 MAD", "for a period of 6 months", "The loan is interest-free", "Le prêteur : ..."), and a
signature is an area with handwriting-like ink next to the word "Signature" (printed words and ruled lines are masked out).

    from documind.inference.contract_text import read_contract_text
    res = read_contract_text(pil_image)       # {"fields": {class: {"value", "raw_text", "box_confidence", "bbox", ...}}, "not_applicable": {...}}

`contracts.extract_contract_bytes` calls it after the detector and uses it for every field the detector missed or read badly. A term the contract does
not contain (a single-repayment loan has no monthly payment and no IBAN) is reported in `not_applicable`, so it is not flagged as "not found".
"""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

from ..ocr import reconstruct, run_ocr
from .postprocess import _lines

_MONTH = (r"(?:jan(?:uary|vier)?|feb(?:ruary)?|f[ée]v(?:rier)?|march?|mars|apr(?:il)?|avr(?:il)?|may|mai|june?|juin|july?|juil(?:let)?|aug(?:ust)?|ao[uû]t|"
          r"sept?(?:ember|embre)?|oct(?:ober|obre)?|nov(?:ember|embre)?|dec(?:ember)?|d[ée]c(?:embre)?)")
DATE_ANY = (rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}\s*[/.\-]\s*\d{{1,2}}\s*[/.\-]\s*\d{{2,4}}|\d{{1,2}}(?:st|nd|rd|th|er)?\s+{_MONTH}\.?,?\s+\d{{4}}|{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})")
CUR = r"(MAD|DHS?|EUR|USD|€|\$|dirhams?|euros?)"
NUM = r"(\d[\d\s\u00a0.,]*\d|\d)"
_PERSON_END = r"(?=\s*(?:,|;|\.|(?i:holder|titulaire|domicili|demeurant|n[ée]e?\b|born|residing|cin\b|id\b|c\.i\.n)))"
_PERSON = r"((?:M\.|Mr\.?|Mrs\.?|Mme\.?|Mlle\.?|Monsieur|Madame|Dr\.?)?\s*[A-ZÉÈÀÂÎÔÛÇ][\w'’.\-]*(?:\s+(?:el|al|ben|bin|de|du|le|la|d'|van)?\s*[A-ZÉÈÀÂÎÔÛÇ][\w'’.\-]*){0,4})"


def fold(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (t or "").lower()) if not unicodedata.combining(c))


# ---------------------------------------------------------------------------------------------- numbers
def to_float(text: str) -> float | None:
    """'20,000' / '20 000,50' / '100.000,00' / '1,250.50' / '450' -> float. A comma or a dot followed by exactly three digits is a thousands mark."""
    c = re.sub(r"[\s\u00a0]", "", text or "").strip(".,")
    if not re.fullmatch(r"[\d.,]+", c or "x"):
        return None
    if "," in c and "." in c:
        dec = "," if c.rfind(",") > c.rfind(".") else "."
        c = c.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif c.count(",") > 1 or c.count(".") > 1 or re.fullmatch(r"\d{1,3}[,.]\d{3}", c):
        c = c.replace(",", "").replace(".", "")
    else:
        c = c.replace(",", ".")
    try:
        return float(c)
    except ValueError:
        return None


_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_EN_CUR = {"dirham": "MAD", "dirhams": "MAD", "mad": "MAD", "euro": "EUR", "euros": "EUR", "eur": "EUR", "dollar": "USD", "dollars": "USD", "usd": "USD",
           "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "gbp": "GBP"}
_SKIP = {"and", "moroccan", "us", "american", "only", "of", "the", "cents", "cent", "centimes", "pence", "penny", "british"}


def english_words_to_number(text: str) -> tuple[int, int, str] | None:
    """'twenty thousand Moroccan Dirhams' -> (20000, 0, 'MAD'); None if a word is not a number or a currency. Same shape as cheques.words_to_number."""
    toks = [t for t in re.findall(r"[a-z]+", (text or "").lower().replace("-", " "))]
    if not toks:
        return None
    currency, total, cur, cents_t, seen_cur = "", 0, 0, [], False
    for t in toks:
        if t in _EN_CUR:
            currency, seen_cur = _EN_CUR[t], True
        elif t in _SKIP:
            continue
        elif seen_cur:
            cents_t.append(t)
        elif t in _ONES:
            cur += _ONES[t]
        elif t in _TENS:
            cur += _TENS[t]
        elif t == "hundred":
            cur = max(cur, 1) * 100
        elif t in _SCALE:
            total += max(cur, 1) * _SCALE[t]
            cur = 0
        else:
            return None
    cents = 0
    if cents_t:
        c = english_words_to_number(" ".join(cents_t))
        if c is None or c[0] > 99:
            return None
        cents = c[0]
    return total + cur, cents, currency


# ---------------------------------------------------------------------------------------------- text with word positions
class _Flat:
    """The page text in reading order, each word remembering its place in the string (to find the box of a match)."""

    def __init__(self, words: list[dict]) -> None:
        self.parts: list[tuple[int, int, dict]] = []
        pos, out = 0, []
        for ln in _lines(words):
            for w in ln["words"]:
                out.append(w["text"])
                self.parts.append((pos, pos + len(w["text"]), w))
                pos += len(w["text"]) + 1
        self.text = " ".join(out)

    def boxes(self, start: int, end: int) -> list[dict]:
        return [w for a, b, w in self.parts if a < end and b > start]

    def field(self, start: int, end: int, raw: str, value: str | None = None) -> dict | None:
        ws = self.boxes(start, end)
        if not ws or not raw.strip():
            return None
        conf = float(np.mean([w.get("conf", 0.0) for w in ws])) / 100.0
        return {"value": value if value is not None else raw, "raw_text": raw, "box_confidence": round(max(0.0, min(conf, 1.0)), 2),
                "bbox": [round(min(w["x0"] for w in ws), 1), round(min(w["y0"] for w in ws), 1), round(max(w["x1"] for w in ws), 1), round(max(w["y1"] for w in ws), 1)],
                "n_boxes": 1, "source": "text"}


def _money_fmt(value: float, cur: str) -> str:
    code = {"dh": "MAD", "dhs": "MAD", "dirham": "MAD", "dirhams": "MAD", "euro": "EUR", "euros": "EUR", "€": "EUR", "$": "USD"}.get(fold(cur), (cur or "").upper())
    return f"{value:.2f} {code}".strip()


# ---------------------------------------------------------------------------------------------- the terms
def parse_terms(words: list[dict], width: float, height: float) -> tuple[dict, set]:
    """-> ({class: field}, not_applicable). English and French sentences of a loan contract."""
    fl = _Flat(words)
    T, F = fl.text, {}
    I = re.I

    def put(name: str, m: re.Match, group: int, raw: str | None = None, value: str | None = None) -> None:
        if name in F:
            return
        raw = (raw if raw is not None else m.group(group)).strip(" ,.;:()")
        f = fl.field(m.start(group), m.end(group), raw, value)
        if f:
            F[name] = f

    if m := re.search(r"(?:loan|credit)\s+agreement|contrat\s+de\s+(?:pr[êe]t|cr[ée]dit)(?:\s+(?:personnel|immobilier|automobile|auto|consommation))?", T, I):
        put("title", m, 0)
        put("title", m, 0)
    if m := re.search(r"\b(?:contract|contrat|agreement)\s*(?:n[°ºo]\.?|no\.?|number|num[ée]ro|#)\s*:?\s*([A-Z0-9][A-Z0-9\-/_.]{3,})", T, I):
        put("contract_number", m, 1)

    top, bottom = [], []
    for m in re.finditer(rf"\bdate\s*:?\s*({DATE_ANY})", T, I):
        ws = fl.boxes(m.start(1), m.end(1))
        cy = np.mean([(w["y0"] + w["y1"]) / 2 for w in ws]) if ws else 0
        (top if cy < 0.3 * height else bottom if cy > 0.7 * height else []).append(m)
    if top:
        put("contract_date", top[0], 1)
    if bottom:
        put("signing_date", bottom[0], 1)
    elif m := re.search(rf"fait\s+[àa]\s+[\w\- ]+?,?\s+le\s+({DATE_ANY})", T, I):
        put("signing_date", m, 1)
    if m := re.search(r"fait\s+[àa]\s+([A-ZÉ][\w\-]+(?:\s+[A-ZÉ][\w\-]+)?)\s*,?\s+le\b", T):
        put("place", m, 1)

    # parties: "The Lender: Ahmed El Mansouri, holder of ID card n° ..." / "Le prêteur : M. ..., titulaire de la CIN ..."
    lender = re.search(rf"\b(?i:(?:the\s+)?lender)\s*:\s*{_PERSON}{_PERSON_END}|\b(?i:le\s+pr[êe]teur)\s*:?\s*{_PERSON}{_PERSON_END}", T)         # names: capitals matter
    borrower = re.search(rf"\b(?i:(?:the\s+)?borrower)\s*:\s*{_PERSON}{_PERSON_END}|\b(?i:l.?\s*emprunteur)\s*:?\s*{_PERSON}{_PERSON_END}", T)
    for name, m in (("lender_name", lender), ("borrower_name", borrower)):
        if m:
            g = 1 if m.group(1) else 2
            put(name, m, g)
    if "lender_name" not in F or "borrower_name" not in F:                                     # fall back on the signature block: "Name: ... Name: ..."
        names = list(re.finditer(r"\b(?i:name|nom)\s*:\s*([A-ZÉ][\w'’.\-]+(?:\s+(?:el|al|ben|de)?\s*[A-ZÉ][\w'’.\-]+){0,3})", T))
        for name, m in zip(("lender_name", "borrower_name"), names[:2]):
            put(name, m, 1)
    if borrower:
        seg_start = borrower.start()
        seg = T[seg_start: seg_start + 400]
        if m := re.search(r"(?:id\s*card|cin|c\.i\.n\.?|carte\s+nationale(?:\s+d.identit[ée])?|passport|passeport)\s*(?:n[°ºo]\.?|no\.?|number|:)?\s*([A-Z]{1,2}\s?\d{4,8}|\d{5,})", seg, I):
            f = fl.field(seg_start + m.start(1), seg_start + m.end(1), m.group(1).replace(" ", ""))
            if f:
                F["borrower_id"] = f
        if m := re.search(r"(?:domicil\w+|demeurant|residing|resident)\s+(?:at|[àa]|in)\s+(.{5,140}?)(?=,\s*(?:hereinafter|ci-apr[èe]s|d[ée]nomm|on the|here\s*after|d['’])|\.\s|[;:]|\bArticle\b|$)", seg, I):
            f = fl.field(seg_start + m.start(1), seg_start + m.end(1), m.group(1).strip(" ,"))
            if f:
                F["borrower_address"] = f

    # amount in figures and in words
    am = (re.search(rf"\b(?:sum|amount|principal|somme|montant|capital)\s+(?:of|de|d['’])\s*{NUM}\s*{CUR}?", T, I)
          or re.search(rf"{NUM}\s*{CUR}\b", T, I))
    if am:
        val = to_float(am.group(1))
        if val is not None:
            cur = am.group(2) if am.lastindex and am.lastindex >= 2 and am.group(2) else ""
            f = fl.field(am.start(1), am.end(am.lastindex if cur else 1), _money_fmt(val, cur or ""), _money_fmt(val, cur or ""))
            if f:
                F["amount"] = f
        if wm := re.match(r"\s*\(([^()]{5,140})\)", T[am.end():]):
            f = fl.field(am.end() + wm.start(1), am.end() + wm.end(1), wm.group(1).strip())
            if f:
                F["amount_words"] = f

    # duration, interest, instalments
    if m := re.search(r"\b(?:period|term|duration|dur[ée]e)\s+(?:of\s+|de\s+)?(\d+)\s*(months?|years?|yrs?|mois|ans?|ann[ée]es?)", T, I) or re.search(r"\b(?:over|pendant|sur|for)\s+(\d+)\s*(months?|years?|mois|ans?)", T, I):
        put("duration", m, 0, raw=f"{m.group(1)} {m.group(2)}")
    if m := re.search(r"\binterest[- ]free\b|\bsans\s+int[ée]r[êe]ts?\b|\bno\s+interest\b|\bint[ée]r[êe]ts?\s+(?:de\s+)?0\s*%|\binterest\s+rate\s+(?:of\s+|is\s+)?0\s*%", T, I):
        put("interest_rate", m, 0, raw="0 %")
    else:
        for k in re.finditer(r"\b(?:interest|int[ée]r[êe]ts?|taux|rate|apr|taeg)\b", T, I):
            if m := re.search(r"(\d+(?:[.,]\d+)?)\s*%", T[k.end(): k.end() + 90]):
                a = k.end() + m.start(1)
                if f := fl.field(a, k.end() + m.end(), m.group(0).strip()):
                    F["interest_rate"] = f
                break
    if m := re.search(rf"(?:monthly\s+(?:payment|installment|instalment|repayment)s?|mensualit[ée]s?|[ée]ch[ée]ances?\s+mensuelles?)[^\d]{{0,30}}?{NUM}\s*{CUR}", T, I):
        if (v := to_float(m.group(1))) is not None:
            put("monthly_payment", m, 1, raw=_money_fmt(v, m.group(2)))
    if m := re.search(rf"first\s+(?:payment|installment|instalment)[^.\d]{{0,50}}({DATE_ANY})|premi[èe]re\s+(?:mensualit[ée]|[ée]ch[ée]ance)[^.\d]{{0,50}}({DATE_ANY})", T, I):
        put("first_payment_date", m, 1 if m.group(1) else 2)
    if m := re.search(r"\biban\s*:?\s*([A-Z]{2}\d{2}(?:\s[A-Z0-9]{4}){3,7}(?:\s[A-Z0-9]{1,4})?|[A-Z]{2}\d{2}[A-Z0-9]{10,30})", T, I):
        put("iban", m, 1)

    na = set()
    if not re.search(r"monthly|installment|instalment|mensualit|[ée]ch[ée]ances?\s+mensuelles?", T, I):
        na.add("monthly_payment")
    if not re.search(r"\biban\b|\brib\b|bank\s+(?:transfer|account)|virement|compte\s+bancaire", T, I):
        na.add("iban")
    return F, na


# ---------------------------------------------------------------------------------------------- signatures
def find_signatures(gray: np.ndarray, words: list[dict]) -> dict:
    """A signature = handwriting-like ink next to the word "Signature". Printed words and ruled lines are masked out first.
    Which party signs is taken from the closest heading above (THE LENDER / LE PRÊTEUR / THE BORROWER / L'EMPRUNTEUR), else left = lender, right = borrower."""
    H, W = gray.shape[:2]
    if not words:
        return {}
    lh = float(np.median([w["y1"] - w["y0"] for w in words]))
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rules = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(0.08 * W)), 1)))
    bw[cv2.dilate(rules, np.ones((3, 1), np.uint8)) > 0] = 0                                   # underline of the signature
    printed = [w for w in words if w.get("conf", 0) >= 70 and len(re.sub(r"[^A-Za-zÀ-ÿ]", "", w["text"])) >= 3]
    labels = [w for w in words if re.fullmatch(r"signature|sign[ée]e?|signatures", fold(w["text"]).strip(":.- "))]
    heads = [(w, "lender" if fold(w["text"]).strip(":.,") in ("lender", "preteur") else "borrower") for w in words
             if fold(w["text"]).strip(":.,") in ("lender", "preteur", "borrower", "emprunteur") and w["y0"] > 0.45 * H]
    out: dict[str, dict] = {}
    for lab in labels:
        cx = (lab["x0"] + lab["x1"]) / 2
        left = cx < W / 2
        x0, x1 = int(lab["x1"] + 0.3 * lh), int(min(lab["x1"] + 0.38 * W, (W * 0.49) if left else (W * 0.99)))
        y0, y1 = int(max(0, lab["y0"] - 1.3 * lh)), int(min(H, lab["y1"] + 1.8 * lh))
        reg = bw[y0:y1, x0:x1].copy()
        for w in printed:                                                                     # printed lines above / below (Name:, Date:); NOT the words on the label's own line,
            if abs((w["y0"] + w["y1"]) / 2 - (lab["y0"] + lab["y1"]) / 2) > 0.7 * lh:          # where the OCR may have "read" the handwriting as a word
                reg[max(0, int(w["y0"]) - 1 - y0): max(0, int(w["y1"]) + 2 - y0), max(0, int(w["x0"]) - 1 - x0): max(0, int(w["x1"]) + 2 - x0)] = 0
        n = int(np.count_nonzero(reg))
        if reg.size == 0 or n < 0.8 * lh * lh:
            continue
        ys, xs = np.nonzero(reg)
        if xs.max() - xs.min() < 2.5 * lh:
            continue
        above = [(abs((h["x0"] + h["x1"]) / 2 - cx), who) for h, who in heads if h["y1"] <= lab["y0"] and lab["y0"] - h["y1"] < 0.25 * H]
        who = min(above)[1] if above else ("lender" if left else "borrower")
        key = f"signature_{who}"
        box = [round(float(x0 + xs.min()), 1), round(float(y0 + ys.min()), 1), round(float(x0 + xs.max()), 1), round(float(y0 + ys.max()), 1)]
        conf = round(min(0.95, 0.5 + n / (30.0 * lh * lh)), 2)
        if key not in out or conf > out[key]["box_confidence"]:
            out[key] = {"value": "signed", "raw_text": "", "box_confidence": conf, "bbox": box, "n_boxes": 1, "source": "text"}
    return out


# ---------------------------------------------------------------------------------------------- entry point
def read_contract_text(img, lang: str | None = None) -> dict:
    """PIL image -> {"fields": {class: field}, "not_applicable": set, "n_words": int}. Every field has "source": "text"."""
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    k = float(np.clip(1800.0 / max(H, W), 1.0, 2.5))                                          # small picture: enlarge, Tesseract wants ~30 px letters
    big = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1.05 else gray
    words = run_ocr(big, scale=k if k > 1.05 else 1.0, lang=lang, psm=6, handwriting=True)
    if words and reconstruct.mode() == "on":
        # Stage B (opt-in: DOCUMIND_RECONSTRUCT=on, after compare_reconstruct shows it wins) (same as the invoice): the doubtful chunks are re-read by the OCR model (pretrained or yours) and put in canonical spelling BEFORE the
        # terms are searched in the sentences. Nothing is trained on these words (the terms come from patterns), so no retraining is needed.
        words = reconstruct.reconstruct_words(big, words, k if k > 1.05 else 1.0)
    if not words:
        return {"fields": {}, "not_applicable": set(), "n_words": 0}
    fields, na = parse_terms(words, float(W), float(H))
    fields.update(find_signatures(gray, words))
    return {"fields": fields, "not_applicable": na, "n_words": len(words), "reconstructed_words": sum(1 for w in words if w.get("refined"))}
