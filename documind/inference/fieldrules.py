"""Fields the detector missed, found from the printed LABELS (French and English).

The YOLO detector learned the FATURA templates (English). On an invoice laid out differently it finds only some fields. Almost every invoice, whatever its
layout or language, prints a label next to a value ("Date d'échéance : 18/10/2025", "Bill To", "Net à payer"). This module reads those labels in the page's words
and adds a field for every class the detector did NOT find. It never overrides a detector box and it never invents: no label and no pattern = no field.

    fill_missing(fields, words, page, classes) -> the extra fields  {"label", "text", "confidence": 0.5, "bbox", "page", "source": "rules"}

Covered: DATE, DUE_DATE, NUMBER, PO_NUMBER, TITLE, SELLER_NAME, SELLER_ADDRESS, SELLER_EMAIL, SELLER_SITE, BUYER, SEND_TO, TOTAL_WORDS, AMOUNT_DUE, DISCOUNT,
CONDITIONS, NOTE, PAYMENT_DETAILS, GSTIN / GSTIN_SELLER / GSTIN_BUYER (also ICE, IF, RC, SIRET, TVA numbers). The amounts TOTAL, SUB_TOTAL and TAX are
left to the rules of documind.inference.postprocess, which also check them against the line items.
"""
from __future__ import annotations

import re
import unicodedata

_DATE = (r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}[ \-./]+[a-z]{3,9}\.?[ \-./,]+\d{2,4}|[a-z]{3,9}\.? \d{1,2},? \d{4}")
_MONEY = r"[-(]?\d[\d  .,]*\d(?:\s?(?:%|[a-z]{1,4}\b))?"
_ID = r"[a-z0-9][a-z0-9/_\-.]{2,}"
_STOP_MONTH = ("jan", "feb", "fev", "mar", "apr", "avr", "may", "mai", "jun", "jui", "aug", "aou", "sep", "oct", "nov", "dec")


def _fch(c: str) -> str:
    """One character folded to lower-case ASCII, always ONE character (so a position in the folded text is a position in the original)."""
    if c in "’‘`´":
        return "'"
    if c in "°º":
        return "o"
    d = unicodedata.normalize("NFKD", c.lower())
    return next((x for x in d if x.isascii()), " ")


def fold(s: str) -> str:
    return "".join(_fch(c) for c in s)


class _Line:
    def __init__(self, ln: dict):
        self.words = ln["words"]
        self.text = " ".join(w["text"] for w in self.words)
        self.f = fold(self.text)
        self.y0, self.y1, self.x0, self.x1 = ln["y0"], ln["y1"], ln["x0"], ln["x1"]
        self.spans, pos = [], 0
        for w in self.words:
            self.spans.append((pos, pos + len(w["text"])))
            pos += len(w["text"]) + 1

    @property
    def cell_words(self) -> list[dict]:
        """The words of the line's FIRST column (a wide gap starts another one: a name on the left, a date on the right of the same line)."""
        gap, out = 2.2 * self.h, [self.words[0]]
        for prev, w in zip(self.words, self.words[1:]):
            if w["x0"] - prev["x1"] > gap:
                break
            out.append(w)
        return out

    @property
    def cell_text(self) -> str:
        return " ".join(w["text"] for w in self.cell_words)

    def words_in(self, a: int, b: int) -> list[dict]:
        return [w for w, (s, e) in zip(self.words, self.spans) if e > a and s < b]

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)


def _box(ws: list[dict]) -> list[float]:
    return [round(min(w["x0"] for w in ws), 1), round(min(w["y0"] for w in ws), 1), round(max(w["x1"] for w in ws), 1), round(max(w["y1"] for w in ws), 1)]


def _hit(label: str, text: str, ws: list[dict], page: int, labelled: bool = True) -> dict:
    """A field found by rules; its confidence comes from the OCR confidence of its words and the validity of its format (documind.inference.confidence)."""
    from .confidence import evidence_confidence, valid_format
    text = text.strip(" :;-|")
    return {"label": label, "text": text, "confidence": evidence_confidence(ws, valid_format(label, text), labelled), "bbox": _box(ws), "page": page, "source": "rules"}


# ---- label vocabulary (folded: no accents, lower case, "n°" -> "no")
LAB = {
    "DUE_DATE": r"date d'?\s*echeance|echeance|due date|date limite|payable (?:avant|le)|a regler avant|a payer avant|payment due|pay by",
    "DATE": r"date d'?\s*emission|date de facture|date de la facture|date facture|issue date|invoice date|date of issue|emise? le|facture du|fait le|date(?! d'?\s*echeance| limite| de livraison)",
    "NUMBER": r"(?:facture|invoice)\s*(?:n\s*o?\b|no\b|num\w*|#|number|ref\w*)|(?:n\s*o|numero|num)\s*(?:de\s*)?(?:facture|invoice)|facture\s*n\b|invoice\s*#",
    "PO_NUMBER": r"bon de commande|purchase order|\bpo\b\s*(?:number|no|#)?|n\s*o\s*(?:de\s*)?commande|order\s*(?:no|number|#)|commande\s*n",
    "BUYER": r"facture[e]? a\b|facturer a|client\b|destinataire|bill(?:ed)? to|sold to|adresse[e]? a|invoice to|customer|acheteur",
    "SEND_TO": r"livre[e]? a\b|adresse de livraison|ship(?:ped)? to|deliver(?:ed)? to|send to|livraison a",
    "SELLER_NAME": r"emetteur|fournisseur|vendeur|raison sociale|\bfrom\s*:|prestataire",
    "TOTAL_WORDS": r"arretee? (?:la )?(?:presente )?facture(?: a la somme(?: de)?)?|a la somme de|somme de|montant en (?:toutes )?lettres|en toutes lettres|amount in words|total in words|in words",
    "AMOUNT_DUE": r"net a payer|amount due|balance due|reste a payer|solde a payer|total a payer|montant a payer|total due",
    "DISCOUNT": r"remise|discount|rabais|reduction",
    "CONDITIONS": r"conditions? (?:de )?(?:paiement|reglement)|payment terms?|terms (?:and|&) conditions|modalites? de (?:paiement|reglement)|delai de paiement",
    "NOTE": r"^notes?\b|remarques?|observations?|mentions? legales?|commentaires?",
    "PAYMENT_DETAILS": r"coordonnees bancaires|payment details|bank details|details? (?:de )?paiement|informations? bancaires?|\brib\b|\biban\b",
    "GSTIN": r"gstin|n\s*o\s*tva|tva intracommunautaire|vat (?:number|no|id|reg\w*)|\bice\b|identifiant fiscal|\bif\b|\brc\b|siret|siren|\btin\b|tax id",
}
_TITLE = r"\b(?:tax invoice|pro ?forma(?: invoice)?|facture(?: pro ?forma| d'?acompte| de? avoir)?|invoice|devis|avoir|bon de livraison|credit note|receipt|recu)\b"
_BANK = re.compile(r"iban|rib|bic|swift|banque|bank|compte|account|\d{10,}|agence|code")
_CURRENCY_WORD = re.compile(r"dirhams?|dhs?|mad|euros?|eur|dollars?|usd|cents?|centimes?|francs?")


def _is_label(f: str, ignore: tuple = ()) -> bool:
    """The line STARTS with a known label (a sentence that merely contains the word "date" is not one)."""
    return any(re.match(p, f.strip()) for k, p in LAB.items() if k not in ignore)


def _cell_end(line: _Line, start: int) -> int:
    """Character position where the cell that contains position `start` ends: a wide horizontal gap between two words starts another column
    ("Bill To :   ....   Invoice No. : INV-1" is two cells, not one)."""
    gap = 2.2 * line.h
    last_end = start
    for w, (s, e) in zip(line.words, line.spans):
        if e <= start:
            continue
        idx = line.words.index(w)
        if idx > 0 and s > start and w["x0"] - line.words[idx - 1]["x1"] > gap:
            return s - 1
        last_end = e
    return last_end


def _after(line: _Line, m: re.Match) -> tuple[str, list[dict]]:
    """The text of the label's own cell after the label (without the ':' / '-'), and its words."""
    a = m.end()
    while a < len(line.f) and line.f[a] in " :-|.#":
        a += 1
    ws = line.words_in(a, _cell_end(line, m.end()))
    return " ".join(w["text"] for w in ws), ws


def _below(lines: list[_Line], i: int, k: int = 1, x_tol: float = 0.35, width: float = 1.0) -> list[_Line]:
    """The next k lines under line i that sit roughly in the same column (their left edge within x_tol of the page width of the label's)."""
    out = []
    for ln in lines[i + 1: i + 1 + k + 2]:
        if len(out) == k or ln.y0 - lines[i].y1 > 2.5 * lines[i].h + 2 * (len(out) + 1) * ln.h:
            break
        if abs(ln.x0 - lines[i].x0) <= x_tol * width or ln.x0 >= lines[i].x0 - 0.05 * width:
            out.append(ln)
    return out


def _value_re(line: _Line, m: re.Match, value: str, lines: list[_Line], i: int, page_w: float, look_below: bool = True):
    """Value after the label on the same line, else the first match on the next line(s) -> (text, words) or None."""
    stop = _cell_end(line, m.end())
    mv = re.search(value, line.f[m.end():stop])
    if mv:
        ws = line.words_in(m.end() + mv.start(), m.end() + mv.end())
        if ws:
            return " ".join(w["text"] for w in ws), ws
    if look_below:
        for nxt in _below(lines, i, 2, width=page_w):
            mv = re.search(value, nxt.f)
            if mv and (ws := nxt.words_in(mv.start(), mv.end())):
                return " ".join(w["text"] for w in ws), ws
    return None


def _find_labelled(lines: list[_Line], field: str, value: str | None, page_w: float, skip: set[int] | None = None, look_below: bool = True):
    pat = re.compile(LAB[field])
    for i, ln in enumerate(lines):
        if skip and i in skip:
            continue
        m = pat.search(ln.f)
        if not m:
            continue
        if value:
            got = _value_re(ln, m, value, lines, i, page_w, look_below)
            if got and got[1]:
                return got[0], got[1], i
        else:
            txt, ws = _after(ln, m)
            if txt:
                return txt, ws, i
            below = _below(lines, i, 1, width=page_w)
            if below and not _is_label(below[0].f):
                return below[0].cell_text, below[0].cell_words, i
    return None


def _block_after(lines: list[_Line], i: int, first: tuple[str, list[dict]] | None, n: int, page_w: float, accept=None, ignore: tuple = ("DATE", "GSTIN", "DISCOUNT")) -> tuple[str, list[dict]]:
    texts, ws = ([first[0]], list(first[1])) if first and first[0] else ([], [])
    for ln in _below(lines, i, n, width=page_w):
        if (accept and not accept(ln)) or _is_label(ln.f, ignore):
            break
        texts.append(ln.cell_text)
        ws += ln.cell_words
    return " ".join(texts), ws


def fill_missing(fields: list[dict], words: list[dict], page: int = 1, classes: list[str] | None = None, page_size: tuple[float, float] | None = None) -> list[dict]:
    """Extra fields (see the module docstring) for the classes that `fields` does not contain."""
    from .postprocess import _lines
    have = {f["label"] for f in fields}
    want = lambda k: k not in have and (classes is None or k in classes)
    if not words:
        return []
    lines = [_Line(ln) for ln in _lines(words)]
    if not lines:
        return []
    W = page_size[0] if page_size else max(l.x1 for l in lines)
    H = page_size[1] if page_size else max(l.y1 for l in lines)
    out: list[dict] = []
    used: set[int] = set()

    # ---- dates, number, purchase order
    if want("DUE_DATE"):
        if (r := _find_labelled(lines, "DUE_DATE", _DATE, W)):
            out.append(_hit("DUE_DATE", r[0], r[1], page)); used.add(r[2])
    if want("DATE"):
        r = _find_labelled(lines, "DATE", _DATE, W, skip=used)
        if not r:                                                                    # no label: the first date in the upper half that is not the due date
            for i, ln in enumerate(lines):
                m = re.search(_DATE, ln.f)
                if m and ln.y0 < 0.5 * H and i not in used and not re.search(LAB["DUE_DATE"], ln.f):
                    r = (" ".join(w["text"] for w in ln.words_in(m.start(), m.end())), ln.words_in(m.start(), m.end()), i, False)
                    break
        if r:
            out.append(_hit("DATE", r[0], r[1], page, labelled=len(r) < 4))
    for k in ("NUMBER", "PO_NUMBER"):
        if want(k) and (r := _find_labelled(lines, k, rf"(?=[a-z0-9/_\-.]*\d)(?!\d{{1,2}}[/\-.]\d{{1,2}}[/\-.]\d{{2,4}}\b){_ID}", W)):
            out.append(_hit(k, r[0], r[1], page))

    # ---- title (the biggest word that is "FACTURE", "INVOICE" ...) in the upper third
    if want("TITLE"):
        for ln in lines:
            if ln.y0 > 0.35 * H:
                break
            m = re.search(_TITLE, ln.f)
            if m and len(ln.text) <= 40:
                ws = ln.words_in(m.start(), m.end())
                out.append(_hit("TITLE", " ".join(w["text"] for w in ws), ws, page))
                break

    # ---- e-mail and web site
    full = [(ln, m) for ln in lines for m in re.finditer(r"[\w.+-]+@[\w-]+\.[\w.-]+", ln.text)]
    if want("SELLER_EMAIL") and full:
        ln, m = next(((l, mm) for l, mm in full if l.y0 < 0.4 * H), full[0])
        ws = ln.words_in(m.start(), m.end())
        out.append(_hit("SELLER_EMAIL", m.group(0).strip(".,;"), ws, page))
    if want("SELLER_SITE"):
        for ln in lines:
            t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", ln.text)
            m = re.search(r"(?:https?://)?(?:www\.)[\w.-]+\.[a-z]{2,}\S*|\b[\w-]{3,}\.(?:com|ma|fr|org|net|io|co|biz|info)\b\S*", t, re.I)
            if m:
                out.append(_hit("SELLER_SITE", m.group(0).strip(".,;"), ln.words_in(*(ln.text.find(m.group(0)), ln.text.find(m.group(0)) + len(m.group(0)))) or ln.words, page))
                break

    # ---- parties: the lines under "Bill To" / "Facturé à" / "Client"
    buyer_line = None
    if want("BUYER") or want("GSTIN_BUYER"):
        pat = re.compile(LAB["BUYER"])
        for i, ln in enumerate(lines):
            m = pat.search(ln.f)
            if not m or len(ln.cell_text) > 60:
                continue
            txt, ws = _after(ln, m)
            if not txt:
                below = _below(lines, i, 1, width=W)
                if below and not re.search(r"\d{5,}|@", below[0].f) and not _is_label(below[0].f):
                    txt, ws = below[0].cell_text, below[0].cell_words
            if txt:
                buyer_line = i
                if want("BUYER"):
                    out.append(_hit("BUYER", txt, ws, page))
                break
    if want("SEND_TO") and (r := _find_labelled(lines, "SEND_TO", None, W)):
        out.append(_hit("SEND_TO", r[0], r[1], page))

    seller_idx = None
    if want("SELLER_NAME") or want("SELLER_ADDRESS"):
        r = _find_labelled(lines, "SELLER_NAME", None, W)
        if r:
            seller_idx = r[2] if want("SELLER_NAME") else r[2]
            if want("SELLER_NAME"):
                out.append(_hit("SELLER_NAME", r[0], r[1], page))
        elif want("SELLER_NAME"):
            def plain(ln):
                return not (re.search(_TITLE, ln.f) or re.search(r"@|www\.|\d{4,}|\b(?:tel|fax|date|n\s*o)\b", ln.f) or _is_label(ln.f))
            med = sorted(l.h for l in lines)[len(lines) // 2]
            cand = [(ln.h, i, ln) for i, ln in enumerate(lines[:8]) if ln.y0 < 0.25 * H and plain(ln) and len(ln.text) >= 3 and ln.h >= 1.2 * med
                    and (buyer_line is None or i < buyer_line)]               # a company name stands out (bigger than the body text); a plain line is not a guess worth making
            if cand:
                _, i, ln = max(cand)
                seller_idx = i
                out.append(_hit("SELLER_NAME", ln.cell_text, ln.cell_words, page, labelled=False))
    if want("SELLER_ADDRESS") and seller_idx is not None:
        below = []
        for ln in _below(lines, seller_idx, 3, width=W):
            if re.search(r"@|www\.|\b(?:tel|fax|gsm|ice|if|rc)\b", ln.f) or re.search(_TITLE, ln.f) or _is_label(ln.f):
                break
            below.append(ln)
        if below and any(re.search(r"\d|rue|avenue|av\b|bd|boulevard|street|road|lot|bp|casablanca|rabat|marrakech|tanger|fes|agadir|paris", b.f) for b in below):
            out.append(_hit("SELLER_ADDRESS", " ".join(b.cell_text for b in below), [w for b in below for w in b.cell_words], page, labelled=False))

    # ---- amount in words, amount due, discount
    if want("TOTAL_WORDS") and (r := _find_labelled(lines, "TOTAL_WORDS", None, W)):
        i = r[2]
        text, ws = r[0], list(r[1])
        if not _CURRENCY_WORD.search(fold(text)):
            text, ws = _block_after(lines, i, (text, ws), 2, W, accept=lambda l: not re.search(r"\d{2,}", l.f) or bool(_CURRENCY_WORD.search(l.f)))
        if re.search(r"[a-z]{3}", fold(text)):
            out.append(_hit("TOTAL_WORDS", text, ws, page))
    if want("AMOUNT_DUE") and (r := _find_labelled(lines, "AMOUNT_DUE", _MONEY, W)):
        out.append(_hit("AMOUNT_DUE", r[0], r[1], page))
    if want("DISCOUNT") and (r := _find_labelled(lines, "DISCOUNT", _MONEY, W, look_below=False)):
        out.append(_hit("DISCOUNT", r[0], r[1], page))

    # ---- terms, note, bank details
    if want("CONDITIONS") and (r := _find_labelled(lines, "CONDITIONS", None, W)):
        text, ws = _block_after(lines, r[2], (r[0], r[1]), 1, W) if len(r[0]) < 12 else (r[0], r[1])
        out.append(_hit("CONDITIONS", text, ws, page))
    if want("NOTE") and (r := _find_labelled(lines, "NOTE", None, W)):
        text, ws = _block_after(lines, r[2], (r[0], r[1]), 2, W)
        out.append(_hit("NOTE", text, ws, page))
    if want("PAYMENT_DETAILS"):
        for i, ln in enumerate(lines):
            if re.search(LAB["PAYMENT_DETAILS"], ln.f):
                text, ws = _block_after(lines, i, (ln.cell_text, ln.cell_words), 3, W, accept=lambda l: bool(_BANK.search(l.f)), ignore=("DATE", "GSTIN", "DISCOUNT", "PAYMENT_DETAILS"))
                out.append(_hit("PAYMENT_DETAILS", text, ws, page))
                break

    # ---- tax identifiers: the buyer's are the ones in the buyer block (under the "Bill To" label), the others are the seller's
    if any(want(k) for k in ("GSTIN", "GSTIN_SELLER", "GSTIN_BUYER")):
        pat = re.compile(LAB["GSTIN"])
        found = []
        for i, ln in enumerate(lines):
            m = pat.search(ln.f)
            if not m:
                continue
            got = _value_re(ln, m, r"(?=[a-z0-9]*\d{5,})[a-z0-9]{6,20}", lines, i, W, look_below=False)
            if got and got[1]:
                found.append((i, got))
        for i, got in found:
            bl = lines[buyer_line] if buyer_line is not None else None
            in_buyer = bl is not None and bl.y0 - bl.h <= lines[i].y0 <= bl.y0 + 9 * bl.h and abs(lines[i].x0 - bl.x0) <= 0.3 * W
            for k in (("GSTIN_BUYER",) if in_buyer else (("GSTIN_SELLER",) if buyer_line is not None else ())) + ("GSTIN",):     # no buyer block found: do not guess whose it is
                if want(k) and not any(o["label"] == k for o in out):
                    out.append(_hit(k, got[0], got[1], page))
    return sorted(out, key=lambda f: (f["bbox"][1], f["bbox"][0]))
