"""Invoice checks: do the numbers, dates and identifiers that were read agree with each other?  (the same idea as the cheque and contract checks)

    run_checks(fields, items, items_check) -> {"checks": {name: True | False | None}, "warnings": [{"code", "message"}], "summary": "7/9 passed; failed: ..."}

True = checked and fine, False = checked and WRONG, None = could not be checked (a field was not found). A failed check becomes a warning on the screen,
so a wrong amount is flagged instead of silently written to the CSV. A check never changes a value.

    total_arithmetic          subtotal + tax - discount = total
    items_sum                 the line items add up to the printed subtotal (or to the total, for tables whose amounts include tax)
    rows_arithmetic           on every row: quantity x unit price (+ the row's tax) = the row's amount
    tax_rate_consistent       tax / subtotal equals the rate printed in the table (else: is a plausible rate, 0 to 30 %)
    dates_valid               the date and the due date are real calendar dates
    due_after_issue           the due date is not before the invoice date
    amount_in_words           the amount written in words (French or English) equals the total
    amount_due_vs_total       the amount due is not more than the total (less = a deposit was paid: not checked)
    required_fields           NUMBER, DATE and TOTAL (or AMOUNT_DUE) were found
    email_valid / iban_valid  the seller's e-mail and the IBAN (mod 97) are well formed
"""
from __future__ import annotations

import re
from datetime import date

from ..features.patterns import EMAIL_RE, parse_amount

_MONTHS = {"jan": 1, "janv": 1, "feb": 2, "fev": 2, "fevr": 2, "mar": 3, "apr": 4, "avr": 4, "may": 5, "mai": 5, "jun": 6, "juin": 6, "jul": 7, "juil": 7,
           "aug": 8, "aou": 8, "aout": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
CHECK_NAMES = ("total_arithmetic", "items_sum", "rows_arithmetic", "tax_rate_consistent", "dates_valid", "due_after_issue", "amount_in_words",
               "amount_due_vs_total", "required_fields", "email_valid", "iban_valid")


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def _month(name: str) -> int | None:
    return _MONTHS.get(name[:4]) or _MONTHS.get(name[:3])


def parse_date(text: str) -> date | None:
    """A calendar date inside a text: 18/09/2025, 2025-09-18, 18-Sep-2025, 18 septembre 2025, Sep 18, 2025. Day first when it is ambiguous. None if there is none or it is not real."""
    t = _fold(text or "")
    try:
        if (m := re.search(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b", t)):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if (m := re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", t)):
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            y += 2000 if y < 100 else 0
            return date(y, b, a) if b <= 12 else date(y, a, b)
        if (m := re.search(r"\b(\d{1,2})[\s\-./,]*([a-z]{3,9})\.?[\s\-./,]*(\d{2,4})\b", t)) and _month(m[2]):
            y = int(m[3]) + (2000 if int(m[3]) < 100 else 0)
            return date(y, _month(m[2]), int(m[1]))
        if (m := re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", t)) and _month(m[1]):
            return date(int(m[3]), _month(m[1]), int(m[2]))
    except ValueError:
        return None
    return None


def _pct(text: str) -> float | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text or "")
    return float(m[1].replace(",", ".")) if m else None


def _iban_ok(raw: str) -> bool | None:
    m = re.search(r"\b([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30})\b", (raw or "").upper())
    if not m:
        return None
    s = re.sub(r"\s", "", m[1])
    if not 15 <= len(s) <= 34:
        return None
    try:
        return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1
    except ValueError:
        return False


def _words_value(text: str) -> float | None:
    from .cheques import words_to_number
    from .contract_text import english_words_to_number
    for fn in (words_to_number, english_words_to_number):
        try:
            r = fn(text)
        except Exception:
            r = None
        if r:
            return r[0] + r[1] / 100.0
    return None


def run_checks(fields: list[dict], items: list[dict] | None = None, items_check: dict | None = None) -> dict:
    from .postprocess import _row_consistent, _val
    items, items_check = items or [], items_check or {}
    F: dict[str, str] = {}
    for f in fields:
        if f.get("source") != "computed":                 # a value computed from the other amounts cannot be used to check them (circular)
            F.setdefault(f["label"], f["text"])
    num = lambda k: parse_amount(F[k]) if k in F else None
    sub, tax, tot, due, gst = num("SUB_TOTAL"), num("TAX"), num("TOTAL"), num("AMOUNT_DUE"), num("GST")
    disc_txt = F.get("DISCOUNT", "")
    disc, disc_pct = num("DISCOUNT"), _pct(disc_txt)
    checks: dict[str, bool | None] = {k: None for k in CHECK_NAMES}
    why: dict[str, str] = {}
    tol = lambda x: max(0.02, 0.0015 * abs(x))

    # subtotal + tax - discount = total
    if sub is not None and tot is not None:
        taxes = (tax or 0.0) + (gst if gst is not None and tax is None else 0.0)
        variants = {sub + taxes, sub + taxes - (disc or 0.0)}
        if disc_pct is not None:
            variants.add(sub * (1 - disc_pct / 100.0) + taxes)
        if abs(tot - sub) <= tol(tot) or any(abs(tot - v) <= tol(tot) for v in variants):
            checks["total_arithmetic"] = True
        elif tax is not None or gst is not None or disc is not None:
            checks["total_arithmetic"] = False
            why["total_arithmetic"] = f"Subtotal {sub:,.2f} + tax {taxes:,.2f}" + (f" - discount {disc:,.2f}" if disc else "") + f" = {sub + taxes - (disc or 0):,.2f}, but the invoice prints a total of {tot:,.2f}."

    # the line items add up
    isum = items_check.get("items_sum")
    if items and isum is not None and (sub is not None or tot is not None):
        ok = any(x is not None and abs(isum - x) <= tol(x) for x in (sub, tot, (sub + tax) if sub is not None and tax is not None else None))
        checks["items_sum"] = bool(ok)
        if not ok:
            why["items_sum"] = f"The line items add up to {isum:,.2f}, the invoice prints a subtotal of {sub:,.2f}" if sub is not None else f"The line items add up to {isum:,.2f}, the invoice prints a total of {tot:,.2f}"
            why["items_sum"] += ": a row may be missing or misread."

    # every row: qty x price (+ tax) = amount
    rows = []
    for it in items:
        a, q, p = _val(str(it.get("amount", ""))), _val(str(it.get("qty", ""))), _val(str(it.get("unit_price", "")))
        if None not in (a, q, p):
            rows.append((it, _row_consistent(it, a, q, p)))
    if rows:
        bad = [r for r in rows if not r[1]]
        checks["rows_arithmetic"] = not bad
        if bad:
            why["rows_arithmetic"] = f"{len(bad)} of {len(rows)} rows do not satisfy quantity x unit price = amount (e.g. {str(bad[0][0].get('description', ''))[:40]!r})."

    # tax rate
    if sub and tax is not None and sub > 0:
        rate = 100.0 * tax / sub
        listed = [r for it in items for k, v in it.items() if re.search(r"(tax|vat|gst|tva)", k) and re.search(r"rate|pct", k) and (r := _pct(str(v)) if "%" in str(v) else _val(str(v))) is not None]
        if listed:
            checks["tax_rate_consistent"] = any(abs(rate - r) <= 0.6 for r in listed)
            if not checks["tax_rate_consistent"]:
                why["tax_rate_consistent"] = f"Tax / subtotal is {rate:.1f} %, the table prints {', '.join(f'{r:g} %' for r in sorted(set(listed)))}."
        else:
            checks["tax_rate_consistent"] = 0.0 <= rate <= 30.0
            if not checks["tax_rate_consistent"]:
                why["tax_rate_consistent"] = f"Tax / subtotal is {rate:.1f} %, which is not a plausible tax rate."

    # dates
    d_issue, d_due = parse_date(F.get("DATE", "")), parse_date(F.get("DUE_DATE", ""))
    present = [k for k in ("DATE", "DUE_DATE") if k in F]
    if present:
        parsed = {"DATE": d_issue, "DUE_DATE": d_due}
        checks["dates_valid"] = all(parsed[k] is not None for k in present)
        if not checks["dates_valid"]:
            bad = [f"{k} ({F[k]!r})" for k in present if parsed[k] is None]
            why["dates_valid"] = "Not a valid calendar date: " + ", ".join(bad) + "."
    if d_issue and d_due:
        checks["due_after_issue"] = d_due >= d_issue
        if not checks["due_after_issue"]:
            why["due_after_issue"] = f"The due date ({d_due:%d/%m/%Y}) is before the invoice date ({d_issue:%d/%m/%Y})."

    # amount in words
    wv = _words_value(F["TOTAL_WORDS"]) if "TOTAL_WORDS" in F else None
    ref = tot if tot is not None else due
    if wv is not None and ref is not None:
        checks["amount_in_words"] = abs(wv - ref) <= max(0.011, 0.001 * abs(ref))
        if not checks["amount_in_words"]:
            why["amount_in_words"] = f"The amount in words is {wv:,.2f}, the invoice prints {ref:,.2f}."

    # amount due
    if due is not None and tot is not None:
        if abs(due - tot) <= tol(tot):
            checks["amount_due_vs_total"] = True
        elif due > tot:
            checks["amount_due_vs_total"] = False
            why["amount_due_vs_total"] = f"The amount due ({due:,.2f}) is more than the total ({tot:,.2f})."

    # required fields
    need = {"NUMBER": "NUMBER" in F, "DATE": "DATE" in F, "TOTAL": "TOTAL" in F or "AMOUNT_DUE" in F}
    checks["required_fields"] = all(need.values())
    if not checks["required_fields"]:
        why["required_fields"] = "Not found: " + ", ".join(k for k, v in need.items() if not v) + "."

    # e-mail, IBAN
    if "SELLER_EMAIL" in F:
        checks["email_valid"] = bool(EMAIL_RE.fullmatch(F["SELLER_EMAIL"].strip()))
        if not checks["email_valid"]:
            why["email_valid"] = f"The seller e-mail {F['SELLER_EMAIL']!r} is not well formed."
    if "PAYMENT_DETAILS" in F:
        checks["iban_valid"] = _iban_ok(F["PAYMENT_DETAILS"])
        if checks["iban_valid"] is False:
            why["iban_valid"] = "The IBAN does not pass its check digits: a digit was probably misread."

    done = {k: v for k, v in checks.items()}
    passed = sum(1 for v in done.values() if v is True)
    failed = [k for k, v in done.items() if v is False]
    checked = passed + len(failed)
    return {"checks": done,
            "warnings": [{"code": "check_failed", "check": k, "message": why.get(k, k)} for k in failed],
            "summary": f"{passed}/{checked} passed" + (f"; failed: {', '.join(failed)}" if failed else "") if checked else "nothing could be checked"}


_EVIDENCE = {"total_arithmetic": ("SUB_TOTAL", "TAX", "GST", "DISCOUNT", "TOTAL"), "items_sum": ("SUB_TOTAL", "TOTAL"), "tax_rate_consistent": ("TAX", "SUB_TOTAL"),
             "dates_valid": ("DATE", "DUE_DATE"), "due_after_issue": ("DATE", "DUE_DATE"), "amount_in_words": ("TOTAL_WORDS", "TOTAL"),
             "amount_due_vs_total": ("AMOUNT_DUE", "TOTAL"), "email_valid": ("SELLER_EMAIL",), "iban_valid": ("PAYMENT_DETAILS",)}


def adjust_confidence(fields: list[dict], checks: dict) -> None:
    """The checks are evidence: a field that a PASSING check confirms gets +0.08 (at most 0.97); a field that a FAILING check contradicts is multiplied by 0.7.
    A value computed from the other amounts is never raised by the check it was computed for (that would be circular). Adds `verified_by` / `contradicted_by`."""
    for f in fields:
        passed = [k for k, labels in _EVIDENCE.items() if checks.get(k) is True and f["label"] in labels]
        failed = [k for k, labels in _EVIDENCE.items() if checks.get(k) is False and f["label"] in labels]
        if f.get("source") == "computed":
            passed = []
        if passed:
            f["confidence"] = round(min(0.97, float(f.get("confidence", 0.0)) + 0.08), 2)
            f["verified_by"] = passed
        if failed:
            f["confidence"] = round(float(f.get("confidence", 0.0)) * 0.7, 2)
            f["contradicted_by"] = failed
