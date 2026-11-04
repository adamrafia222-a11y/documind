"""Label merging: several annotation names that mean the SAME thing become one class.

FATURA names a field differently from template to template, which nobody can learn from the picture:

* ``GST(1%)``, ``GST(5%)``, ``GST(7%)`` ... ``GST(20%)``  -> ``GST``   (the rate is a property of the template, the
  words are the same "GST ... amount" line; the model cannot read a rate from the layout)
* ``BILL_TO`` -> ``BUYER``   (the same block - name, address, tel, e-mail, site - called "Bill to" in some templates
  and "Buyer" in others; ``SEND_TO`` stays separate: it is the *shipping* block)

Merging is applied when training (``TrainConfig.merge_labels``), so ``corpus.jsonl`` is never modified and the
original names stay available for the audit.
"""
from __future__ import annotations

import re

from ..config import BACKGROUND_LABEL

MERGE_RULES: list[tuple[str, str]] = [
    (r"^GST\(.*\)$", "GST"),
    (r"^BILL_TO$", "BUYER"),
]


def merge_label(label: str, rules=MERGE_RULES) -> str:
    for pattern, target in rules:
        if re.match(pattern, label):
            return target
    return label


def merge_labels(docs: list[dict], rules=MERGE_RULES) -> list[dict]:
    """Copy of `docs` with merged labels (the input is not modified)."""
    cache: dict[str, str] = {}
    out = []
    for d in docs:
        words = []
        for w in d["words"]:
            lab = w["label"]
            if lab not in cache:
                cache[lab] = merge_label(lab, rules)
            words.append({**w, "label": cache[lab]} if cache[lab] != lab else w)
        out.append({**d, "words": words})
    return out


def drop_labels(docs: list[dict], names: set[str]) -> list[dict]:
    """Copy of `docs` where the given labels become background."""
    return merge_labels(docs, [(rf"^{re.escape(n)}$", BACKGROUND_LABEL) for n in names])
