"""Evaluation: word-level precision / recall / F1 and field-level (what ends up in the CSV) scores."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, f1_score

from ..config import BACKGROUND_LABEL
from ..features import assign_lines


def evaluate(name: str, model, X, y, classes: list[str], verbose: bool = True) -> dict:
    """Word-level report.

    `macro_f1_fields` averages ONLY the fields that really occur in this set (support > 0). The old version also
    averaged fields with no word at all in the test templates, each counted as F1 = 0, which lowered the score
    for a reason unrelated to the model. `macro_f1_all_fields` keeps that stricter number for reference.
    """
    pred = model.predict(X)
    labels = list(range(len(classes)))
    rep = classification_report(y, pred, labels=labels, target_names=classes, zero_division=0, output_dict=True)
    field_ids = [i for i, c in enumerate(classes) if c != BACKGROUND_LABEL]
    present = [i for i in field_ids if int((y == i).sum()) > 0]
    macro_present = f1_score(y, pred, labels=present, average="macro", zero_division=0) if present else 0.0
    macro_all = f1_score(y, pred, labels=field_ids, average="macro", zero_division=0)
    if verbose:
        text = classification_report(y, pred, labels=labels, target_names=classes, zero_division=0)
        print(f"\n===== {name} (word level) =====\n{text}")
        print(f"macro-F1 on the {len(present)} fields present in this set: {macro_present:.4f}   "
              f"(all {len(field_ids)} fields, absent ones counted as 0: {macro_all:.4f})")
    return {"macro_f1_fields": float(macro_present), "macro_f1_all_fields": float(macro_all), "report": rep}


# ----------------------------------------------------------------------------- field level
def predict_docs(feat, model, classes: list[str], docs: list[dict], chunk: int = 400) -> list[list[str]]:
    """Predicted label of EVERY word of every document (nothing dropped, same order as the words)."""
    out: list[list[str]] = []
    for i, X in feat.transform_chunks(docs, chunk):
        ids = np.asarray(model.predict(X)).astype(int)
        pos = 0
        for d in docs[i:i + chunk]:
            n = len(d["words"])
            out.append([classes[k] for k in ids[pos:pos + n]])
            pos += n
    return out


def _field_texts(words: list[dict], labels: list[str]) -> dict[str, list[str]]:
    """label -> words of that label in reading order (rows top to bottom, words left to right)."""
    layout = assign_lines(words)
    res: dict[str, list[str]] = {}
    for ids in layout.lines:
        for j in ids:
            if labels[j] != BACKGROUND_LABEL:
                res.setdefault(labels[j], []).append(words[j]["text"].lower())
    return res


def evaluate_fields(docs: list[dict], pred: list[list[str]], classes: list[str], verbose: bool = True) -> dict:
    """What the user gets in the CSV: for each (document, field) compare the extracted text with the true text.

    exact = the two texts are identical; f1 = word overlap (partial credit). Only the fields that exist in the
    true annotation of a document are scored for recall; wrong extra fields count against precision.
    """
    known = set(classes)
    stats: dict[str, dict] = {}
    for d, p in zip(docs, pred):
        gold_labels = [w["label"] if w["label"] in known else BACKGROUND_LABEL for w in d["words"]]
        g, q = _field_texts(d["words"], gold_labels), _field_texts(d["words"], p)
        for c in set(g) | set(q):
            s = stats.setdefault(c, {"n_true": 0, "n_pred": 0, "exact": 0, "tp": 0, "np": 0, "ng": 0})
            gt, qt = g.get(c), q.get(c)
            s["n_true"] += gt is not None
            s["n_pred"] += qt is not None
            if gt is not None and qt is not None:
                s["exact"] += gt == qt
                overlap = sum((Counter(gt) & Counter(qt)).values())
                s["tp"] += overlap
            s["ng"] += len(gt or [])
            s["np"] += len(qt or [])
    rows = {}
    for c, s in sorted(stats.items()):
        prec = s["tp"] / s["np"] if s["np"] else 0.0
        rec = s["tp"] / s["ng"] if s["ng"] else 0.0
        rows[c] = {"docs_with_field": s["n_true"], "exact_match": s["exact"] / s["n_true"] if s["n_true"] else 0.0,
                   "token_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    present = [r for r in rows.values() if r["docs_with_field"] > 0]
    summary = {"exact_match_mean": float(np.mean([r["exact_match"] for r in present])) if present else 0.0,
               "token_f1_mean": float(np.mean([r["token_f1"] for r in present])) if present else 0.0}
    if verbose:
        print(f"\n{'field':<18}{'docs':>7}{'exact':>8}{'token-F1':>10}")
        for c, r in rows.items():
            if r["docs_with_field"]:
                print(f"{c:<18}{r['docs_with_field']:>7}{r['exact_match']:>8.2f}{r['token_f1']:>10.2f}")
        print(f"{'MEAN':<18}{'':>7}{summary['exact_match_mean']:>8.2f}{summary['token_f1_mean']:>10.2f}")
    return {"per_field": rows, **summary}
