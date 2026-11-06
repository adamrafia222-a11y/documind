"""Is the corpus trustworthy?  Run this BEFORE tuning any model.

A model can only be as good as its labels. These checks tell you, in a minute, whether low scores come from the
model or from the data (labels shifted, OCR missing words, wrong field names...).

    python -m documind.data.audit                              # uses data/processed/corpus.jsonl
    python -m documind.data.audit --data data/raw/FATURA --n 60   # + compares annotation boxes with the OCR words

Checks
  1. label purity   : do words labelled DATE look like dates, TOTAL like amounts, SELLER_EMAIL like e-mails ...?
                      (if DATE words are mostly NOT dates, the labels are not on the right words)
  2. alignment      : which coordinate convention was used, how well the boxes fit the OCR words
  3. box coverage   : annotation boxes that contain NO OCR word at all (the OCR missed that text)
  4. text agreement : when the annotation file has the text of each box, how much does it match the OCR text?
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BACKGROUND_LABEL, CORPUS_PATH, OCR_LANG, OCR_PSM
from ..features.builder import DATE_RE, EMAIL_RE, MONEY_RE, URL_RE, parse_amount
from .corpus import load_corpus

EXPECTED = {   # label -> which word-type check should be high
    "DATE": "date", "DUE_DATE": "date", "TOTAL": "money", "SUB_TOTAL": "money", "TAX": "money",
    "DISCOUNT": "money", "AMOUNT_DUE": "money", "SELLER_EMAIL": "email", "SELLER_SITE": "url",
}


def _kind(text: str) -> dict:
    t = text.strip()
    return {"date": bool(DATE_RE.match(t.rstrip(":"))) or any(c.isdigit() for c in t) and bool(DATE_RE.match(t)),
            "money": bool(MONEY_RE.match(t)) and parse_amount(t) is not None and (any(c in t for c in ",.$€£")),
            "email": bool(EMAIL_RE.search(t)), "url": bool(URL_RE.search(t)),
            "digit": any(c.isdigit() for c in t)}


def label_purity(docs: list[dict], min_doc_hit: float = 0.8) -> pd.DataFrame:
    """Do the words under each label look like that field?

    The word-level percentages are for information only: an annotation box also contains the key words of the
    field ("Date:", "Total:", "EUR"...), so even perfectly placed labels never reach 100 % (a DATE box
    "Date: 20-Mar-2008" is at most 50 % date-like). The verdict therefore uses the DOCUMENT level:
    `%docs hit` = share of the documents where the label's words contain at least one word of the expected
    kind (date, amount, e-mail, url). Below `min_doc_hit` (80 %) the label sits on the wrong words.
    """
    rows: dict[str, Counter] = {}
    for d in docs:
        hit_in_doc: dict[str, set] = {}
        for w in d["words"]:
            c = rows.setdefault(w["label"], Counter())
            c["words"] += 1
            for k, v in _kind(w["text"]).items():
                c[k] += v
                if v:
                    hit_in_doc.setdefault(w["label"], set()).add(k)
        for lab in {w["label"] for w in d["words"]}:
            rows[lab]["docs"] += 1
            exp = EXPECTED.get(lab)
            if exp and exp in hit_in_doc.get(lab, ()):
                rows[lab]["docs_hit"] += 1
    out = []
    for lab, c in sorted(rows.items(), key=lambda kv: -kv[1]["words"]):
        n = c["words"]
        exp = EXPECTED.get(lab)
        hit = 100 * c["docs_hit"] / max(c["docs"], 1)
        out.append({"label": lab, "words": n, "%date-like": 100 * c["date"] / n, "%money-like": 100 * c["money"] / n,
                    "%email": 100 * c["email"] / n, "%url": 100 * c["url"] / n, "%has digit": 100 * c["digit"] / n,
                    "expected": exp or "", "%docs hit": hit if exp else np.nan,
                    "ok": "" if not exp else "OK" if hit >= 100 * min_doc_hit else "SUSPICIOUS"})
    return pd.DataFrame(out).round(1)


def alignment_overview(docs: list[dict]) -> dict:
    modes = Counter(d.get("align") for d in docs)
    fits = np.array([d.get("align_fit", 0.0) / max(d.get("n_annotations", 1), 1) for d in docs])
    lab_frac = np.array([np.mean([w["label"] != BACKGROUND_LABEL for w in d["words"]]) for d in docs])
    per_t: dict[str, list[float]] = {}
    for d, f in zip(docs, lab_frac):
        per_t.setdefault(d["template"], []).append(f)
    return {"modes": dict(modes), "mean_fit_per_box": float(fits.mean()), "fit_p10": float(np.percentile(fits, 10)),
            "words_per_doc": float(np.mean([len(d["words"]) for d in docs])),
            "labelled_fraction_mean": float(lab_frac.mean()),
            "templates_with_<5%_labelled": sorted(t for t, v in per_t.items() if np.mean(v) < 0.05)}


def audit_boxes(data_dir: str, n: int = 60, lang: str = OCR_LANG, psm: int = OCR_PSM, seed: int = 0) -> pd.DataFrame:
    """Compare annotation boxes with OCR words on n random invoices (uses the OCR cache when it exists)."""
    from .labeling import _cover, _word_boxes, align_labels
    from .loader import find_samples, read_annotation, to_pixels
    from .prepare import _ocr_words
    samples = find_samples(data_dir)
    rng = np.random.default_rng(seed)
    picked = [samples[i] for i in rng.permutation(len(samples))[:n]]
    rows = []
    for s in picked:
        words, W, H = _ocr_words(s, lang, psm, False, seed)
        anns = []
        for p in s["annotations"]:
            try:
                anns.append(read_annotation(p))
            except Exception:
                pass
        info = align_labels(words, anns, (W, H))
        if info["source"] < 0:
            rows.append({"stem": s["stem"], "label": "(NO ALIGNMENT)", "n_words": 0, "text_sim": np.nan})
            continue
        a = anns[info["source"]]
        boxes = to_pixels(np.array([b.vals for b in a.boxes], dtype=float), info["mode"], W, H, a.ref_size)
        cover = _cover(_word_boxes(words), boxes)
        for j, b in enumerate(a.boxes):
            idx = np.flatnonzero(cover[:, j] >= 0.5)
            ocr_txt = " ".join(words[i]["text"] for i in sorted(idx, key=lambda i: (round(words[i]["y0"] / 10), words[i]["x0"])))
            sim = np.nan
            if b.text:
                sim = SequenceMatcher(None, "".join(ch for ch in b.text.lower() if ch.isalnum()),
                                      "".join(ch for ch in ocr_txt.lower() if ch.isalnum())).ratio()
            rows.append({"stem": s["stem"], "label": b.label, "n_words": len(idx), "text_sim": sim,
                         "mode": info["mode"], "fit": info["fit"]})
    return pd.DataFrame(rows)


def summarize_boxes(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("label")
    return pd.DataFrame({"boxes": g.size(), "%empty (no OCR word)": 100 * g["n_words"].apply(lambda s: (s == 0).mean()),
                         "mean words/box": g["n_words"].mean(),
                         "text match 0-1": g["text_sim"].mean()}).round(2).sort_values("boxes", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--data", default=None, help="raw dataset folder: also audit annotation boxes vs OCR words")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    docs, meta = load_corpus(Path(a.corpus))
    print(f"{len(docs)} documents from {meta.get('source')}")
    print("\n[1] LABEL PURITY (SUSPICIOUS = the words under this label do not look like that field)")
    print(label_purity(docs).to_string(index=False))
    print("\n[2] ALIGNMENT")
    for k, v in alignment_overview(docs).items():
        print(f"  {k}: {v}")
    if a.data:
        print(f"\n[3/4] ANNOTATION BOXES vs OCR WORDS ({a.n} random invoices)")
        df = audit_boxes(a.data, a.n)
        print(summarize_boxes(df).to_string())


if __name__ == "__main__":
    main()
