"""Turns OCR words into numeric + text features for LightGBM.

One OCR word = one training row. The classifier must decide which field the word
belongs to (vendor, date, total, ... or "O" for background). Features describe:
  * the word itself (shape: digits, amount, date, e-mail ... + character n-grams),
  * its position on the page (normalised coordinates, line rank),
  * its context (words to the left on the same row, to the right, and the text
    right above it, which is where table headers / keys usually are).
"""
from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .layout import Layout, assign_lines, build_structure, cell_above

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|f[eé]vr|avr|juin|juil|ao[uû]t|sept|d[eé]c)[a-z]*"
DATE_RE = re.compile(
    r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$"                       # 20/03/2008, 2008-03-20
    rf"|^{_MONTH}\.?,?$"                                              # a month name on its own
    rf"|^\d{{1,2}}[/\-. ]{_MONTH}[/\-. ,]*\d{{2,4}}$"                 # 20-Mar-2008  (FATURA's format)
    rf"|^{_MONTH}[/\-. ]\d{{1,2}}[/\-. ,]+\d{{2,4}}$",                # Mar-20-2008, Mar 20, 2008
    re.I,
)
MONEY_RE = re.compile(r"^[-(]?[$€£]?\d[\d,.\s]*[$€£]?\)?$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"(https?://|www\.|\.com\b|\.org\b|\.net\b|\.ma\b|\.fr\b)", re.I)
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
CURRENCY = set("$€£")

NUM_FEATURE_NAMES = [
    "nx0", "ny0", "nx1", "ny1", "ncx", "ncy", "nw", "nh",
    "line_rank", "line_idx", "pos_in_line", "n_in_line", "is_first", "is_last",
    "gap_prev", "gap_next", "line_x0", "line_x1", "line_nw",
    "conf", "length", "digit_ratio", "alpha_ratio", "upper_ratio", "punct_ratio",
    "has_digit", "is_digits", "is_money", "has_currency", "is_date", "is_email",
    "is_url", "url_www", "url_scheme", "is_phone", "is_title", "ends_colon", "is_upper",
    # cell / block structure (key-value cells, stacked address-like blocks)
    "cell_n_words", "pos_in_cell", "cell_idx_in_line", "n_cells_in_line", "cell_is_last",
    "blk_n_lines", "blk_line_idx", "blk_n_words", "blk_has_email", "blk_has_money", "blk_has_date",
    "blk_nx0", "blk_ny0", "blk_is_head_line",
    # document-level ranks: the total is the largest amount, due date comes after the date...
    "amt_rank_desc", "amt_rel_max", "amt_rank_y", "n_amounts", "date_order", "n_dates",
    "row_n_money", "row_has_key",
]

# absolute page-position features: powerful inside one template, but they differ from template to template
POSITION_FEATURES = ("nx0", "ny0", "nx1", "ny1", "ncx", "ncy", "line_x0", "line_x1", "line_idx", "line_rank")


def norm_text(t: str) -> str:
    return re.sub(r"\d", "0", t.lower())


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _numeric(w: dict, W: float, H: float) -> dict:
    t = w["text"]
    n = len(t)
    digits = sum(c.isdigit() for c in t)
    alpha = sum(c.isalpha() for c in t)
    upper = sum(c.isupper() for c in t)
    punct = n - digits - alpha
    stripped = t.rstrip(":")
    return {
        "length": n,
        "digit_ratio": _ratio(digits, n),
        "alpha_ratio": _ratio(alpha, n),
        "upper_ratio": _ratio(upper, max(alpha, 1)),
        "punct_ratio": _ratio(punct, n),
        "has_digit": float(digits > 0),
        "is_digits": float(t.isdigit()),
        "is_money": float(bool(MONEY_RE.match(t)) and (digits > 0) and (any(c in t for c in ",.") or any(c in CURRENCY for c in t))),
        "has_currency": float(any(c in CURRENCY for c in t)),
        "is_date": float(bool(DATE_RE.match(stripped))),
        "is_email": float(bool(EMAIL_RE.search(t))),
        "is_url": float(bool(URL_RE.search(t))),
        # a seller site is written "www.Name.org", a buyer site "Site:https://name.com/": the shape tells them apart
        "url_www": float(t.lower().startswith("www.")),
        "url_scheme": float(bool(re.match(r"^(site:)?https?://", t, re.I))),
        "is_phone": float(bool(PHONE_RE.match(t)) and digits >= 7),
        "is_title": float(t[:1].isupper() and t[1:].islower()),
        "ends_colon": float(t.endswith(":")),
        "is_upper": float(alpha > 1 and upper == alpha),
    }


def parse_amount(t: str) -> float | None:
    """'$1,234.50' -> 1234.5 ; '1.234,50' -> 1234.5 ; None when it is not an amount."""
    core = re.sub(r"[^\d.,]", "", t)
    if not core or not any(c.isdigit() for c in core):
        return None
    last_dot, last_com = core.rfind("."), core.rfind(",")
    if last_dot >= 0 and last_com >= 0:
        dec = max(last_dot, last_com)
        core = core[:dec].replace(".", "").replace(",", "") + "." + core[dec + 1:]
    elif last_com >= 0:
        core = core.replace(",", ".") if len(core) - last_com - 1 in (1, 2) and core.count(",") == 1 else core.replace(",", "")
    elif core.count(".") > 1:
        core = core.replace(".", "")
    try:
        return float(core)
    except ValueError:
        return None


TEXT_KEYS = ("word", "left", "right", "above", "bhead", "bkey")


def _short(words: list[dict], ids: list[int], n: int, last: bool = False) -> str:
    sel = ids[-n:] if last else ids[:n]
    return " ".join(norm_text(words[j]["text"]) for j in sel)


def build_table(words: list[dict], width: float, height: float):
    """Return (numeric matrix, text dict) for one document."""
    n = len(words)
    if n == 0:
        return np.zeros((0, len(NUM_FEATURE_NAMES)), dtype=np.float32), {
            k: [] for k in TEXT_KEYS}, Layout([], [])

    W, H = max(width, 1.0), max(height, 1.0)
    layout = assign_lines(words)
    n_lines = len(layout.lines)
    rows = np.zeros((n, len(NUM_FEATURE_NAMES)), dtype=np.float32)
    texts = {k: [] for k in TEXT_KEYS}
    idx = {name: k for k, name in enumerate(NUM_FEATURE_NAMES)}

    line_bounds = []
    for li, ids in enumerate(layout.lines):
        line_bounds.append((words[ids[0]]["x0"], words[ids[-1]]["x1"]))

    # ---- structure: cells, blocks, per-document ranks (computed once per document)
    struct = build_structure(words, layout)
    base = [_numeric(w, W, H) for w in words]
    block_info = []
    for bi, cids in enumerate(struct.blocks):
        wids = [j for c in cids for j in struct.cells[c]["ids"]]
        head = struct.cells[cids[0]]
        key_txt = ""
        if head["pos"] > 0:                                   # key written to the left of the block
            left = struct.cells[struct.cells_in_line[head["line"]][head["pos"] - 1]]
            key_txt = _short(words, left["ids"], 4, last=True)
        else:                                                 # key written above the block
            up = cell_above(struct, cids[0])
            if up is not None:
                key_txt = _short(words, struct.cells[up]["ids"], 4)
        block_info.append({
            "wids": wids, "n_lines": len({struct.cells[c]["line"] for c in cids}),
            "has_email": float(any(base[j]["is_email"] for j in wids)),
            "has_money": float(any(base[j]["is_money"] for j in wids)),
            "has_date": float(any(base[j]["is_date"] for j in wids)),
            "x0": min(struct.cells[c]["x0"] for c in cids) / W, "y0": min(struct.cells[c]["y0"] for c in cids) / H,
            "head_line": head["line"], "head": _short(words, head["ids"], 6), "key": key_txt,
        })
    reading = sorted(range(n), key=lambda j: (layout.line_of[j], words[j]["x0"]))
    amounts = [(j, parse_amount(words[j]["text"])) for j in reading if base[j]["is_money"]]
    amounts = [(j, v) for j, v in amounts if v is not None]
    amt = {}
    if amounts:
        vmax = max(v for _, v in amounts) or 1.0
        order = sorted(range(len(amounts)), key=lambda a: -amounts[a][1])
        for rank, a in enumerate(order):
            j, v = amounts[a]
            amt[j] = (rank, v / vmax)
        amt = {j: (*amt[j], k / max(1, len(amounts) - 1)) for k, (j, _) in enumerate(amounts)}
    dates = [j for j in reading if base[j]["is_date"]]
    date_pos = {j: k for k, j in enumerate(dates)}
    line_money = [sum(base[j]["is_money"] for j in ids) for ids in layout.lines]
    line_key = [float(any(words[j]["text"].endswith(":") for j in ids)) for ids in layout.lines]

    for i, w in enumerate(words):
        li = layout.line_of[i]
        ids = layout.lines[li]
        pos = ids.index(i)
        prev_w = words[ids[pos - 1]] if pos > 0 else None
        next_w = words[ids[pos + 1]] if pos < len(ids) - 1 else None
        wh = max(1.0, w["y1"] - w["y0"])
        f = dict(base[i])
        ci = struct.cell_of[i]
        cell = struct.cells[ci]
        bi = struct.block_of_cell[ci]
        bl = block_info[bi]
        a_rank, a_rel, a_y = amt.get(i, (-1, -1.0, -1.0))
        f.update(
            nx0=w["x0"] / W, ny0=w["y0"] / H, nx1=w["x1"] / W, ny1=w["y1"] / H,
            ncx=(w["x0"] + w["x1"]) / (2 * W), ncy=(w["y0"] + w["y1"]) / (2 * H),
            nw=(w["x1"] - w["x0"]) / W, nh=(w["y1"] - w["y0"]) / H,
            line_rank=li / max(1, n_lines - 1), line_idx=min(li, 200),
            pos_in_line=pos, n_in_line=len(ids),
            is_first=float(pos == 0), is_last=float(pos == len(ids) - 1),
            gap_prev=((w["x0"] - prev_w["x1"]) / wh) if prev_w else -1.0,
            gap_next=((next_w["x0"] - w["x1"]) / wh) if next_w else -1.0,
            line_x0=line_bounds[li][0] / W, line_x1=line_bounds[li][1] / W,
            line_nw=(line_bounds[li][1] - line_bounds[li][0]) / W,
            conf=w["conf"] / 100.0,
            cell_n_words=len(cell["ids"]), pos_in_cell=cell["ids"].index(i), cell_idx_in_line=cell["pos"],
            n_cells_in_line=len(struct.cells_in_line[li]), cell_is_last=float(cell["pos"] == len(struct.cells_in_line[li]) - 1),
            blk_n_lines=bl["n_lines"], blk_line_idx=li - bl["head_line"], blk_n_words=len(bl["wids"]),
            blk_has_email=bl["has_email"], blk_has_money=bl["has_money"], blk_has_date=bl["has_date"],
            blk_nx0=bl["x0"], blk_ny0=bl["y0"], blk_is_head_line=float(li == bl["head_line"]),
            amt_rank_desc=a_rank, amt_rel_max=a_rel, amt_rank_y=a_y, n_amounts=len(amounts),
            date_order=date_pos.get(i, -1), n_dates=len(dates),
            row_n_money=line_money[li], row_has_key=line_key[li],
        )
        for name, val in f.items():
            rows[i, idx[name]] = val

        texts["word"].append(norm_text(w["text"]))
        texts["left"].append(" ".join(norm_text(words[j]["text"]) for j in ids[max(0, pos - 4):pos]))
        texts["right"].append(" ".join(norm_text(words[j]["text"]) for j in ids[pos + 1:pos + 3]))

        up = cell_above(struct, ci)
        texts["above"].append(_short(words, struct.cells[up]["ids"], 5) if up is not None else "")
        texts["bhead"].append(bl["head"])
        texts["bkey"].append(bl["key"])
    return rows, texts, layout


class Featurizer:
    """Fits TF-IDF vocabularies on the training documents and builds sparse matrices.

    position="full"  keeps the absolute page-position features (strong inside one template);
    position="none"  drops them, so the model must rely on the word, its key (left / above cell) and its block.
    On unseen templates "none" is often better: compare both (the training notebook does it).
    """

    def __init__(self, position: str = "full") -> None:
        if position not in ("full", "none"):
            raise ValueError("position must be 'full' or 'none'")
        self.position = position
        self.v_word = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=2500,
                                      min_df=2, sublinear_tf=True, lowercase=False, dtype=np.float32)
        ctx = dict(analyzer="word", ngram_range=(1, 2), max_features=1000, min_df=2,
                   sublinear_tf=True, lowercase=False, token_pattern=r"[a-z0-9]+|[:#$€%/]", dtype=np.float32)
        self.v_left = TfidfVectorizer(**ctx)
        self.v_right = TfidfVectorizer(**{**ctx, "max_features": 500})
        self.v_above = TfidfVectorizer(**ctx)
        self.v_bhead = TfidfVectorizer(**ctx)
        self.v_bkey = TfidfVectorizer(**ctx)
        drop = set(POSITION_FEATURES) if position == "none" else set()
        self.num_cols = [i for i, name in enumerate(NUM_FEATURE_NAMES) if name not in drop]

    def _vectorizers(self):
        return (("word", self.v_word), ("left", self.v_left), ("right", self.v_right),
                ("above", self.v_above), ("bhead", self.v_bhead), ("bkey", self.v_bkey))

    @staticmethod
    def _tables(docs: list[dict]):
        nums, texts = [], {k: [] for k in TEXT_KEYS}
        for d in docs:
            n, t, _ = build_table(d["words"], d["width"], d["height"])
            nums.append(n)
            for k in texts:
                texts[k].extend(t[k])
        return np.vstack(nums) if nums else np.zeros((0, len(NUM_FEATURE_NAMES)), np.float32), texts

    def fit(self, docs: list[dict], max_docs: int = 2000, seed: int = 0) -> "Featurizer":
        """Learn the TF-IDF vocabularies. A random sample of `max_docs` documents is enough (the vocabularies
        are capped at a few thousand n-grams) and keeps memory flat when the corpus has 10 000 documents."""
        if len(docs) > max_docs:
            idx = np.random.default_rng(seed).choice(len(docs), size=max_docs, replace=False)
            docs = [docs[i] for i in sorted(idx)]
        _, t = self._tables(docs)
        for key, vec in self._vectorizers():
            vec.fit(t[key])
        return self

    def transform(self, docs: list[dict]) -> sparse.csr_matrix:
        num, t = self._tables(docs)
        mats = [sparse.csr_matrix(num[:, self.num_cols])] + [vec.transform(t[key]) for key, vec in self._vectorizers()]
        return sparse.hstack(mats, format="csr", dtype=np.float32)

    def transform_chunks(self, docs: list[dict], chunk: int = 400):
        """Yield (first_doc_index, sparse matrix) per chunk of documents, so huge corpora never need one giant matrix."""
        for i in range(0, len(docs), chunk):
            yield i, self.transform(docs[i:i + chunk])

    def feature_names(self) -> list[str]:
        names = [NUM_FEATURE_NAMES[i] for i in self.num_cols]
        for tag, vec in self._vectorizers():
            names += [f"{tag}:{x}" for x in vec.get_feature_names_out()]
        return names
