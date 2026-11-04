"""Train / validation / test splitting and class balancing."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..config import BACKGROUND_LABEL


def split_docs(docs: list[dict], mode: str, seed: int):
    """Template split = validation/test contain layouts never seen in training (honest score)."""
    rng = np.random.default_rng(seed)
    templates = sorted({d["template"] for d in docs})
    if mode == "template" and len(templates) >= 5:
        order = rng.permutation(len(templates))
        n_test = max(1, round(0.2 * len(templates)))
        n_val = max(1, round(0.15 * len(templates)))
        test_t = {templates[i] for i in order[:n_test]}
        val_t = {templates[i] for i in order[n_test:n_test + n_val]}
        train = [d for d in docs if d["template"] not in test_t | val_t]
        val = [d for d in docs if d["template"] in val_t]
        test = [d for d in docs if d["template"] in test_t]
        return train, val, test, f"by template ({len(templates)} templates)"
    if mode == "template":
        print(f"[warn] only {len(templates)} template(s) found -> falling back to a random split. "
              "Scores will be optimistic.")
    idx = rng.permutation(len(docs))
    n_test, n_val = int(0.15 * len(docs)), int(0.15 * len(docs))
    test = [docs[i] for i in idx[:n_test]]
    val = [docs[i] for i in idx[n_test:n_test + n_val]]
    train = [docs[i] for i in idx[n_test + n_val:]]
    return train, val, test, "random by document"


def labels_of(docs: list[dict]) -> np.ndarray:
    return np.array([w["label"] for d in docs for w in d["words"]])


def subsample_background(X, y: np.ndarray, classes: list[str], ratio: float, seed: int):
    """Keep every field word, but at most `ratio` x as many background words (speeds up training)."""
    bg = classes.index(BACKGROUND_LABEL)
    is_bg = y == bg
    n_fields = int((~is_bg).sum())
    keep_bg = int(ratio * max(n_fields, 1))
    if is_bg.sum() <= keep_bg:
        return X, y
    rng = np.random.default_rng(seed)
    bg_idx = np.flatnonzero(is_bg)
    chosen = rng.choice(bg_idx, size=keep_bg, replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(~is_bg), chosen]))
    return X[keep], y[keep]


def build_matrix(feat, docs: list[dict], to_id: dict[str, int], bg_keep: float = 1.0, seed: int = 0, chunk: int = 400):
    """Feature matrix + class ids, built chunk by chunk (flat memory on a 10 000-document corpus).

    * words whose class never appeared in training are dropped (to_id has no entry for them);
    * only `bg_keep` (0-1) of the background words are kept: field words are always kept.
    """
    rng = np.random.default_rng(seed)
    bg = to_id[BACKGROUND_LABEL]
    mats, ys = [], []
    for i, X in feat.transform_chunks(docs, chunk):
        txt = labels_of(docs[i:i + chunk])
        y = np.array([to_id.get(c, -1) for c in txt], dtype=np.int64)
        keep = y >= 0
        if bg_keep < 1.0:
            keep &= (y != bg) | (rng.random(len(y)) < bg_keep)
        mats.append(X[keep])
        ys.append(y[keep])
    if not mats:
        return sparse.csr_matrix((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return sparse.vstack(mats, format="csr", dtype=np.float32), np.concatenate(ys)
