"""Train / validation / test splitting of the invoices (used by documind.data.prepare_invoices)."""
from __future__ import annotations

import numpy as np


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
