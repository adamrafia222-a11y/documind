"""STEP 2 - training: data/processed/corpus.jsonl  ->  models/field_extractor.joblib

    python -m documind.training.train --baseline
    python -m documind.training.train --position none          # no absolute page-position features

The same code is used by the notebook (notebooks/train_documind.ipynb): `fit_model(docs, TrainConfig(...))`.
Outputs: models/field_extractor.joblib and reports/training_report.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from ..config import BACKGROUND_LABEL, CORPUS_PATH, MODEL_PATH, MODELS_DIR, REPORTS_DIR
from ..data.corpus import load_corpus
from ..data.labels import merge_labels
from ..features import Featurizer
from .evaluate import evaluate, evaluate_fields, predict_docs
from .splits import build_matrix, labels_of, split_docs

warnings.filterwarnings("ignore", message=".*eval_set.*")


@dataclass
class TrainConfig:
    split: str = "template"          # "template" = validation/test layouts never seen in training
    position: str = "full"           # "full" | "none" (drop absolute page-position features)
    o_ratio: float = 4.0             # max background/field word ratio kept in training
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 40
    colsample: float = 0.3
    reg_lambda: float = 5.0
    extra_trees: bool = True         # random split thresholds: much less layout memorisation
    class_weight_power: float = 0.5  # 0 = none, 1 = fully balanced (the old setting), 0.5 = square-root balancing
    early_stopping: int = 50
    baseline: bool = False
    seed: int = 42
    merge_labels: bool = True        # GST(x%) -> GST, BILL_TO -> BUYER (same content, template-specific name)


def _sample_weights(y: np.ndarray, n_classes: int, power: float) -> np.ndarray | None:
    if power <= 0:
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (len(y) / (np.maximum(counts, 1.0) * np.count_nonzero(counts))) ** power
    return w[y].astype(np.float32)


def fit_model(docs: list[dict], cfg: TrainConfig, log=print, evaluate_test: bool = True) -> dict:
    """Split, featurise, train LightGBM, evaluate. Returns everything needed to save or inspect the model."""
    t0 = time.time()
    if cfg.merge_labels:
        docs = merge_labels(docs)
    train, val, test, how = split_docs(docs, cfg.split, cfg.seed)
    log(f"Split {how}: train={len(train)} val={len(val)} test={len(test)} documents")

    y_tr_txt = labels_of(train)
    classes = sorted(set(y_tr_txt) | {BACKGROUND_LABEL})
    to_id = {c: i for i, c in enumerate(classes)}
    n_bg = int((y_tr_txt == BACKGROUND_LABEL).sum())
    n_fld = len(y_tr_txt) - n_bg
    bg_keep = min(1.0, cfg.o_ratio * max(n_fld, 1) / max(n_bg, 1))
    del y_tr_txt

    feat = Featurizer(position=cfg.position).fit(train)
    X_fit, y_fit = build_matrix(feat, train, to_id, bg_keep=bg_keep, seed=cfg.seed)
    X_va, y_va = build_matrix(feat, val, to_id)
    X_te, y_te = build_matrix(feat, test, to_id)
    log(f"Feature matrix: {X_fit.shape[1]} columns, {X_fit.shape[0]} training rows "
        f"({n_fld} field words + background sub-sampled to {bg_keep:.0%}), {len(classes) - 1} fields")

    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves, colsample_bytree=cfg.colsample, subsample=0.8, subsample_freq=1,
        min_child_samples=cfg.min_child_samples, reg_lambda=cfg.reg_lambda, extra_trees=cfg.extra_trees,
        max_bin=63, n_jobs=-1, random_state=cfg.seed, verbose=-1,
    )
    callbacks = [lgb.log_evaluation(50)]
    fit_kwargs: dict = {"sample_weight": _sample_weights(y_fit, len(classes), cfg.class_weight_power)}
    if len(y_va):
        fit_kwargs.update(eval_set=[(X_va, y_va)], eval_metric="multi_logloss")
        callbacks.append(lgb.early_stopping(cfg.early_stopping, verbose=True))
    model.fit(X_fit, y_fit, callbacks=callbacks, **fit_kwargs)

    results: dict = {}
    if len(y_va):
        results["validation"] = evaluate("VALIDATION", model, X_va, y_va, classes, verbose=False)
    if len(y_te) and evaluate_test:
        results["test"] = evaluate("TEST (unseen documents)", model, X_te, y_te, classes, verbose=False)
        pred = predict_docs(feat, model, classes, test)
        results["test_fields"] = evaluate_fields(test, pred, classes, verbose=False)

    out = {"feat": feat, "model": model, "classes": classes, "results": results, "how": how,
           "train": train, "val": val, "test": test, "X_te": X_te, "y_te": y_te, "X_fit": X_fit, "y_fit": y_fit,
           "bg_keep": bg_keep, "seconds": time.time() - t0, "cfg": cfg}
    if cfg.baseline and len(y_te):
        scaler = MaxAbsScaler().fit(X_fit)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_fit), y_fit)

        class _Wrap:
            def predict(self, X):
                return lr.predict(scaler.transform(X))
        out["baseline"] = evaluate("BASELINE logistic regression (TEST)", _Wrap(), X_te, y_te, classes, verbose=False)
    return out


def save_model(fit: dict, corpus_meta: dict, n_docs: int, n_words: int) -> None:
    cfg: TrainConfig = fit["cfg"]
    model = fit["model"]
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"), "args": asdict(cfg), "split": fit["how"],
        "classes": fit["classes"], "corpus": corpus_meta, "n_documents": n_docs, "n_words": n_words,
        "best_iteration": int(model.best_iteration_ or cfg.n_estimators),
        "ocr_lang": corpus_meta.get("ocr_lang"), "ocr_psm": corpus_meta.get("ocr_psm"),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"featurizer": fit["feat"], "model": model, "classes": fit["classes"], "meta": meta},
                MODEL_PATH, compress=3)
    res = fit["results"]
    report = {"meta": meta,
              "macro_f1_fields": {k: v["macro_f1_fields"] for k, v in res.items() if "macro_f1_fields" in v},
              "field_level_test": res.get("test_fields"),
              "details": {k: v["report"] for k, v in res.items() if "report" in v}}
    (REPORTS_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TrainConfig()
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--split", choices=["template", "random"], default=d.split)
    ap.add_argument("--position", choices=["full", "none"], default=d.position)
    ap.add_argument("--o-ratio", type=float, default=d.o_ratio)
    ap.add_argument("--n-estimators", type=int, default=d.n_estimators)
    ap.add_argument("--learning-rate", type=float, default=d.learning_rate)
    ap.add_argument("--num-leaves", type=int, default=d.num_leaves)
    ap.add_argument("--min-child-samples", type=int, default=d.min_child_samples)
    ap.add_argument("--class-weight-power", type=float, default=d.class_weight_power)
    ap.add_argument("--no-extra-trees", action="store_true")
    ap.add_argument("--no-merge-labels", action="store_true", help="keep GST(x%%) and BILL_TO as separate classes")
    ap.add_argument("--baseline", action="store_true", help="also train a logistic-regression baseline")
    ap.add_argument("--seed", type=int, default=d.seed)
    a = ap.parse_args()
    cfg = TrainConfig(split=a.split, position=a.position, o_ratio=a.o_ratio, n_estimators=a.n_estimators,
                      learning_rate=a.learning_rate, num_leaves=a.num_leaves, min_child_samples=a.min_child_samples,
                      class_weight_power=a.class_weight_power, extra_trees=not a.no_extra_trees,
                      baseline=a.baseline, seed=a.seed, merge_labels=not a.no_merge_labels)

    docs, meta = load_corpus(a.corpus)
    n_words = sum(len(x["words"]) for x in docs)
    n_fields = sum(w["label"] != BACKGROUND_LABEL for x in docs for w in x["words"])
    print(f"Corpus: {meta.get('source', '?')} (built {meta.get('created_at', '?')})")
    if "toy" in str(meta.get("source", "")).lower():
        print("[warn] this corpus comes from the TOY dataset, not from FATURA.")
    print(f"{len(docs)} documents, {n_words} OCR words, {n_fields} field words")

    fit = fit_model(docs, cfg)
    res = fit["results"]
    for k, title in (("validation", "VALIDATION"), ("test", "TEST (unseen documents)")):
        if k in res:
            print(f"{title}: word-level macro-F1 on present fields = {res[k]['macro_f1_fields']:.4f}")
    if "test_fields" in res:
        print(f"TEST field-level: exact match {res['test_fields']['exact_match_mean']:.3f}, "
              f"token-F1 {res['test_fields']['token_f1_mean']:.3f}")
    if "baseline" in fit:
        print(f"BASELINE logistic regression (TEST): {fit['baseline']['macro_f1_fields']:.4f}")
    save_model(fit, meta, len(docs), n_words)
    names = fit["feat"].feature_names()
    imp = fit["model"].booster_.feature_importance(importance_type="gain")
    print("Top features:", ", ".join(names[i] for i in np.argsort(imp)[::-1][:15]))
    print(f"Model saved to {MODEL_PATH}  ({fit['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
