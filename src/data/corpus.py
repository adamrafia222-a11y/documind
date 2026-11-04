"""Read / write the processed corpus (data/processed/corpus.jsonl): one document per line."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import CORPUS_PATH


def meta_path(path: Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def save_corpus(docs: list[dict], meta: dict, path: Path = CORPUS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    meta_path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_corpus(path: Path = CORPUS_PATH) -> tuple[list[dict], dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first: python -m documind.data.prepare --data data/raw/FATURA")
    with path.open(encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh if line.strip()]
    mp = meta_path(path)
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    return docs, meta
