# data/
* `raw/`       original datasets (e.g. `raw/FATURA/`). Read only: never edit these files.
* `interim/`   OCR cache created by `documind.data.prepare` (safe to delete).
* `processed/` `corpus.jsonl` (+ `corpus.meta.json`): OCR words with labels, input of the training step.
* `toy/`       fake dataset from `scripts/make_toy_data.py`, only to test the pipeline.
