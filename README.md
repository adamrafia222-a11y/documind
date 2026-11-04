# DocuMind AI backend - picture -> CSV

```
data/raw  ->  [1 prepare: pre-processing + OCR + labels]  ->  data/processed  ->  [2 train]  ->  models/
                                                                                                   |
image / PDF  ->  [3 API or CLI: pre-processing -> OCR -> LightGBM -> fields]  ->  CSV  <-----------+
```
LightGBM does not read pictures: it classifies each OCR word (vendor, date, total... or "O" = nothing).

## Folder structure
```
documind_backend/
├── data/
│   ├── raw/          original datasets (put the unzipped FATURA folder here, never modify)
│   ├── interim/      OCR cache (can be deleted at any time)
│   ├── processed/    corpus.jsonl = OCR words + labels, ready for training
│   └── toy/          tiny fake dataset to test the pipeline (generated)
├── documind/                     the code, one folder per stage
│   ├── config.py                 folders + settings (.env)
│   ├── preprocessing/            image.py (deskew, resize, shadows...), pdf.py, augment.py
│   ├── ocr/                      engine.py (Tesseract)
│   ├── features/                 layout.py (rows), builder.py (numeric + TF-IDF features)
│   ├── data/                     loader.py, labeling.py, corpus.py, prepare.py (STEP 1), inspect_dataset.py
│   ├── training/                 train.py (STEP 2), splits.py, evaluate.py
│   ├── inference/                extractor.py, grouping.py, export.py, cli.py
│   └── api/                      main.py (FastAPI)
├── models/           field_extractor.joblib (trained model)
├── reports/          training_report.json (F1 per field)
├── scripts/          make_toy_data.py
├── tests/            unit tests + end-to-end test
├── .env.example      copy to .env (Tesseract path, OCR language)
└── requirements.txt
```

## Install
1. Python 3.10+.
2. Tesseract OCR program  
   Windows: https://github.com/UB-Mannheim/tesseract/wiki (then put its path in `.env`)  
   Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-fra`
3. `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set `TESSERACT_CMD` (Windows).

## Run (all commands from the project root)
**Test with fake data first (3 minutes)**
```
python scripts/make_toy_data.py
python -m documind.data.prepare --data data/toy --per-template 12 --workers 2
python -m documind.training.train --baseline
```

**Real training with FATURA**
1. Download FATURA.zip (https://zenodo.org/records/10371464), unzip it into `data/raw/FATURA/`
2. Check that the labels are understood (OCRs 3 images, tests every box convention, saves overlay pictures in `reports/`):
   `python -m documind.data.inspect_dataset --data data/raw/FATURA --check`
3. STEP 1 prepare - **all invoices by default** (FATURA: 10 000 = 50 templates x 200):
   `python -m documind.data.prepare --data data/raw/FATURA --workers 4`
   * first run: about 35 min for 10 000 images with 4 workers (OCR is cached, later runs take seconds)
   * quicker trial: `--per-template 40` (2 000 images). `--per-template 0` (default) = everything
   * `--include-colored` also uses the coloured-background copy of every invoice (x2 images)
   * `--augment` adds phone-photo-like copies, `--drop-labels ""` keeps the TABLE class (dropped by default)
   * it must end with `Corpus saved`. If it stops with "No OCR word was matched", the corpus is NOT replaced:
     run step 2 and send `data/interim/alignment_debug.json`
4. STEP 2 train: `python -m documind.training.train --baseline`
   (built chunk by chunk: about 1.2 GB RAM for 10 000 documents)

The prepare step prints which box convention it detected (x,y,w,h / x0,y0,x1,y1 / 0-1000 / 0-1) and the number of words
per field. Check that every field you expect appears there before training.

**Check the data, then train (recommended order)**
```
C:\ai_env\Scripts\activate                      # the venv that has lightgbm / scikit-learn (NOT "base")
python -m documind.data.audit --data data/raw/FATURA --n 60   # are the labels on the right words?
python -m documind.training.train --position none          # or --position full ; the notebook compares both
```
or open `notebooks/train_documind.ipynb` (kernel = ai_env) and run it top to bottom: audit, position vs
no-position comparison, final model, per-template scores, save. A model trained with an older version of the
code must be retrained (the feature set changed).
`python scripts/make_synthetic_corpus.py` builds a fake multi-template corpus in seconds to test the code without OCR.

**Use the model**
```
uvicorn documind.api.main:app --reload --port 8000          # http://localhost:8000/docs
python -m documind.inference.cli invoice1.jpg scan.pdf -o result.csv
```
API: `POST /extract?format=json|csv` (one file), `POST /extract/batch` (many files -> one CSV), `GET /health`, `GET /model`.

**Tests**: `pytest -q` (unit tests) - the end-to-end test needs Tesseract.

## Why two steps (prepare, then train)?
OCR is slow. Step 1 runs it once and saves `data/processed/corpus.jsonl`; you can then retrain
and tune LightGBM many times in seconds without redoing OCR.

## Honest limits
* FATURA is synthetic and English: expect lower scores on real or French invoices. Keep 30-50 real
  anonymised documents as a final test; for French set `OCR_LANG=fra` and prepare/retrain on French data.
* Line items (tables) are not extracted, only header fields.
* If `--baseline` beats LightGBM, trust the numbers and report both.
