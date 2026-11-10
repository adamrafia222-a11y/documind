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
API: `POST /extract?format=json|csv&doc_type=auto|invoice|receipt` (one file), `POST /extract/batch` (many files -> one CSV per document type), `GET /health`, `GET /model`.

**Tests**: `pytest -q` (unit tests) - the end-to-end test needs Tesseract.

### Receipts (YOLO detector)

Data: made by `scripts/generate_receipts.py` (your generator; 13 classes, boxes on the VALUES, exact labels + `ground_truth.jsonl`):
store_name, address, phone, date, time, receipt_number, item_name, item_qty, item_price, subtotal, tax, total, payment_method.
```
pip install -r requirements-receipts.txt                  # once, in the environment with the GPU build of PyTorch
python scripts/generate_receipts.py --n 10000 --out data/raw/receipts            # 8000 train / 1000 val / 1000 test  (--currency EUR, --seed, --preview 10)
python -m documind.data.prepare_receipts                   # STEP 1: reads data/raw/receipts (images/<split> layout and its data.yaml, or <split>/images),
                                                           #   checks labels, finds leaks -> data/processed/receipts_yolo/ , reports/receipts_preview/ (LOOK at it)
python -m documind.training.train_receipts                 # STEP 2: train 100 epochs (no early stop, best epoch kept) + score val and test -> models/receipts_yolo.pt
                                                           #   scores: precision, recall, mAP and found_rate in reports/receipts_report.json,
                                                           #   then a verdict against --target 0.90 (printed, never hidden)
python -m documind.inference.receipts ticket.jpg           # STEP 3 (command line): every field + item rows (qty, name, price)
```
or `notebooks/train_receipts_yolo.ipynb`. Tests: `pytest -q tests/test_receipts.py tests/test_router_service.py`.
A model trained on the older 5 classes (store / date / total / receipt_no / item) still loads, but models trained on the 13 classes are needed for the new fields:
retrain once. `receipts.csv` columns: file, STORE, ADDRESS, PHONE, DATE, TIME, RECEIPT_NO, SUBTOTAL, TAX, TOTAL, PAYMENT, N_ITEMS, ITEMS.

### Cheques (YOLO detector + checks)

Data: made by `scripts/generate_cheques.py` (10 classes, boxes on the values, exact labels + `ground_truth.jsonl`):
bank_name, cheque_number, amount_words, payee, amount_digits, place, date, signature, account_number, micr_line.
```
python scripts/generate_cheques.py --n 10000 --out data/raw/cheques             # 8000 train / 1000 val / 1000 test  (--currency EUR, --fonts-dir ./handwriting_fonts)
python -m documind.data.prepare_cheques                    # STEP 1: data/raw/cheques (else data/raw/cheque_dataset) -> data/processed/cheques_yolo
                                                           #   same checks as the receipts, reports/cheques_prepare_report.txt + reports/cheques_preview/ (LOOK at it)
python -m documind.training.train_cheques                  # STEP 2: yolo11s, 60 epochs (early stop 15), imgsz 1280, batch 4 -> models/cheques_yolo.pt
                                                           #   precision / recall / mAP50 / found_rate per class on val AND test (reports/cheques_report.json) + the 0.90 verdict
python -m documind.inference.cheques cheque.jpg            # STEP 3: fields + checks + the CSV row
```
Notebook version of the same steps (GPU check, settings, data look, training, per-class scores on val and test, predictions, the checks on 30 test cheques): `notebooks/train_cheques_yolo.ipynb`.

`prepare_cheques` / `train_cheques` are thin: they reuse the receipt code (`prepare_receipts.run`, `train_receipts.run_cli` with a different `Profile`),
so both types get the same validation, scores and target check, and neither overwrites the other's reports or weights.

`cheques.csv` columns: file, BANK, CHEQUE_NO, PAYEE, AMOUNT, CURRENCY, AMOUNT_WORDS, PLACE, DATE, ACCOUNT_NO, MICR, SIGNATURE, AMOUNT_CHECK.
Checks (shown as warnings on the web page, never silently corrected): amount in figures = amount in words (the French words are converted back to a number),
cheque number = the one in the MICR line, valid date, account number length (24 digits RIB / 23 IBAN), signature found.
On a sample of generated cheques (true boxes, Tesseract): bank, cheque number, account number and signature 40/40, amount in figures 39/40, date 39/40, payee 35/40,
amount in words understood on 31/40 (the checker never said "mismatch" for a consistent cheque, and never said "ok" for a wrong amount). Tesseract reads real
cursive handwriting badly: to read real cheques replace `read_text` in `documind/inference/cheques.py` by a handwriting model (e.g. TrOCR).

### Several document types in one app (invoice + receipt + cheque)
Each type keeps its own model and its own CSV columns; they are never merged.
```
picture / PDF -> router (page shape + key words, or the type chosen in the web page) -> invoice extractor (LightGBM)  -> invoices.csv
                                                                                    -> receipt extractor (YOLO+OCR) -> receipts.csv
                                                                                    -> cheque extractor (YOLO+OCR+checks) -> cheques.csv
```
* `POST /extract?doc_type=auto|invoice|receipt|cheque` (default `auto`); the answer has `doc_type`, `doc_type_source` (auto / manual), `columns`.
* `POST /extract/batch` -> one CSV per type: a plain `invoices.csv` / `receipts.csv` / `cheques.csv` if the batch has one type, `extractions.zip` with one csv per type if it has several.
* Web page: "Document type" selector (Auto-detect by default), a type badge on every document, "Read as ... instead" if the
  router guessed wrong, and `CSV (per type)` downloads `invoices.csv`, `receipts.csv` and `cheques.csv` separately. History and the dataset export
  (`dataset_invoices.csv`, `dataset_receipts.csv`, `dataset_cheques.csv`) are split the same way.
* A type whose model file is missing is simply not offered: with only the invoice model installed, everything goes to the invoice extractor.
* New type later (ID card, bank statement ...): write an extractor that returns the same kind of dict and add it in `documind/inference/service.py`.

## Why two steps (prepare, then train)?
OCR is slow. Step 1 runs it once and saves `data/processed/corpus.jsonl`; you can then retrain
and tune LightGBM many times in seconds without redoing OCR.

## Honest limits
* FATURA is synthetic and English: expect lower scores on real or French invoices. Keep 30-50 real
  anonymised documents as a final test; for French set `OCR_LANG=fra` and prepare/retrain on French data.
* Line items (tables) are not extracted, only header fields.
* If `--baseline` beats LightGBM, trust the numbers and report both.
