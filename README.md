# DocuMind AI backend - picture -> CSV

```
data/raw  ->  [1 prepare: annotations -> YOLO labels]  ->  data/processed  ->  [2 train YOLO]  ->  models/
                                                                                                   |
image / PDF  ->  [3 API or CLI: pre-processing -> YOLO boxes -> OCR of each box -> fields]  ->  CSV <-+
```
Every document type works the same way: a YOLO detector draws one box on the VALUE of each field, then each box is read by the OCR engines.
The invoice line items come from a rules reader (documind.inference.postprocess).

## Folder structure
```
documind_backend/
├── data/
│   ├── raw/          original datasets (put the unzipped FATURA folder here, never modify)
│   ├── interim/      OCR cache (can be deleted at any time)
│   ├── processed/    the cleaned YOLO datasets (invoices_yolo, receipts_yolo ...)
│   └── toy/          tiny fake dataset to test the pipeline (generated)
├── documind/                     the code, one folder per stage
│   ├── config.py                 folders + settings (.env)
│   ├── preprocessing/            image.py (deskew, resize, shadows...), canonical.py (picture reconstruction), pdf.py, augment.py
│   ├── ocr/                      engine.py (Tesseract), trocr.py, handwriting.py, ensemble.py, reconstruct.py
│   ├── features/                 layout.py (rows), patterns.py (dates, amounts)
│   ├── data/                     loader.py, labeling.py, prepare_invoices.py (STEP 1), inspect_dataset.py ...
│   ├── training/                 train_invoices.py (STEP 2), splits.py, compare_reconstruct.py ...
│   ├── inference/                extractor.py, invoice_yolo.py, postprocess.py (line items), export.py, cli.py
│   └── api/                      main.py (FastAPI)
├── models/           invoices_yolo.pt, receipts_yolo.pt, cheques_yolo.pt, contracts_yolo.pt
├── reports/          invoices_report.json (scores per field) ...
├── scripts/          make_toy_data.py
├── tests/            unit tests
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
**Invoices with FATURA (YOLO detector)**
1. Download FATURA.zip (https://zenodo.org/records/10371464), unzip it into `data/raw/FATURA/`
2. Check that the labels are understood (tests every box convention, saves overlay pictures in `reports/`):
   `python -m documind.data.inspect_dataset --data data/raw/FATURA --check`
3. `pip install -r requirements.txt` (once, in the environment with the GPU build of PyTorch)
4. STEP 1 prepare: `python -m documind.data.prepare_invoices --fatura data/raw/FATURA`
   * FATURA annotations become YOLO label files, split by TEMPLATE (`--split random` = by invoice); `--max-images 200` for a quick trial
   * `--photo-copies 2` adds 2 phone-photo-like copies (uneven light, blur, noise, JPEG, lost resolution) of every TRAINING page, same boxes: use it when your real invoices are photos
     and the confidence is low (the model learned clean renders). Val and test stay untouched, so the scores stay honest
   * `--include-colored` also uses the coloured-background copy of every invoice; `--keep-table` keeps the TABLE box as a class
   * then the shared checks run: look at `reports/invoices_preview/*.png` (boxes drawn on the pictures) and `reports/invoices_prepare_report.txt`
5. STEP 2 train: `python -m documind.training.train_invoices` (about 60 epochs; writes `models/invoices_yolo.pt` and `reports/invoices_report.json`)
6. STEP 3 try: `python -m documind.inference.invoice_yolo invoice.jpg`

**Test the code with fake data first (no GPU)**
```
python scripts/make_toy_data.py
python -m documind.data.prepare_invoices --fatura data/toy --raw-out data/raw/invoices_toy --out data/processed/invoices_toy --workers 1
```

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
pip install -r requirements.txt                  # once, in the environment with the GPU build of PyTorch
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

### Loan contracts (YOLO detector + checks)

Data: made by `scripts/generate_contracts.py` (19 classes, one A4 page per contract: personal loan, car loan, mortgage; boxes on the values; exact labels + `ground_truth.jsonl`):
title, contract_number, contract_date, lender_name, borrower_name, borrower_id, borrower_address, amount, amount_words, interest_rate, duration, monthly_payment,
first_payment_date, iban, place, signing_date, signature_lender, signature_borrower, stamp.
```
python scripts/generate_contracts.py --n 10000 --out data/raw/contracts          # 8000 train / 1000 val / 1000 test  (--currency EUR, --type car)
python -m documind.data.prepare_contracts                  # STEP 1: data/raw/contracts (else data/raw/contract_dataset) -> data/processed/contracts_yolo
                                                           #   same checks as receipts and cheques, reports/contracts_prepare_report.txt + reports/contracts_preview/ (LOOK at it)
python -m documind.training.train_contracts                # STEP 2: yolo11s, 60 epochs (early stop 15), imgsz 1280, batch 4, no mosaic -> models/contracts_yolo.pt
                                                           #   precision / recall / mAP50 / found_rate per class on val AND test (reports/contracts_report.json) + the 0.90 verdict
python -m documind.inference.contracts contract.jpg        # STEP 3: fields + checks + the CSV row
```
Notebook version: `notebooks/train_contracts_yolo.ipynb`. `prepare_contracts` / `train_contracts` reuse the receipt code (`prepare_receipts.run`, `train_receipts.run_cli` with a
`Profile`), so every type gets the same validation, scores and target check, and none overwrites another's reports or weights.

A contract is running text, so **several boxes of one class per page are normal**: the lender's name is printed twice (letterhead + first paragraph) and a value that does not
fit at the end of a line wraps onto the next one (one box per line). The reader puts the boxes of a field back in reading order (the page tilt is measured and removed first,
`documind/ocr/lines.py`) and joins their texts; the most confident copy of the lender's name is kept.

`contracts.csv` columns: file, CONTRACT_TYPE, TITLE, CONTRACT_NO, CONTRACT_DATE, LENDER, BORROWER, BORROWER_ID, BORROWER_ADDRESS, AMOUNT, CURRENCY, AMOUNT_WORDS, RATE,
DURATION_MONTHS, MONTHLY_PAYMENT, FIRST_PAYMENT_DATE, IBAN, PLACE, SIGNING_DATE, SIGNED_LENDER, SIGNED_BORROWER, STAMP, AMOUNT_CHECK, PAYMENT_CHECK.
Checks (shown as warnings, never silently corrected): amount in figures = amount in words (French words converted back to a number, millions included); monthly payment = the
loan formula for the amount, the rate and the duration (within 1 %); real dates, first payment after the contract date, signing date not before it; IBAN length (MA 28, FR 27);
both signatures found. On generated contracts (true boxes, Tesseract, 16 pages) amount, duration, monthly payment, currency and type were read 16/16, rate 15/16,
borrower 15/16, IBAN 12/16 (long digit strings: the IBAN length check and the payment formula caught the slips). The router recognises a contract page from its French words
(contrat, prêteur, emprunteur, soussignés, TAEG ...).

### OCR (train the text reader on your datasets)

The detectors say WHERE a value is, the OCR says WHAT is written there. Tesseract is general-purpose: it drops accents, mixes 0 / 9 / 8, loses
capitals and handwriting. This module fine-tunes a recognition model (TrOCR) on text-line crops cut out of **receipts**, **cheques**, **contracts** and **FATURA**.
```
pip install -r requirements.txt                         # transformers, sentencepiece, accelerate (next to your GPU build of PyTorch)
python -m documind.data.prepare_ocr                         # STEP 1: crops + transcriptions -> data/processed/ocr_crops ; prints Tesseract's score (the number to beat)
python -m documind.training.train_ocr --smoke               # 1 minute: checks that everything runs
python -m documind.training.train_ocr                       # STEP 2: fine-tune microsoft/trocr-small-printed, 4 epochs, best epoch -> models/ocr_trocr
python -m documind.training.evaluate_ocr                    # STEP 3: Tesseract vs the new model per source and field -> reports/ocr_report.json
```
Notebook version: `notebooks/train_ocr_trocr.ipynb`. Options: `--base microsoft/trocr-base-handwritten --batch 8` (handwriting of the cheques),
`--sources cheques`, `--epochs 6`. Dataset folders are found automatically (`data/raw/receipts|receipt_dataset`, `cheques|cheque_dataset`, `contracts|contract_dataset`, `FATURA`);
`--fatura-mode xywh|xyxy` if the FATURA crops of `reports/ocr_crops_preview.png` look shifted.

* Labels: the receipt and cheque transcriptions come from the generators' `ground_truth.jsonl` + the box of every field (checked against Tesseract:
  field by field the label matches what is printed; the cheque number is printed `N° 1234567`, the account number in groups of 4); FATURA uses the
  transcription stored with each annotated box. Splits: the generators' own, FATURA by invoice.
* Using it: nothing to change. When `models/ocr_trocr` exists, **both engines read every box** of the receipt, cheque and contract readers and the better reading is
  kept (`documind/ocr/ensemble.py`). The rules, in order: only one engine returned text -> that one; both read the same -> that; a cross-check accepts only one (the cheque's
  amount in words must be the amount in figures) -> that one; the format of the field (a date must be a date, an amount must look like an amount, a phone number has 10
  digits, an IBAN the length of its country, the amount in words must be understood as a number) clearly fits one reading better -> that one; otherwise the trained model
  decides (`DOCUMIND_OCR_PRIMARY=tesseract` to let Tesseract decide). When both look fine and differ the checker cannot know which is true, it only rejects readings that
  cannot be right. `DOCUMIND_OCR=auto` (default, Tesseract alone if no model) | `both` (error if the model is missing) | `trocr` | `tesseract`.
  For invoices Tesseract still finds the words; `DOCUMIND_OCR_REFINE=lowconf` lets the trained model re-read the doubtful ones and the same arbitration keeps the better
  reading (off by default; check the invoice scores before keeping it, the classifier was trained on Tesseract text). `documind.ocr.ensemble.summary()` counts which rule decided.
* Where the gain is: on generated crops Tesseract (with each crop scaled to ~40 px high) already has a CER of about 2 % on receipts and 4 % on cheques; its mistakes are
  accents and capitals, the long handwritten amount in words, the MICR line and single digits. Run `prepare_ocr` (it prints these numbers for YOUR data) and
  `evaluate_ocr` to see what the training really changes.
* Careful: val / test crops come from the same generators as the training crops, so a very low error says the model learned these documents. Try real pictures.

### Invoices (YOLO detector, like the other types)

A YOLO detector draws one box on the VALUE of each field (TOTAL, DATE, NUMBER, SELLER_ADDRESS ...), then every box is read by the OCR engines (Tesseract + the OCR model +
the handwriting model, the better reading is kept). The line items come from the rules reader (the table is not a detector class by default). The commands are in "Run" above.
The upload, the history and the CSV use `models/invoices_yolo.pt` as soon as it exists (`INVOICE_MODEL_PATH` in `.env` to change the file).
FATURA has a few dozen layouts; with the default template split, val and test are layouts never seen in training (the honest score).
Measure the picture reconstruction on your files: `python -m documind.training.compare_reconstruct --data data/raw/FATURA --n 60 --photo`.

**Fields the detector misses, and checks** (`documind/inference/fieldrules.py`, `invoice_checks.py`)
The detector learned the FATURA templates. On another layout or language it finds only some fields, so the labels printed on the page ("Date d'échéance :", "Bill To", "Net à payer",
"Arrêtée la présente facture à la somme de ...") are read in the page's words, in French and English, to fill the fields it did not find (marked `source: rules` and a lower confidence;
a detector box is never overridden, and nothing is invented when there is no label). Then the invoice gets `checks` like the cheques and contracts, shown on the Read page and in the API result:
`total_arithmetic` (subtotal + tax - discount = total), `items_sum`, `rows_arithmetic` (quantity x unit price (+ tax) = amount on every row), `tax_rate_consistent`, `dates_valid`,
`due_after_issue`, `amount_in_words` (French or English), `amount_due_vs_total`, `required_fields`, `email_valid`, `iban_valid`. True / False / not checked; a failed check becomes a warning
and the CSV has a `CHECKS` column ("7/9 passed; failed: ..."). The confidence of a field found by rules is not a fixed number: it comes from the OCR confidence of its words, the validity of its format (a real date, a well-formed e-mail ...) and the checks (a passing check raises the fields it confirms, a failing one lowers them): `documind/inference/confidence.py`. The line-item reader also accepts a ruled table whose header is white on a dark band and whose page has a frame or a shadow.

### OCR reconstruction (picture fixed on every upload, no retraining)

When a picture is uploaded it is rebuilt in the form the models were trained on (clean, upright, flat page, clean words) BEFORE the models read it.
The dataset and the trained models are not touched.

| Stage | Where | What it does |
|---|---|---|
| A. picture | `documind/preprocessing/canonical.py` | cuts the page out, turns it upright (only when Tesseract is sure), flattens uneven light; boxes are mapped back to the original upload |
| B. words | `documind/ocr/reconstruct.py` | a pretrained OCR model (`microsoft/trocr-base-printed`, or your `models/ocr_trocr` if you trained it) re-reads the doubtful chunks Tesseract found (same `ensemble.choose` rules), then one canonical spelling (`2O25` -> `2025`) |
| C. preview | `render_clean()` | redraws the words on a clean page, to look at what was read |

The first upload downloads the pretrained model once (~1.3 GB, needs `pip install -r requirements.txt`); then it is cached. Without torch or internet the step falls back to Tesseract alone with a warning.

```powershell
python -m documind.ocr.reconstruct photo.jpg --doc receipt      # look at it: reports/reconstruct has the rectified page, the rebuilt page, words.json
```

`.env`: `DOCUMIND_RECONSTRUCT=auto|on|off` (default `auto`: stage A only; stage B is opt-in because on 60 FATURA invoices it scored 33.1 % against 36.1 % without), `RECONSTRUCT_OCR_MODEL` (`microsoft/trocr-small-printed` is 3x faster), `DOCUMIND_PRETRAINED_OCR=off` to forbid the download.
Optional: `prepare --reconstruct` then `train` makes the invoice classifier learn from the same kind of words; you do not need it.
Contracts get the same two stages (stage B only with `DOCUMIND_RECONSTRUCT=on`): the page is straightened, and the full-page text reader (`contract_text.read_contract_text`) re-reads the doubtful words with the OCR model before it looks for the amount, rate, duration and dates in the sentences. Nothing to retrain.
Contracts: `python -m documind.training.compare_reconstruct --type contract --data data/raw/contracts --n 60 --photo`.
Measure the gain on your own files: `python -m documind.training.compare_reconstruct --data data/raw/FATURA --n 60 --photo` (runs the real pipeline three ways: off, auto = picture only, on = picture + words).

### Front end with Streamlit (sign in, dashboard, history, profile)

```
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py                     # opens http://localhost:8501
```
A dark, animated interface (gradient background, a document being scanned on the sign-in page, count-up numbers, confidence rings, a reading animation; CSS only: no CDN, no font to
download, `prefers-reduced-motion` switches it off): drifting blurred light behind everything, shimmering titles, a sheen sweeping over the main buttons, a rotating word
(invoice / receipt / cheque / contract) in the sign-in headline, confetti when documents were read, cards and lists that slide in one after the other, and on the dashboard numbers that
count up, lines that draw themselves, donut segments that sweep round, bars that grow or rise and a heat map that fades in diagonally. Pages:

* **Sign in / Create account / Continue as a guest.** Accounts use the same SQLite database as the API (`data/documind.db`, salted PBKDF2 hashes, session tokens), so the API and this
  page share the accounts. A guest can read documents but nothing is saved.
* **Dashboard** (the landing page after signing in). Four animated KPI cards (documents in the last 7 / 30 / 90 days with a sparkline and the change against the previous period,
  average confidence, average time, share of documents that need a look), documents per day stacked by type (interactive: tooltips), share by type (donut), how sure the readings were
  (histogram, red / amber / green), the fields read with the least confidence, warning rate and average time by type, the checks that fail most (with plain names such as "Amount in
  figures ≠ amount in words"), a weekday × hour heat map of when you read documents, and the latest documents (open one in the History). Statistics come from `GET /stats`.
* **Read documents.** Drop invoices, receipts, cheques and contracts (JPG, PNG, PDF, several at once): the type is found (or chosen), the fields are drawn on the picture (a value wrapped
  over two lines gets one box per line), the checks and warnings are listed, and you get **one CSV per document type** (`invoices.csv`, `receipts.csv`, `cheques.csv`, `contracts.csv`,
  never merged) plus a zip. "Read as ..." buttons read a document again as another type. One bad file does not stop the others.
* **History.** Every reading of your account, newest first: numbers, filters (type, file name, only with warnings), open a document to see it again with its stored picture and boxes, delete
  it (the picture too), export the filtered documents as CSV (one file per type).
* **Profile.** Avatar, member since, statistics (documents, average confidence and time, how many had warnings, most read type, activity of the last 30 days, split by type), edit the name,
  change the password (the other sessions are signed out), download all your data as a zip, delete the account (password required; history and pictures are removed).

Where it runs (sign-in page, "Connection"): **This computer** loads the models and the accounts inside the Streamlit process (nothing else to start), **API server** talks to the FastAPI
server (`uvicorn documind.api.main:app`). New API endpoints for this: `PATCH /auth/me`, `POST /auth/password`, `POST /auth/delete`, `GET /stats`; `/history` also returns `n_warnings`.
Files: `streamlit_app.py`, `documind/ui/views.py` (the pages), `documind/ui/theme.py` (the look), `documind/ui/backend.py` (the two backends, same behaviour: one test scenario runs on both),
`documind/ui/helpers.py` (reading, drawing, CSV, history filters), `.streamlit/config.toml` (dark theme, 50 MB upload limit).
Limits: you stay signed in only as long as the browser tab session lives (reloading the page asks for the password again); the look was checked by tests (well-formed HTML, escaped text,
every animation defined), not by eye in every browser.

### Several document types in one app (invoice + receipt + cheque + contract)
Each type keeps its own model and its own CSV columns; they are never merged.
```
picture / PDF -> router (page shape + key words, or the type chosen in the web page) -> invoice extractor (YOLO+OCR+rules) -> invoices.csv
                                                                                    -> receipt extractor (YOLO+OCR) -> receipts.csv
                                                                                    -> cheque extractor (YOLO+OCR+checks) -> cheques.csv
                                                                                    -> contract extractor (YOLO+OCR+checks) -> contracts.csv
```
* `POST /extract?doc_type=auto|invoice|receipt|cheque|contract` (default `auto`); the answer has `doc_type`, `doc_type_source` (auto / manual), `columns`.
* `POST /extract/batch` -> one CSV per type: a plain `invoices.csv` / `receipts.csv` / `cheques.csv` if the batch has one type, `extractions.zip` with one csv per type if it has several.
* Web page: "Document type" selector (Auto-detect by default), a type badge on every document, "Read as ... instead" if the
  router guessed wrong, and `CSV (per type)` downloads `invoices.csv`, `receipts.csv`, `cheques.csv` and `contracts.csv` separately. History and the dataset export
  (`dataset_invoices.csv`, `dataset_receipts.csv`, `dataset_cheques.csv`, `dataset_contracts.csv`) are split the same way.
* A type whose model file is missing is simply not offered: with only the invoice model installed, everything goes to the invoice extractor.
* New type later (ID card, bank statement ...): write an extractor that returns the same kind of dict and add it in `documind/inference/service.py`.

## Why two steps (prepare, then train)?
Step 1 turns the annotation files into clean YOLO datasets and checks them once (labels, duplicates, pictures found in two splits); you can then retrain
the detector with other settings without redoing it.

## Honest limits
* FATURA is synthetic and English: expect lower scores on real or French invoices. Keep 30-50 real
  anonymised documents as a final test; for French set `OCR_LANG=fra` and prepare/retrain on French data.
* The detector ignores the table on purpose (the TABLE class is dropped in training). Line items come from a separate rules reader (`USE_RULES=1`, the default): it finds the table by its header or by its rows of amounts. It can miss a table on a blurry or very small picture; the result then has `items: []` and `N_ITEMS` = 0.
