# DocuMind AI - step by step: generate, prepare, train

This guide explains **each step** to build the models of the project, in the order you run them, with the exact command for each one.
All commands are for **Windows PowerShell**, run **from the project folder**, in the environment that has the **GPU build of PyTorch**.
(Every command also accepts `--help`, which lists all its options.)

---------------------------------------------------------------------------------------------------

## 0. The big picture

DocuMind reads four kinds of documents. Each one has its own detector, built the same way:

```
 STEP 1  GET THE DATA        STEP 2  PREPARE (pre-treatment)       STEP 3  TRAIN            STEP 4  USE
 ----------------------      ----------------------------------    ---------------------    ------------------
 generate  (receipts,        check the labels, remove duplicates,   YOLO learns to draw       upload a picture:
 cheques, contracts)         resize, split train / val / test,      one box on each field     boxes -> OCR of each
 or download (invoices)      write the YOLO folders                 -> models/*.pt            box -> fields + checks
        |                              |                                    |
   data/raw/...               data/processed/..._yolo               models/..._yolo.pt  +  reports/..._report.json
```

| Document | Step 1 (data) | Step 2 (prepare) | Step 3 (train) | Model file |
|---|---|---|---|---|
| **Invoice** | download **FATURA** (real annotated dataset) | `prepare_invoices` | `train_invoices` | `models/invoices_yolo.pt` |
| **Receipt** | `scripts/generate_receipts.py` | `prepare_receipts` | `train_receipts` | `models/receipts_yolo.pt` |
| **Cheque** | `scripts/generate_cheques.py` | `prepare_cheques` | `train_cheques` | `models/cheques_yolo.pt` |
| **Contract** | `scripts/generate_contracts.py` | `prepare_contracts` | `train_contracts` | `models/contracts_yolo.pt` |
| **OCR (optional)** | crops cut from the datasets above | `prepare_ocr` | `train_ocr` | `models/ocr_trocr/` |

You do **not** need to train everything. Train only the document types you want to read. The OCR model is optional: without it the project uses
Tesseract and the **pretrained** OCR model (downloaded automatically).

---------------------------------------------------------------------------------------------------

## 1. One-time setup

```powershell
conda activate pytorch_gpu                                   # your environment with the GPU build of PyTorch
pip install -r requirements.txt                              # YOLO (ultralytics), OCR (transformers), API, UI ...
copy .env.example .env                                       # then edit .env (see below)
```

**Check that PyTorch sees your GPU** (training on CPU works but is very slow):
```powershell
python -c "import torch; print(torch.__version__, 'GPU:', torch.cuda.is_available())"
```
It must print `GPU: True`.

**Tesseract** (the text reader) must be installed, with the English and French language files. Check:
```powershell
tesseract --version
tesseract --list-langs
```
If `tesseract` is "not recognized", put its path in `.env`: `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.
For French documents the language line is `OCR_LANG=eng+fra`.

**Download the handwriting model once** (used for the cheque amount, the payee, handwritten notes; about 1.3 GB, then cached):
```powershell
python -m documind.ocr.handwriting
```

---------------------------------------------------------------------------------------------------

## 2. Folders - where things go

```
data/raw/           STEP 1 writes here   (original datasets: never edit them)
data/processed/     STEP 2 writes here   (cleaned YOLO datasets: <name>_yolo/{train,val,test}/{images,labels} + data.yaml)
models/             STEP 3 writes here   (the trained .pt files)
reports/            every step writes its report here (scores, previews, curves)
```

Quick rule: **look at `reports/<name>_preview/` after step 2** (pictures with their boxes drawn) **and at `reports/<name>_report.json` after step 3** (scores).

---------------------------------------------------------------------------------------------------

## 3. Test everything in minutes first (recommended)

Before a long training, run the whole chain on a **tiny** dataset to be sure each command works. Example with receipts:

```powershell
python scripts/generate_receipts.py --n 50 --out data/raw/receipts --preview 5
python -m documind.data.prepare_receipts --data data/raw/receipts --workers 1
python -m documind.training.train_receipts --fraction 0.5 --epochs 3
```
If these three run without error, the real run will too. Then delete the test data (`data\raw\receipts`, `data\processed\receipts_yolo`) and generate the real one.

---------------------------------------------------------------------------------------------------

## 4. INVOICES  (FATURA dataset  ->  `models/invoices_yolo.pt`)

Invoices use a **real** annotated dataset, FATURA, instead of a generator.

### Step 1 - get the data
Download **FATURA** from https://zenodo.org/records/10371464, unzip it into `data/raw/FATURA/`.
Check that the project understands the files and the box coordinates:
```powershell
python -m documind.data.inspect_dataset --data data/raw/FATURA --check
```
(It lists the files and label names, reads 3 images and tests every coordinate convention.)

### Step 2 - prepare (pre-treatment)
```powershell
python -m documind.data.prepare_invoices --fatura data/raw/FATURA --photo-copies 2
```
What it does, in order:
1. reads the FATURA annotation files and converts them into **YOLO label files** (one box per field value),
2. merges label variants (`GST(5%)` -> `GST`, `BILL_TO` -> `BUYER`) and drops `TABLE` (the line items come from a separate rules reader),
3. **splits by template** (validation and test are layouts never seen in training - the honest score),
4. adds **2 phone-photo-like copies of every training page** (uneven light, blur, noise, JPEG, lost resolution; same boxes) so the detector also knows photos,
5. runs the shared checks (bad labels, duplicates, the same picture in two splits) and writes the YOLO folders.

Useful options: `--split random` (split by invoice instead), `--include-colored` (use the coloured-background copies too), `--keep-table`
(keep TABLE as a class), `--max-images 200` (quick test), `--workers 4`.

**Look at** `reports/invoices_preview/*.png`: the boxes must sit **on the values**. Also read `reports/invoices_prepare_report.txt`.

### Step 3 - train
```powershell
python -m documind.training.train_invoices
```
Defaults: `yolo11s.pt`, 60 epochs, `--imgsz 1280`, `--batch 4`, stops early after 15 epochs without progress.
Output: `models/invoices_yolo.pt` and `reports/invoices_report.json` (precision, recall, mAP50 and found-rate per class, on validation **and** test).

Common variations:
```powershell
python -m documind.training.train_invoices --imgsz 1536 --batch 2          # more pixels for small text (slower)
python -m documind.training.train_invoices --model yolo11m.pt --batch 2    # bigger model, more accurate (slower)
python -m documind.training.train_invoices --fraction 0.2 --epochs 10      # quick test run
python -m documind.training.train_invoices --eval-only                     # only re-score the existing model
```

### Step 4 - try it
```powershell
python -m documind.inference.invoice_yolo path\to\invoice.jpg              # every field found, with its confidence
python -m documind.inference.invoice_yolo path\to\invoice.jpg --conf 0.05  # show even the weak detections
```

---------------------------------------------------------------------------------------------------

## 5. RECEIPTS  (`models/receipts_yolo.pt`)

### Step 1 - generate the data
```powershell
python scripts/generate_receipts.py --n 10000 --out data/raw/receipts
```
Makes realistic receipt photos (thermal paper, torn edges, shadow, rotation, blur, noise) **with their labels**. 13 classes: store_name, address, phone, date,
time, receipt_number, item_name, item_qty, item_price, subtotal, tax, total, payment_method.
Options: `--currency MAD|EUR`, `--seed 42`, `--workers 4`, `--preview 10` (saves 10 pictures with boxes drawn, to check).
Output in the folder you gave: `images/<split>/`, `labels/<split>/` (YOLO), `annotations/`, `ground_truth.jsonl`, `data.yaml`.

### Step 2 - prepare
```powershell
python -m documind.data.prepare_receipts --data data/raw/receipts --workers 4
```
Checks every label, removes duplicates and tiny boxes, applies the EXIF rotation, resizes (`--max-side`, default 1100) and writes `data/processed/receipts_yolo`.
**Look at** `reports/receipts_preview/` and `reports/receipts_prepare_report.txt`.

### Step 3 - train
```powershell
python -m documind.training.train_receipts
```
Defaults: `yolo11s.pt`, 100 epochs, `--imgsz 960` (use 1280 for thin boxes), `--batch 8`. Output: `models/receipts_yolo.pt`, `reports/receipts_report.json`.

If the score is low because some labels are wrong, find them:
```powershell
python -m documind.training.audit_receipts --split val
```
(lists labels that are missing or wrong, in `reports/receipts_audit/`; fix them, then train again - do **not** rerun step 2 afterwards, it would overwrite your fixes.)

### Step 4 - try it
```powershell
python -m documind.inference.receipts ticket.jpg
```

---------------------------------------------------------------------------------------------------

## 6. CHEQUES  (`models/cheques_yolo.pt`)

### Step 1 - generate
```powershell
python scripts/generate_cheques.py --n 10000 --out data/raw/cheques
```
10 classes: bank_name, cheque_number, amount_words, payee, amount_digits, place, date, signature, account_number, micr_line (handwriting, signature, scan noise).
Options: `--currency MAD|EUR`, `--fonts-dir handwriting_fonts` (put free handwriting `.ttf` fonts there for much more realistic handwriting),
`--workers 4`, `--preview 10`.

### Step 2 - prepare
```powershell
python -m documind.data.prepare_cheques --data data/raw/cheques --workers 4
```
A cheque is wide (about 1600 x 700), so `--max-side` defaults to 1600 to keep the handwriting and the thin MICR line readable. Output: `data/processed/cheques_yolo`.

### Step 3 - train
```powershell
python -m documind.training.train_cheques
```
Defaults: 60 epochs, `--imgsz 1280`, `--batch 4`. Output: `models/cheques_yolo.pt`, `reports/cheques_report.json`.

### Step 4 - try it
```powershell
python -m documind.inference.cheques cheque.jpg            # fields, checks (amount in words = amount in digits, date ...), CSV row
python -m documind.inference.cheques cheque.jpg --json     # the full result with boxes
```

---------------------------------------------------------------------------------------------------

## 7. CONTRACTS  (`models/contracts_yolo.pt`)

### Step 1 - generate
```powershell
python scripts/generate_contracts.py --n 10000 --out data/raw/contracts
```
French loan contracts (personal, car, mortgage), one A4 page each, 19 classes (amount, rate, duration, monthly payment, IBAN, parties, signatures, stamp ...).
Options: `--type random|personal|car|mortgage`, `--currency MAD|EUR`, `--workers 4`, `--preview 10`.
It also writes `ground_truth.jsonl` with the true value of every field (used to measure the accuracy, see section 10).

### Step 2 - prepare
```powershell
python -m documind.data.prepare_contracts --data data/raw/contracts --workers 4
```
A contract page has 20 px text: `--max-side` defaults to 1800 so nothing is shrunk. Output: `data/processed/contracts_yolo`.

### Step 3 - train
```powershell
python -m documind.training.train_contracts
```
Defaults: 60 epochs, `--imgsz 1280`, `--batch 4`. Output: `models/contracts_yolo.pt`, `reports/contracts_report.json`.

### Step 4 - try it
```powershell
python -m documind.inference.contracts contract.jpg
```

---------------------------------------------------------------------------------------------------

## 8. OCR model (optional) - a text reader fine-tuned on YOUR documents

The detectors find **where** each field is; the OCR reads **what** it says. Fine-tuning a TrOCR model on crops of your own documents makes the reading more accurate.
Needs the datasets of steps 4-7 first (it cuts the boxes out of them).

### Step 1 + 2 - cut the crops (prepare)
```powershell
python -m documind.data.prepare_ocr --check 150
```
Cuts every labelled box out of the receipts, cheques, contracts and FATURA datasets into `data/processed/ocr_crops` with its text, and prints
**Tesseract's error rate** on a sample (the number your model has to beat). Options: `--sources receipts,cheques` (only some), `--max-per-source 40000`.

### Step 3 - train
```powershell
python -m documind.training.train_ocr --smoke          # 1 minute: checks that everything runs (do this first)
python -m documind.training.train_ocr                  # the real run: microsoft/trocr-small-printed, 4 epochs
```
Variations:
```powershell
python -m documind.training.train_ocr --base microsoft/trocr-base-handwritten --batch 8      # better for handwriting (cheques)
python -m documind.training.train_ocr --sources cheques --epochs 6                           # a model for one source only
```
Output: `models/ocr_trocr/` and `reports/ocr_train_log.json`. The first run downloads the base model from Hugging Face.

### Step 4 - compare with Tesseract
```powershell
python -m documind.training.evaluate_ocr
```
Shows the character error rate of Tesseract against your model on crops it has never seen. If your model is better, the project uses it automatically.

---------------------------------------------------------------------------------------------------

## 9. Reading the scores (after every training)

`reports/<name>_report.json` and the console table show, per class, on **validation and test**:

| Score | Meaning | Good |
|---|---|---|
| `precision` | of the boxes drawn, how many are right | > 0.90 |
| `recall` | of the real fields, how many were found | > 0.90 |
| `mAP50` | overall detection quality (box overlap >= 50 %) | > 0.90 |
| `found_rate` | share of real fields that got at least one correct box | > 0.90 |

The default target is `--target 0.90` on both validation **and** test. If it is not reached, the training prints what to try. The usual order:

1. **Look at the preview pictures** - wrong boxes in step 2 mean wrong learning (this is the most common cause).
2. **Weak small classes?** raise the resolution: `--imgsz 1536 --batch 2`.
3. **Still improving when it stopped?** train longer: `--epochs 100 --patience 100`.
4. **Need more precision?** a bigger model: `--model yolo11m.pt --batch 2`.
5. **Good on val/test but bad on your real documents?** the data is too different from reality: add real, labelled examples of your own documents.

---------------------------------------------------------------------------------------------------

## 10. Use the models, and measure them on your files

```powershell
python -m documind.inference.cli invoice1.jpg scan.pdf -o result.csv     # read files -> one CSV
uvicorn documind.api.main:app --reload --port 8000                       # the API  (http://localhost:8000/docs)
streamlit run streamlit_app.py                                           # the web app (http://localhost:8501), in another terminal
```

**Picture reconstruction.** Every upload is straightened (page cut out, turned upright, lighting evened out) before reading.
`DOCUMIND_RECONSTRUCT` in `.env`: `auto` (default, picture only), `on` (also re-reads the words with the OCR model), `off`.
Measure which is best **on your own files** (runs the real pipeline three ways and prints the winner):
```powershell
python -m documind.training.compare_reconstruct --data data/raw/FATURA --n 60 --photo
python -m documind.training.compare_reconstruct --type contract --data data/raw/contracts --n 60 --photo
```
`--photo` turns the clean test images into phone-like photos first (that is what the reconstruction is for).

---------------------------------------------------------------------------------------------------

## 11. The whole thing, copy-paste (PowerShell)

```powershell
# --- once
conda activate pytorch_gpu
pip install -r requirements.txt
python -m documind.ocr.handwriting

# --- invoices  (FATURA unzipped into data\raw\FATURA)
python -m documind.data.prepare_invoices --fatura data/raw/FATURA --photo-copies 2
python -m documind.training.train_invoices

# --- receipts
python scripts/generate_receipts.py --n 10000 --out data/raw/receipts
python -m documind.data.prepare_receipts --data data/raw/receipts --workers 4
python -m documind.training.train_receipts

# --- cheques
python scripts/generate_cheques.py --n 10000 --out data/raw/cheques
python -m documind.data.prepare_cheques --data data/raw/cheques --workers 4
python -m documind.training.train_cheques

# --- contracts
python scripts/generate_contracts.py --n 10000 --out data/raw/contracts
python -m documind.data.prepare_contracts --data data/raw/contracts --workers 4
python -m documind.training.train_contracts

# --- optional OCR model
python -m documind.data.prepare_ocr --check 150
python -m documind.training.train_ocr --smoke
python -m documind.training.train_ocr
python -m documind.training.evaluate_ocr

# --- run
uvicorn documind.api.main:app --reload --port 8000
streamlit run streamlit_app.py
```

---------------------------------------------------------------------------------------------------

## 12. Problems and fixes

| Message / symptom | Fix |
|---|---|
| `GPU: False` | install the GPU build of PyTorch in this environment, then `pip install -r requirements.txt` again |
| `CUDA out of memory` while training | lower the batch: `--batch 2` (or 1); or lower `--imgsz` |
| `Ultralytics is not installed` | `pip install -r requirements.txt` in the **same** environment |
| `... not found. Train it first` | the model file is missing in `models\`: run step 3 of that document type |
| `The invoice model is not installed` | train it (section 4) or put the file at `models\invoices_yolo.pt` (or set `INVOICE_MODEL_PATH` in `.env`) |
| `tesseract is not recognized` | install Tesseract and set `TESSERACT_CMD` in `.env` |
| `No image with an annotation file` (invoices) | check the FATURA folder with `python -m documind.data.inspect_dataset --data data/raw/FATURA --check` |
| boxes in the preview are shifted | the annotation coordinates are read the wrong way: for FATURA check with `inspect_dataset --check`; for your own YOLO data try `--ignore-exif` |
| low confidence on real photos | retrain the invoices with `--photo-copies 2`; add real labelled invoices; see `INVOICE_YOLO_CONF` in `.env.example` |
| the score is high but real documents are bad | the training data is too clean or too different: add real labelled examples |
| you re-ran step 2 and lost fixed labels | step 2 rebuilds `data\processed` from `data\raw`; keep a copy of corrected labels before rerunning it |
