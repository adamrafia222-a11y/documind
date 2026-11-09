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

### Several document types in one app (invoice + receipt)
Each type keeps its own model and its own CSV columns; they are never merged.
```
picture / PDF -> router (page shape + key words, or the type chosen in the web page) -> invoice extractor (LightGBM)  -> invoices.csv
                                                                                    -> receipt extractor (YOLO+OCR) -> receipts.csv
```
* `POST /extract?doc_type=auto|invoice|receipt` (default `auto`); the answer has `doc_type`, `doc_type_source` (auto / manual), `columns`.
* `POST /extract/batch` -> one CSV per type: a plain `invoices.csv` / `receipts.csv` if the batch has one type, `extractions.zip` with both if it has two.
* Web page: "Document type" selector (Auto-detect by default), a type badge on every document, "Read as receipt/invoice instead" if the
  router guessed wrong, and `CSV (per type)` downloads `invoices.csv` and `receipts.csv` separately. History and the dataset export
  (`dataset_invoices.csv`, `dataset_receipts.csv`) are split the same way.
* A type whose model file is missing is simply not offered: with only the invoice model installed, everything goes to the invoice extractor.
* New type later (ID card, bank statement ...): write an extractor that returns the same kind of dict and add it in `documind/inference/service.py`.
