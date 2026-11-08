
### Receipts (YOLO detector)

Data: `data/raw/receipts/{train,val,test}/{images,labels}` (YOLO labels: `class x_center y_center width height`).
```
pip install -r requirements-receipts.txt                  # once, in the environment with the GPU build of PyTorch
python -m documind.data.prepare_receipts                   # STEP 1: check labels, fix rotation, shrink pictures (1100 px), find leaks
                                                           #   -> data/processed/receipts_yolo/ , reports/receipts_preview/ (LOOK at it)
python -m documind.training.train_receipts                 # STEP 2: train (50 epochs max, early stop) + score val and test -> models/receipts_yolo.pt
                                                           #   scores: precision, recall, mAP and found_rate (did it find the field?) in reports/receipts_report.json
python -m documind.training.audit_receipts --split val     # STEP 2b: list the missing / wrong labels that cap the mAP -> reports/receipts_audit/
python -m documind.inference.receipts ticket.jpg           # STEP 3: store / date / total / receipt number / items
```
or `notebooks/train_receipts_yolo.ipynb`. Tests: `pytest -q tests/test_receipts.py`.
