"""Invoice YOLO detector: FATURA -> YOLO labels, boxes -> fields, and the extractor choosing the detector. A fake model stands in for ultralytics (no GPU, no weights)."""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from documind.inference import invoice_yolo
from documind.inference.extractor import FieldExtractor, ModelNotFound


class FakeBox:
    def __init__(self, cls, xyxy, conf):
        self.cls, self.conf = cls, conf
        self.xyxy = [type("T", (), {"tolist": lambda s, v=xyxy: v})()]


class FakeYolo:
    names = {0: "NUMBER", 1: "TOTAL", 2: "SELLER_ADDRESS"}
    overrides = {"imgsz": 640}

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, img, imgsz=None, conf=None, verbose=False):
        return [type("R", (), {"boxes": [FakeBox(c, b, p) for c, b, p in self.boxes]})()]


def test_kinds_and_blocks():
    k = invoice_yolo.kind_of
    assert k("DUE_DATE") == "date" and k("TOTAL") == "total" and k("SUB_TOTAL") == "subtotal" and k("GST") == "tax" and k("SELLER_PHONE") == "phone" and k("TITLE") is None
    assert invoice_yolo.is_block("SELLER_ADDRESS") and invoice_yolo.is_block("BUYER") and not invoice_yolo.is_block("TOTAL")


def test_fields_keep_best_single_box_and_join_blocks_top_to_bottom(monkeypatch):
    texts = {(10, 10): "INV-001", (10, 500): "$ 99.00", (10, 300): "second line", (10, 270): "12 Main Street"}
    monkeypatch.setattr(invoice_yolo, "_read", lambda img, xyxy, label, lang: texts[(int(xyxy[0]), int(xyxy[1]))])
    dets = [("NUMBER", [10, 10, 100, 30], 0.6), ("NUMBER", [10, 700, 100, 720], 0.3),           # printed once: the 0.6 box wins (the other would KeyError if read)
            ("total", [10, 500, 100, 520], 0.9),
            ("SELLER_ADDRESS", [10, 300, 200, 320], 0.8), ("SELLER_ADDRESS", [10, 270, 200, 290], 0.7)]
    out = invoice_yolo.fields_from_dets(Image.new("RGB", (400, 800), "white"), dets, 2)
    got = {f["label"]: f for f in out}
    assert got["NUMBER"]["text"] == "INV-001" and got["TOTAL"]["text"] == "$ 99.00"
    assert got["SELLER_ADDRESS"]["text"] == "12 Main Street second line" and got["SELLER_ADDRESS"]["bbox"] == [10.0, 270.0, 200.0, 320.0]
    assert [f["label"] for f in out] == ["NUMBER", "SELLER_ADDRESS", "TOTAL"] and all(f["page"] == 2 for f in out)


def test_extractor_uses_the_detector_and_reports_its_fields(monkeypatch):
    fake = FakeYolo([(0, [40, 30, 200, 60], 0.9), (1, [300, 400, 420, 430], 0.8)])
    monkeypatch.setattr(invoice_yolo, "model_available", lambda path=None: True)
    monkeypatch.setattr(invoice_yolo, "load_model", lambda path=None: fake)
    monkeypatch.setattr(invoice_yolo, "_read", lambda img, xyxy, label, lang: {"NUMBER": "A-77", "TOTAL": "$ 10.00"}[label])
    ex = FieldExtractor()
    assert ex.classes == ["NUMBER", "TOTAL", "SELLER_ADDRESS"] and ex.meta["backend"] == "yolo" and "TOTAL" in ex.columns
    page = np.full((600, 500, 3), 255, np.uint8)
    cv2.putText(page, "Invoice A-77", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    r = ex.extract_image(page, 1)
    assert {f["label"]: f["text"] for f in r["fields"]}["NUMBER"] == "A-77" and r["size"] == [500, 600]


def test_extractor_without_detector_file_explains_how_to_train_it(monkeypatch):
    monkeypatch.setattr(invoice_yolo, "model_available", lambda path=None: False)
    with pytest.raises(ModelNotFound, match="train_invoices"):
        FieldExtractor()


def test_invoice_is_available_as_soon_as_the_detector_file_exists(monkeypatch, tmp_path):
    """The screen said "invoice model not installed" while models/invoices_yolo.pt existed: availability must look at the detector file only."""
    from documind.inference.service import DocumentService
    pt = tmp_path / "invoices_yolo.pt"
    monkeypatch.setattr(invoice_yolo, "MODEL_PATH", pt)
    assert DocumentService().available()["invoice"] is False
    pt.write_bytes(b"x")
    assert DocumentService().available()["invoice"] is True


def test_fatura_annotations_become_yolo_labels(tmp_path):
    from documind.data.prepare_invoices import convert
    src = tmp_path / "FATURA"
    (src / "images").mkdir(parents=True); (src / "Annotations" / "COCO_compatible_format").mkdir(parents=True)
    for t in range(6):
        Image.new("RGB", (1000, 1400), "white").save(src / "images" / f"Template{t}_Instance0.jpg")
        coco = {"images": [{"width": 1000, "height": 1400}], "categories": [{"id": 1, "name": "TOTAL"}, {"id": 2, "name": "TABLE"}, {"id": 3, "name": "GST(5%)"}],
                "annotations": [{"bbox": [100, 200, 300, 50], "category_id": 1}, {"bbox": [0, 500, 900, 300], "category_id": 2}, {"bbox": [50, 1000, 200, 40], "category_id": 3}]}
        (src / "Annotations" / "COCO_compatible_format" / f"Template{t}_Instance0.json").write_text(json.dumps(coco))
    names = convert(src, tmp_path / "out", split="random", seed=1)
    assert names == ["GST", "TOTAL"]                                            # TABLE dropped, GST(5%) merged to GST
    lab = next((tmp_path / "out" / "labels").rglob("*.txt")).read_text().splitlines()
    parts = {int(l.split()[0]): [float(v) for v in l.split()[1:]] for l in lab}
    assert parts[1] == pytest.approx([0.25, 225 / 1400, 0.3, 50 / 1400], abs=1e-4)      # TOTAL: x 100..400 of 1000, y 200..250 of 1400
    assert (tmp_path / "out" / "data.yaml").read_text().count(":") >= 3


def test_photo_copies_are_added_to_training_only_with_the_same_labels(tmp_path):
    from documind.data.prepare_invoices import convert
    src = tmp_path / "FATURA"
    (src / "images").mkdir(parents=True); (src / "Annotations" / "COCO_compatible_format").mkdir(parents=True)
    for t in range(6):
        im = np.full((700, 500, 3), 255, np.uint8)
        cv2.putText(im, "TOTAL 99", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.imwrite(str(src / "images" / f"Template{t}_Instance0.jpg"), im)
        coco = {"images": [{"width": 500, "height": 700}], "categories": [{"id": 1, "name": "TOTAL"}], "annotations": [{"bbox": [100, 170, 250, 40], "category_id": 1}]}
        (src / "Annotations" / "COCO_compatible_format" / f"Template{t}_Instance0.json").write_text(json.dumps(coco))
    convert(src, tmp_path / "out", split="random", seed=1, photo_copies=2)
    train = sorted((tmp_path / "out" / "labels" / "train").glob("*.txt"))
    orig = [p for p in train if "_photo" not in p.name]
    assert len(train) == 3 * len(orig) and not list((tmp_path / "out" / "labels" / "val").glob("*_photo*")) and not list((tmp_path / "out" / "labels" / "test").glob("*_photo*"))
    assert all((p.parent / (p.stem + "_photo1.txt")).read_text() == p.read_text() for p in orig)
