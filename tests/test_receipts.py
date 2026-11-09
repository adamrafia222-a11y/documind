"""Tests of the receipt pipeline that need neither Ultralytics nor Tesseract (pytest -q)."""
from pathlib import Path

from PIL import Image

from documind.data.prepare_receipts import DEFAULT_NAMES, LEGACY_NAMES, parse_label, read_names_from_yaml, run
from documind.inference import receipts


def test_parse_label_fixes_and_drops(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(
        "0 0.5 0.1 0.4 0.1\n"
        "0 0.5 0.1 0.4 0.1\n"                                   # duplicate
        "2 0.98 0.5 0.1 0.1\n"                                  # sticks out a little: clipped
        "4 0.3 0.3 0.5 0.3 0.5 0.35 0.3 0.35\n"                 # polygon -> box
        "x y z\n"                                               # unreadable
        "1 0.5 0.5\n"                                           # wrong number of values
        "3 0.5 0.5 0.0001 0.1\n",                               # far too thin
        encoding="utf-8")
    boxes, issues = parse_label(f)
    assert [b[0] for b in boxes] == [0, 2, 4]
    assert all(0 <= v <= 1 for b in boxes for v in b[1:])
    assert issues["duplicate line removed"] == 1 and issues["small overflow clipped"] == 1
    assert issues["polygon converted to box"] == 1 and issues["unreadable line"] == 1
    assert issues["wrong number of values"] == 1 and issues["box too small (dropped)"] == 1


def _make(split_dir: Path, name: str, size=(300, 600), label="0 0.5 0.1 0.4 0.1\n2 0.4 0.5 0.3 0.05\n", seed=0):
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, (240, 240, 240))
    for i in range(0, size[0], 7 + seed):                       # a pattern that differs between pictures (so they are not "identical")
        for j in range(0, size[1], 40 + 9 * seed):
            im.putpixel((i, j), (0, 0, 0))
    im.save(split_dir / "images" / f"{name}.jpg")
    if label is not None:
        (split_dir / "labels" / f"{name}.txt").write_text(label)


def test_prepare_end_to_end(tmp_path):
    src, dst, rep = tmp_path / "raw", tmp_path / "out", tmp_path / "rep"
    _make(src / "train", "big", size=(3200, 2400), seed=1)
    _make(src / "train", "ok", seed=2)
    _make(src / "train", "nolabel", label=None, seed=3)
    _make(src / "val", "v", seed=4)
    _make(src / "test", "t", seed=5)
    _make(src / "test", "leak", seed=2)                         # same picture as train/ok
    stats = run(src, dst, LEGACY_NAMES, workers=1, reports_dir=rep)          # the older 5-class labelling
    assert stats["splits"]["train"]["images"] == 2              # the unlabeled picture is dropped
    assert max(Image.open(dst / "train" / "images" / "big.jpg").size) <= 1600
    yaml_text = (dst / "data.yaml").read_text(encoding="utf-8")
    assert '0: "store"' in yaml_text and '4: "item"' in yaml_text and "test: test/images" in yaml_text
    assert (rep / "receipts_preview").is_dir()
    report = (rep / "receipts_prepare_report.txt").read_text(encoding="utf-8")
    assert "no label file" in report and "more than one split" in report


def test_fields_from_boxes_parsing(monkeypatch):
    texts = {"store": "PANDA MARKET", "date": "تاريخ 30 / 09 / 2022", "total": "إجمالي بعد الخصم 548.98", "receipt_no": "عملية رقم 3347048", "item": "item row"}
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, **kw: texts[dets_by_box[tuple(xyxy)]])
    dets = [("store", [0, 0, 10, 10], 0.9), ("date", [0, 20, 10, 30], 0.8), ("total", [0, 40, 10, 50], 0.7), ("total", [0, 41, 10, 51], 0.4),
            ("receipt_no", [0, 60, 10, 70], 0.8), ("item", [0, 90, 10, 100], 0.9), ("item", [0, 80, 10, 90], 0.9)]
    dets_by_box = {tuple(b): n for n, b, _ in dets}
    out = receipts.fields_from_boxes(Image.new("RGB", (50, 120)), dets)
    assert out["date"]["value"] == "30/09/2022"
    assert out["total"]["value"] == "548.98" and out["total"]["box_confidence"] == 0.7   # most confident box wins
    assert out["receipt_no"]["value"] == "3347048"
    assert len(out["items"]) == 2
    texts["total"] = "Total 1 548,98 MAD"                                                  # a space as thousands separator belongs to the amount
    assert receipts.fields_from_boxes(Image.new("RGB", (50, 120)), dets)["total"]["value"] == "1548,98"


# ------------------------------------------------------------------------------------------------ generate_receipts.py data (13 classes, images/<split> layout)
def _make_generator_layout(src: Path) -> None:
    """what scripts/generate_receipts.py writes: images/<split>/x.jpg, labels/<split>/x.txt and a data.yaml with a flow-style names list."""
    for k, split in enumerate(("train", "val", "test")):
        _make(src / "tmp", "x", seed=10 + k, label="0 0.5 0.1 0.4 0.1\n11 0.4 0.8 0.3 0.05\n")
        (src / "images" / split).mkdir(parents=True, exist_ok=True)
        (src / "labels" / split).mkdir(parents=True, exist_ok=True)
        (src / "tmp" / "images" / "x.jpg").replace(src / "images" / split / f"r{k}.jpg")
        (src / "tmp" / "labels" / "x.txt").replace(src / "labels" / split / f"r{k}.txt")
    (src / "data.yaml").write_text(f"path: {src}\ntrain: images/train\nnc: {len(DEFAULT_NAMES)}\nnames: {DEFAULT_NAMES}\n", encoding="utf-8")


def test_read_names_from_yaml_both_styles(tmp_path):
    flow = tmp_path / "a.yaml"
    flow.write_text("path: /x\nnc: 3\nnames: ['store_name', 'date', 'total']\n", encoding="utf-8")
    block = tmp_path / "b.yaml"
    block.write_text('path: /x\nnames:\n  0: "store"\n  1: "date"\n  2: "total"\n', encoding="utf-8")
    assert read_names_from_yaml(flow) == ["store_name", "date", "total"] and read_names_from_yaml(block) == ["store", "date", "total"]
    assert read_names_from_yaml(tmp_path / "missing.yaml") is None


def test_prepare_reads_the_generator_layout_and_its_class_names(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    _make_generator_layout(src)
    stats = run(src, dst, workers=1, reports_dir=tmp_path / "rep")
    assert {s: v["images"] for s, v in stats["splits"].items()} == {"train": 1, "val": 1, "test": 1}
    assert stats["names"][0] == "store_name" and stats["names"][11] == "total" and len(stats["names"]) == 13
    yaml_text = (dst / "data.yaml").read_text(encoding="utf-8")
    assert '0: "store_name"' in yaml_text and '12: "payment_method"' in yaml_text and (dst / "val" / "labels" / "r1.txt").exists()


def test_item_rows_are_grouped_on_a_tilted_receipt():
    """The receipt is rotated: the price (right) of an item sits almost one line lower than its name (left). Rows must still pair up."""
    import math
    parts = []
    for i in range(6):
        for kind, x in (("item_qty", 30), ("item_name", 60), ("item_price", 380)):
            y = 100 + 34 * i + math.tan(math.radians(4)) * x                       # 4 degrees of tilt: 380 px to the right = 26 px lower
            parts.append((kind, [x, y, x + 40, y + 22], 0.9))
    rows = receipts.group_item_rows(parts)
    assert len(rows) == 6 and all(set(r) == {"item_qty", "item_name", "item_price"} for r in rows)
    assert receipts.group_item_rows([]) == []


def test_fields_from_boxes_with_the_13_classes(monkeypatch):
    texts = {"store_name": "SUPERMARCHE ATLAS", "address": "12 Av. Hassan II", "phone": "05 24 11 22 33", "date": "30/09/2025", "time": "14:35",
             "receipt_number": "3347048", "subtotal": "100,00", "tax": "20,00", "total": "120,00 DH", "payment_method": "Especes"}
    cells = {"A": "Lait 1L", "q1": "1", "p1": "10,50", "B": "Pain", "q2": "2", "p2": "3,00"}
    dets = [(n, [0, 10 * i, 10, 10 * i + 8], 0.9) for i, n in enumerate(texts)]
    dets += [("address", [0, 200, 10, 208], 0.8)]
    texts_by_box = {tuple(b): texts[n] for n, b, _ in dets if n != "address" or b[1] != 200}
    texts_by_box[(0, 200, 10, 208)] = "40000 Marrakech"
    rows = [("item_qty", [0, 300, 10, 310], "q1"), ("item_name", [20, 301, 90, 311], "A"), ("item_price", [300, 300, 340, 310], "p1"),
            ("item_name", [20, 340, 90, 350], "B"), ("item_qty", [0, 341, 10, 351], "q2"), ("item_price", [300, 340, 340, 350], "p2")]
    for n, b, key in rows:
        dets.append((n, b, 0.85)); texts_by_box[tuple(b)] = cells[key]
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, **kw: texts_by_box[tuple(xyxy)])
    out = receipts.fields_from_boxes(Image.new("RGB", (400, 400)), dets)
    assert out["store"]["value"] == "SUPERMARCHE ATLAS" and out["receipt_no"]["value"] == "3347048" and out["date"]["value"] == "30/09/2025"
    assert out["time"]["value"] == "14:35" and out["total"]["value"] == "120,00" and out["subtotal"]["value"] == "100,00" and out["tax"]["value"] == "20,00"
    assert out["address"]["value"] == "12 Av. Hassan II, 40000 Marrakech" and out["payment_method"]["value"] == "Especes"
    assert [(i["qty"], i["description"], i["amount"]) for i in out["items"]] == [("1", "Lait 1L", "10,50"), ("2", "Pain", "3,00")]


# ------------------------------------------------------------------------------------------------ evaluation helpers (no model needed)
def test_match_image_found_and_suspects():
    from documind.training.receipts_eval import match_image, summarize
    names = {0: "store", 1: "date", 2: "total", 3: "receipt_no", 4: "item"}
    gt = [(2, [0.1, 0.8, 0.5, 0.85]), (4, [0.1, 0.3, 0.9, 0.34]), (4, [0.1, 0.4, 0.9, 0.44]), (4, [0.1, 0.5, 0.9, 0.54])]
    pred = [(2, 0.9, [0.11, 0.8, 0.5, 0.85]),            # total found
            (2, 0.4, [0.1, 0.2, 0.3, 0.25]),             # weaker second total: ignored for found-rate
            (1, 0.8, [0.1, 0.1, 0.4, 0.15]),             # a date the model sees but nobody labelled -> suspect (missing label)
            (4, 0.9, [0.1, 0.3, 0.9, 0.34]), (4, 0.8, [0.1, 0.4, 0.9, 0.44])]   # 2 of 3 item rows found
    r = match_image(gt, pred, names)
    assert r["total"]["found"] is True and r["date"]["found"] is None
    assert r["date"]["extra_hi"][0][2] == "no label of this class in the picture"
    assert r["item"]["found"] == 2 and len(r["item"]["missed_gt"]) == 1
    s = summarize([r, match_image(gt, [(2, 0.9, [0.6, 0.1, 0.9, 0.2])], names)])        # 2nd picture: the total is in the wrong place
    assert s["total"]["found_rate"] == 0.5 and s["item"]["labelled_boxes"] == 6


def test_split_dir_reads_data_yaml(tmp_path):
    from documind.training.receipts_eval import split_dir
    y = tmp_path / "data.yaml"
    y.write_text('path: /x/y\ntrain: train/images\nval: val/images\nnames:\n  0: "store"\n', encoding="utf-8")
    assert split_dir(y, "val") == Path("/x/y") / "val/images" and split_dir(y, "test") is None
