"""Tests of the receipt pipeline that need neither Ultralytics nor Tesseract (pytest -q)."""
from pathlib import Path

from PIL import Image

from documind.data.prepare_receipts import parse_label, run
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
    stats = run(src, dst, workers=1, reports_dir=rep)
    assert stats["splits"]["train"]["images"] == 2              # the unlabeled picture is dropped
    assert max(Image.open(dst / "train" / "images" / "big.jpg").size) <= 1600
    yaml_text = (dst / "data.yaml").read_text(encoding="utf-8")
    assert '0: "store"' in yaml_text and '4: "item"' in yaml_text and "test: test/images" in yaml_text
    assert (rep / "receipts_preview").is_dir()
    report = (rep / "receipts_prepare_report.txt").read_text(encoding="utf-8")
    assert "no label file" in report and "more than one split" in report


def test_fields_from_boxes_parsing(monkeypatch):
    texts = {"store": "PANDA MARKET", "date": "تاريخ 30 / 09 / 2022", "total": "إجمالي بعد الخصم 548.98", "receipt_no": "عملية رقم 3347048", "item": "item row"}
    monkeypatch.setattr(receipts, "ocr_box", lambda img, xyxy, psm=7, pad=4, scale=2.0, lang=None: texts[dets_by_box[tuple(xyxy)]])
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
