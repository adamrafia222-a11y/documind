"""Fast unit tests (no Tesseract needed):  pytest -q"""
import json
import numpy as np

from documind.data.labeling import assign_labels
from documind.data.loader import parse_annotation, template_of
from documind.features import Featurizer, assign_lines
from documind.inference.export import fields_to_row, rows_to_csv
from documind.inference.grouping import group_fields


def word(text, x0, y0, x1, y1, label="O"):
    return {"text": text, "conf": 90.0, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": label}


def test_template_of():
    assert template_of("Template12_Instance34") == "template12"
    assert template_of("Template3_Instance0") == "template3"


def test_parse_coco(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"images": [], "categories": [{"id": 1, "name": "TOTAL"}],
                             "annotations": [{"bbox": [10, 20, 30, 40], "category_id": 1}]}))
    assert parse_annotation(p, (100, 100)) == [("TOTAL", (10, 20, 40, 60))]


def test_assign_labels_prefers_overlapping_box():
    words = [word("Total:", 0, 0, 50, 20), word("$10.00", 60, 0, 120, 20)]
    assign_labels(words, [("TOTAL", (55, -2, 125, 22))])
    assert [w["label"] for w in words] == ["O", "TOTAL"]


def test_lines_and_features():
    words = [word("Invoice", 10, 10, 80, 30), word("No:", 90, 11, 120, 31), word("INV-123", 130, 10, 220, 30),
             word("Date", 10, 60, 60, 80), word("01/02/2024", 70, 60, 180, 80)]
    layout = assign_lines(words)
    assert len(layout.lines) == 2 and layout.line_of == [0, 0, 0, 1, 1]
    docs = [{"width": 300.0, "height": 400.0, "words": words}] * 3
    X = Featurizer().fit(docs).transform(docs)
    assert X.shape[0] == 15


def test_group_and_csv():
    words = [word("Atlas", 10, 10, 60, 30), word("Supplies", 70, 10, 150, 30), word("Date:", 10, 80, 60, 100),
             word("01/02/2024", 70, 80, 180, 100)]
    labels = ["VENDOR", "VENDOR", "O", "DATE"]
    fields = group_fields(words, labels, np.array([0.9, 0.8, 0.9, 0.95]), page=1)
    assert [(f["label"], f["text"]) for f in fields] == [("VENDOR", "Atlas Supplies"), ("DATE", "01/02/2024")]
    row = fields_to_row(fields)
    csv_bytes = rows_to_csv([{"file": "a.jpg", **row}], ["file", "VENDOR", "DATE"])
    assert csv_bytes.decode("utf-8-sig").splitlines() == ["file,VENDOR,DATE", "a.jpg,Atlas Supplies,01/02/2024"]


# ---------------------------------------------------------------- annotation conventions / alignment
from documind.data.labeling import align_labels
from documind.data.loader import find_samples, read_annotation, select_samples


def _words():
    # two OCR words that must be found whatever the annotation convention is
    return [word("Total:", 100, 500, 200, 530), word("$10.00", 220, 500, 340, 530), word("noise", 600, 900, 700, 930)]


def _ann_file(tmp_path, bbox, name="a.json", ref=None):
    imgs = [{"id": 1, "width": ref[0], "height": ref[1]}] if ref else []
    p = tmp_path / name
    p.write_text(json.dumps({"images": imgs, "categories": [{"id": 1, "name": "Total Amount"}],
                             "annotations": [{"bbox": bbox, "category_id": 1}]}))
    return read_annotation(p)


def test_alignment_detects_every_box_convention(tmp_path):
    W, H = 1000, 1400
    x0, y0, x1, y1 = 215, 497, 345, 533               # box around "$10.00"
    cases = {
        "xywh": [x0, y0, x1 - x0, y1 - y0],
        "xyxy": [x0, y0, x1, y1],
        "xyxy_n1000": [x0 * 1000 / W, y0 * 1000 / H, x1 * 1000 / W, y1 * 1000 / H],
        "xyxy_n1": [x0 / W, y0 / H, x1 / W, y1 / H],
    }
    for expected, bbox in cases.items():
        words = _words()
        info = align_labels(words, [_ann_file(tmp_path, bbox)], (W, H))
        assert info["mode"] == expected, (expected, info)
        assert [w["label"] for w in words] == ["O", "TOTAL_AMOUNT", "O"], expected


def test_alignment_rescales_when_file_has_another_image_size(tmp_path):
    words = _words()
    # the file was written for a 500x700 image, the real image is 1000x1400
    info = align_labels(words, [_ann_file(tmp_path, [107.5, 248.5, 65, 18], ref=(500, 700))], (1000, 1400))
    assert info["mode"] == "xywh_ref"
    assert words[1]["label"] == "TOTAL_AMOUNT"


def test_alignment_never_trusts_a_wrong_hint_and_reports_no_match(tmp_path):
    words = _words()
    info = align_labels(words, [_ann_file(tmp_path, [5000, 5000, 10, 10])], (1000, 1400))   # off the page
    assert info["mode"] == "none" and all(w["label"] == "O" for w in words)


def test_drop_labels_become_background(tmp_path):
    words = _words()
    align_labels(words, [_ann_file(tmp_path, [215, 497, 345, 533])], (1000, 1400), drop={"TOTAL_AMOUNT"})
    assert all(w["label"] == "O" for w in words)


def test_read_original_fatura_like_formats(tmp_path):
    # class -> {text, bbox}
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"TOTAL": {"text": "$1", "bbox": [1, 2, 30, 40]}, "BUYER": {"text": "x", "bbox": [5, 6, 7, 8]}}))
    assert {(b.label, b.vals) for b in read_annotation(p).boxes} == {("TOTAL", (1, 2, 30, 40)), ("BUYER", (5, 6, 7, 8))}
    # class -> list of {box: {xmin..}}
    p.write_text(json.dumps({"seller name": [{"bounding_box": {"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}}]}))
    (b,) = read_annotation(p).boxes
    assert (b.label, b.vals, b.hint) == ("SELLER_NAME", (1, 2, 3, 4), "xyxy")
    # HuggingFace / LayoutLM
    p.write_text(json.dumps({"words": ["a", "b"], "bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "ner_tags": [0, 2],
                             "label_names": ["O", "B-DATE", "TOTAL"]}))
    assert [b.label for b in read_annotation(p).boxes] == ["O", "TOTAL"]


def test_uses_all_images_by_default_and_keeps_one_copy_of_coloured_duplicates(tmp_path):
    for sub in ("white", "colored"):
        (tmp_path / sub).mkdir()
        for t in (1, 2):
            for k in range(5):
                (tmp_path / sub / f"Template{t}_Instance{k}.jpg").write_bytes(b"x")
    (tmp_path / "ann").mkdir()
    for t in (1, 2):
        for k in range(5):
            (tmp_path / "ann" / f"Template{t}_Instance{k}.json").write_text("{}")
    one = find_samples(tmp_path)
    assert len(one) == 10 and not any(s["colored"] for s in one)
    assert len(find_samples(tmp_path, include_colored=True)) == 20
    assert len(select_samples(one, 0, 0, 1)) == 10          # 0 = everything
    assert len(select_samples(one, 3, 0, 1)) == 6           # 3 per template x 2 templates


# ---------------------------------------------------------------- features v2 / audit / metrics
def test_parse_amount():
    from documind.features.builder import parse_amount
    assert parse_amount("$1,234.50") == 1234.5
    assert parse_amount("1.234,50") == 1234.5
    assert parse_amount("12,5") == 12.5
    assert parse_amount("abc") is None


def test_structure_blocks_and_amount_rank():
    from documind.features.builder import NUM_FEATURE_NAMES, build_table
    words = [word("Bill", 10, 10, 50, 30), word("To:", 60, 10, 90, 30), word("Alice", 10, 40, 60, 60),
             word("Main", 10, 70, 60, 90), word("Total:", 400, 10, 460, 30), word("$50.00", 480, 10, 560, 30),
             word("Tax:", 400, 40, 440, 60), word("$5.00", 480, 40, 540, 60)]
    rows, texts, _ = build_table(words, 600.0, 800.0)
    ix = {n: i for i, n in enumerate(NUM_FEATURE_NAMES)}
    assert rows[2, ix["blk_n_lines"]] == 3 and rows[2, ix["blk_line_idx"]] == 1      # "Alice" is line 2 of the address block
    assert texts["bhead"][3].startswith("bill to")                                    # block header is the key of the address
    assert rows[5, ix["amt_rank_desc"]] == 0 and rows[7, ix["amt_rank_desc"]] == 1     # the total is the largest amount


def test_featurizer_position_none_drops_page_position_columns():
    words = [word("Invoice", 10, 10, 80, 30), word("No:", 90, 11, 120, 31), word("INV-123", 130, 10, 220, 30),
             word("Date", 10, 60, 60, 80), word("01/02/2024", 70, 60, 180, 80)]
    docs = [{"width": 300.0, "height": 400.0, "words": words}] * 3
    full = Featurizer("full").fit(docs)
    none = Featurizer("none").fit(docs)
    assert "ncy" in full.feature_names() and "ncy" not in none.feature_names()
    assert full.transform(docs).shape[0] == none.transform(docs).shape[0] == 15


def test_macro_f1_ignores_absent_fields():
    import numpy as np
    from documind.training.evaluate import evaluate

    class M:
        def predict(self, X):
            return np.array([0, 1, 1])
    r = evaluate("t", M(), None, np.array([0, 1, 1]), ["O", "A", "B"], verbose=False)   # B never occurs in y
    assert r["macro_f1_fields"] == 1.0 and r["macro_f1_all_fields"] == 0.5   # B absent: 0 in the strict number only
    r = evaluate("t", M(), None, np.array([0, 1, 2]), ["O", "A", "B"], verbose=False)
    assert 0 < r["macro_f1_fields"] < 1.0


def test_label_purity_flags_wrong_labels():
    from documind.data.audit import label_purity
    good = {"template": "t", "words": [{**word("01/02/2024", 0, 0, 1, 1), "label": "DATE"}]}
    bad = {"template": "t", "words": [{**word("hello", 0, 0, 1, 1), "label": "DATE"}]}
    assert label_purity([good]).iloc[0]["ok"] == "OK"
    assert label_purity([bad]).iloc[0]["ok"] == "SUSPICIOUS"


def test_alignment_handles_bottom_left_origin_like_fatura_original_format(tmp_path):
    # FATURA "Original_Format": [[x0, y0], [x1, y1]] counted from the BOTTOM of the page (PDF style).
    # Image 595x841; the word sits at y = 100..112 from the top -> 729..741 from the bottom.
    words = [word("Date:", 20, 100, 60, 112), word("20-Mar-2008", 65, 100, 160, 112), word("noise", 300, 500, 340, 512)]
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"DATE": {"bbox": [[18.0, 741.0], [162.0, 729.0]], "text": "Date: 20-Mar-2008"}}))
    info = align_labels(words, [read_annotation(p)], (595, 841))
    assert info["mode"] == "xyxy_flipy", info
    assert [w["label"] for w in words] == ["DATE", "DATE", "O"]


def test_date_regex_knows_fatura_dates():
    from documind.features.builder import DATE_RE
    for ok in ("20-Mar-2008", "16-Oct-2016", "20/03/2008", "2008-03-20", "Mar-20-2008"):
        assert DATE_RE.match(ok), ok
    for no in ("Date:", "Total", "12-345", "20-Mar"):
        assert not DATE_RE.match(no), no


def test_label_purity_is_judged_per_document_not_per_word():
    from documind.data.audit import label_purity
    # a DATE box also holds the word "Date:" -> only 50 % of the words are date-like, and that is fine
    doc = {"template": "t", "words": [{**word("Date:", 0, 0, 1, 1), "label": "DATE"},
                                      {**word("20-Mar-2008", 2, 0, 3, 1), "label": "DATE"}]}
    row = label_purity([doc]).iloc[0]
    assert row["ok"] == "OK" and row["%docs hit"] == 100.0
