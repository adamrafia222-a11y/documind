"""OCR training data and plumbing (no GPU, no PyTorch, no Hugging Face download needed):
the crops + transcriptions of the three datasets, the error-rate maths, the engine switch used by the receipt / cheque readers.
The training loop itself (documind.training.train_ocr) needs PyTorch + the pretrained model: run  python -m documind.training.train_ocr --smoke  once.
    pytest -q tests/test_ocr_training.py
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from documind.data import prepare_ocr
from documind.inference import receipts
from documind.ocr import crops, metrics, trocr
from documind.training import train_ocr

ROOT = Path(__file__).resolve().parent.parent
RECEIPT_GEN, CHEQUE_GEN = ROOT / "scripts" / "generate_receipts.py", ROOT / "scripts" / "generate_cheques.py"


# ------------------------------------------------------------------------------------------------ metrics
def test_error_rates():
    assert metrics.edit_distance("kitten", "sitting") == 3 and metrics.edit_distance("", "abc") == 3 and metrics.edit_distance(["a", "b"], ["a", "c"]) == 1
    assert metrics.cer("12,50", "12,50") == 0 and metrics.cer("12,50", "12.50") == pytest.approx(0.2) and metrics.cer("", "x") == 1.0
    assert metrics.wer("deux cent", "deux cents") == 0.5
    s = metrics.summarize([("abcd", "abcd"), ("abcd", "abxd")])
    assert s == {"n": 2, "cer": 0.125, "wer": 0.5, "exact": 0.5} and metrics.summarize([])["n"] == 0


# ------------------------------------------------------------------------------------------------ which text belongs to which box
def test_receipt_texts_pair_items_by_drawing_order():
    truth = {"store_name": "Marché Central", "address": "12 Rue X, 40000 Marrakech", "phone": "05 24 00 00 00", "total": "120,00 DH",
             "items": [{"name": "Lait", "qty": 2, "unit_price": "5,00", "price": "10,00"}, {"name": "Pain", "qty": 1, "unit_price": "2,00", "price": "2,00"}]}
    box = (0, 0, 10, 10)
    drawn = [("store_name", box), ("address", box), ("address", box), ("item_qty", box), ("item_name", box), ("item_price", box), ("item_qty", box),
             ("item_name", box), ("item_price", box), ("total", box), ("item_name", box)]       # an 3rd item_name that the truth does not have: dropped
    got = [(c, t) for c, _, t in crops.receipt_texts(truth, drawn)]
    assert got == [("store_name", "Marché Central"), ("address", "12 Rue X"), ("address", "40000 Marrakech"), ("item_qty", "2"), ("item_name", "Lait"), ("item_price", "10,00"),
                   ("item_qty", "1"), ("item_name", "Pain"), ("item_price", "2,00"), ("total", "120,00 DH")]


def test_cheque_texts_use_what_is_printed():
    truth = {"cheque_number": "1234567", "account_number": "0413525601230989101399", "payee": "Omar Tazi", "signature": True, "amount_value": 12.5}
    box = (0, 0, 10, 10)
    got = dict((c, t) for c, _, t in crops.cheque_texts(truth, [("cheque_number", box), ("account_number", box), ("payee", box), ("signature", box), ("date", box)]))
    assert got == {"cheque_number": "N° 1234567", "account_number": "0413 5256 0123 0989 1013 99", "payee": "Omar Tazi"}


def test_usable_filters_blocks_and_empty_text():
    assert crops.usable("12,50", (0, 0, 40, 20)) and not crops.usable("", (0, 0, 40, 20)) and not crops.usable("a\nb", (0, 0, 40, 20)) and not crops.usable("x" * 200, (0, 0, 40, 20))
    assert not crops.usable("ok", (0, 0, 3, 3))


# ------------------------------------------------------------------------------------------------ real generator output -> crops
def _generate(script: Path, out: Path, n: int = 12) -> None:
    subprocess.run([sys.executable, str(script), "--n", str(n), "--out", str(out), "--workers", "1", "--preview", "0"], check=True, capture_output=True)


@pytest.mark.skipif(not (RECEIPT_GEN.exists() and CHEQUE_GEN.exists()), reason="generators not present")
def test_prepare_ocr_on_generated_datasets(tmp_path):
    _generate(RECEIPT_GEN, tmp_path / "r")
    _generate(CHEQUE_GEN, tmp_path / "c")
    out = tmp_path / "ocr"
    stats = prepare_ocr.run(out, ["receipts", "cheques"], tmp_path / "r", tmp_path / "c", None, max_per_source=100000, reports_dir=tmp_path / "rep")
    assert stats["n_chars"] > 30 and (tmp_path / "rep" / "ocr_crops_preview.png").exists() and (tmp_path / "rep" / "ocr_crops_stats.json").exists()
    tr = train_ocr.read_labels(out, "train")
    assert {r["source"] for r in tr} == {"receipts", "cheques"} and all(r["path"].exists() for r in tr)
    kinds = {r["kind"] for r in tr}
    assert {"item_name", "item_qty", "item_price", "total", "amount_words", "payee", "account_number", "cheque_number"} <= kinds and "signature" not in kinds
    assert all(r["text"].startswith("N° ") for r in tr if r["kind"] == "cheque_number")
    assert train_ocr.read_labels(out, "train", ["cheques"]) and not any(r["source"] == "receipts" for r in train_ocr.read_labels(out, "train", ["cheques"]))
    val = train_ocr.read_labels(out, "val")
    assert val and not ({r["path"] for r in val} & {r["path"] for r in tr}) and all("/val/" in str(r["path"]).replace("\\", "/") for r in val)
    for r in tr[:8]:
        im = Image.open(r["path"])
        assert im.width >= 4 and im.height >= 4


def test_cap_balanced_shares_the_budget_between_kinds():
    cs = [crops.Crop(Path("x"), (0, 0, 9, 9), "t", "receipts", "item_qty", "train") for _ in range(900)] + [crops.Crop(Path("y"), (0, 0, 9, 9), "t", "receipts", "total", "train") for _ in range(30)]
    got = prepare_ocr.cap_balanced(cs, 100, random.Random(0))
    assert len(got) == 100 and sum(c.kind == "total" for c in got) == 30 and sum(c.kind == "item_qty" for c in got) == 70      # the rare kind keeps all its crops
    assert prepare_ocr.cap_balanced(cs[:10], 100, random.Random(0)) == cs[:10]


def test_fatura_crops_from_original_format_annotations(tmp_path):
    root = tmp_path / "FATURA"
    root.mkdir()
    im = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(im).text((110, 60), "INVOICE", fill="black")
    im.save(root / "Template1_Instance0.jpg")
    (root / "Template1_Instance0.json").write_text(json.dumps({"TITLE": {"text": "INVOICE", "bbox": [100, 50, 400, 110]}, "DATE": {"text": "12/05/2024", "bbox": [900, 200, 1100, 240]},
                                                               "TABLE": {"text": "many\nlines", "bbox": [50, 600, 1100, 900]}}))
    got = list(crops.fatura_crops(root))
    assert {(c.kind, c.text) for c in got} == {("TITLE", "INVOICE"), ("DATE", "12/05/2024")} and all(c.source == "fatura" for c in got)
    assert {c.split for c in got} == {got[0].split} and got[0].split in ("train", "val", "test")             # one invoice = one split


# ------------------------------------------------------------------------------------------------ the engine switch
class FakeEngine:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def read(self, images, batch_size=32):
        self.calls += 1
        return [self.text] * len(images)


def test_receipt_reader_uses_the_trained_model_when_there_is_one(monkeypatch):
    eng = FakeEngine("1 234,50 DH")
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: eng)
    img = Image.new("RGB", (100, 40), "white")
    assert receipts.ocr_box(img, (5, 5, 90, 30)) == "1 234,50 DH" and eng.calls == 1
    assert receipts.ocr_box(img, (5, 5, 90, 30), whitelist="0123456789,. ") == "1 234,50"                       # same character whitelist as the Tesseract path
    assert trocr.filter_chars("a1b2", None) == "a1b2" and trocr.filter_chars("a1b2", "0123456789") == "12"


def test_engine_is_off_without_a_trained_model_or_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(trocr, "_engine", None)
    monkeypatch.setattr(trocr, "_tried", False)
    assert trocr.get_engine(tmp_path / "nothing") is None and not trocr.available(tmp_path / "nothing")
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "config.json").write_text("{}")
    assert trocr.available(tmp_path / "m")
    monkeypatch.setenv("DOCUMIND_OCR", "tesseract")
    assert trocr.get_engine(tmp_path / "m") is None
    monkeypatch.setenv("DOCUMIND_OCR", "trocr")
    with pytest.raises(SystemExit):
        trocr.get_engine(tmp_path / "nothing")


def test_tesseract_still_reads_when_no_model(monkeypatch):
    pytest.importorskip("pytesseract")
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: None)
    img = Image.new("RGB", (220, 60), "white")
    ImageDraw.Draw(img).text((10, 20), "TOTAL 12", fill="black")
    try:
        text = receipts.ocr_box(img, (0, 0, 220, 60))
    except Exception as exc:                                                                                       # Tesseract not installed here
        pytest.skip(str(exc))
    assert isinstance(text, str)


def test_augmentation_keeps_a_usable_picture():
    rng = random.Random(0)
    base = Image.new("RGB", (160, 32), "white")
    ImageDraw.Draw(base).text((6, 8), "Farine 1kg", fill="black")
    for _ in range(30):
        out = train_ocr.augment(base.copy(), rng)
        assert out.mode == "RGB" and out.width >= 8 and out.height >= 8


# ------------------------------------------------------------------------------------------------ optional refinement of the invoice words
def test_refine_words_is_off_by_default_and_replaces_only_doubtful_words(monkeypatch):
    import numpy as np
    from documind.ocr import refine
    gray = np.full((100, 300), 255, dtype="uint8")
    words = [{"text": "T0TAL", "conf": 40.0, "x0": 10, "y0": 10, "x1": 80, "y1": 30}, {"text": "Date", "conf": 96.0, "x0": 100, "y0": 10, "x1": 150, "y1": 30}]
    monkeypatch.delenv("DOCUMIND_OCR_REFINE", raising=False)
    assert refine.refine_words(gray, [dict(w) for w in words], 1.0, FakeEngine("TOTAL"))[0]["text"] == "T0TAL"                       # off: unchanged
    monkeypatch.setenv("DOCUMIND_OCR_REFINE", "lowconf")
    got = refine.refine_words(gray, [dict(w) for w in words], 1.0, FakeEngine("TOTAL"))
    assert got[0]["text"] == "TOTAL" and got[0]["refined"] and got[0]["conf"] == 95.0 and got[1]["text"] == "Date" and "refined" not in got[1]
    monkeypatch.setenv("DOCUMIND_OCR_REFINE", "all")
    assert refine.refine_words(gray, [dict(w) for w in words], 1.0, FakeEngine("X"))[1]["text"] == "Date"                                # both look fine: Tesseract's reading stays
    unsure = [{"text": "Tota1", "conf": 70.0, "x0": 10, "y0": 10, "x1": 80, "y1": 30}]
    assert refine.refine_words(gray, [dict(w) for w in unsure], 1.0, FakeEngine("Total"))[0]["text"] == "Total"                          # clearly better (a digit inside a word): replaced
    assert refine.refine_words(gray, [dict(words[0])], 1.0, FakeEngine("two words"))[0]["text"] == "T0TAL"                          # a multi-word answer for one word box is dropped
