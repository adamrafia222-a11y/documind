"""OCR reconstruction: picture in training form (Stage A), words in training form (Stage B), clean re-render (Stage C).
No GPU, PyTorch or trained model needed (the recognizer is a fake).     pytest -q tests/test_reconstruct.py
"""
import cv2
import numpy as np
import pytest

from documind.ocr import reconstruct
from documind.ocr.reconstruct import normalize_token, reconstruct_words, render_clean
from documind.preprocessing.canonical import Canon, canonicalize, find_page, flatten_light, rotate_page


class Fake:
    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def read(self, images, batch_size=32):
        self.calls += len(images)
        return [self.answers.pop(0) if self.answers else "" for _ in images]


def w(text, x0, y0, x1, y1, conf=90.0, **kw):
    return {"text": text, "conf": conf, "x0": x0, "y0": y0, "x1": x1, "y1": y1, **kw}


# ------------------------------------------------------------------------------------------------ canonical spelling
def test_normalize_token_fixes_numbers_only():
    n = normalize_token
    assert n("2O25") == "2025" and n("l5/O3/2O21") == "15/03/2021" and n("1,5S3.04") == "1,553.04" and n("$I2O.50") == "$120.50"
    assert n("ISO9001") == "ISO9001" and n("BOSS") == "BOSS" and n("TOTAL") == "TOTAL" and n("A1B2") == "A1B2" and n("Mar-2O") == "Mar-2O"
    assert n("1–2") == "1-2" and n("“hi”") == '"hi"' and n("|") == "" and n("__x__") == "x" and n("a\u200bb") == "ab"


# ------------------------------------------------------------------------------------------------ Stage B
def test_reconstruct_replaces_doubtful_words_with_the_recognizer_reading():
    gray = np.full((200, 600), 255, np.uint8)
    words = [w("TOTAL", 10, 10, 90, 40, 95), w("l2,5O", 200, 10, 300, 40, 30)]
    out = reconstruct_words(gray, words, 1.0, engine=Fake(["TOTAL", "12,50"]))
    assert [x["text"] for x in out] == ["TOTAL", "12,50"] or [x["text"] for x in out] == ["TOTAL", "12,50"]
    assert out[1]["conf"] == reconstruct.REPLACED_CONF and out[1]["x0"] == 200 and out[1]["x1"] == 300
    assert words[1]["text"] == "l2,5O"                                                 # the input list is not modified


def test_reconstruct_skips_confident_and_handwritten_words():
    gray = np.full((100, 400), 255, np.uint8)
    eng = Fake(["WRONG", "WRONG"])
    words = [w("Hello", 10, 10, 90, 40, 99), w("Quarante", 150, 10, 300, 40, 40, handwriting=True)]
    out = reconstruct_words(gray, words, 1.0, engine=eng)
    assert eng.calls == 0 and [x["text"] for x in out] == ["Hello", "Quarante"]


def test_reconstruct_different_token_count_cuts_the_span_in_proportion():
    gray = np.full((100, 600), 255, np.uint8)
    words = [w("Tot", 100, 10, 160, 40, 20), w("al", 165, 10, 200, 40, 20)]            # one word split in two by Tesseract
    out = reconstruct_words(gray, words, 1.0, engine=Fake(["Total"]), normalize=False)
    assert [x["text"] for x in out] == ["Total"] and out[0]["x0"] == 100 and out[0]["x1"] == 200
    words = [w("Net", 100, 10, 200, 40, 20)]
    out = reconstruct_words(gray, words, 1.0, engine=Fake(["Net amount"]), normalize=False)
    assert [x["text"] for x in out] == ["Net", "amount"] and out[0]["x1"] <= out[1]["x0"] and out[1]["x1"] == 200


def test_reconstruct_without_recognizer_only_spells_and_drops_junk(monkeypatch):
    from documind.ocr import trocr
    monkeypatch.setattr(trocr, "get_engine", lambda *a, **k: None)
    out = reconstruct_words(np.full((50, 300), 255, np.uint8), [w("2O25", 0, 0, 40, 20), w("|", 50, 0, 55, 20), w("Date", 60, 0, 100, 20)])
    assert [x["text"] for x in out] == ["2025", "Date"]


def test_modes(monkeypatch):
    monkeypatch.delenv("DOCUMIND_RECONSTRUCT", raising=False)
    assert reconstruct.pages_active() and not reconstruct.words_active()                # default (auto): the picture yes, the words no
    monkeypatch.setenv("DOCUMIND_RECONSTRUCT", "on")
    assert reconstruct.words_active()
    monkeypatch.setenv("DOCUMIND_RECONSTRUCT", "off")
    assert not reconstruct.pages_active() and not reconstruct.words_active()


# ------------------------------------------------------------------------------------------------ Stage A
def page_on_table(angle=0.0, shade=False):
    """A white 600x800 page with text-like bars, photographed on a dark table, tilted / in perspective. -> (photo, page corners in the photo)."""
    page = np.full((800, 600, 3), 245, np.uint8)
    for y in range(60, 740, 40):
        cv2.rectangle(page, (50, y), (50 + (y * 7) % 400 + 80, y + 14), (30, 30, 30), -1)
    photo = np.full((1000, 1200, 3), 60, np.uint8)
    src = np.float32([[0, 0], [600, 0], [600, 800], [0, 800]])
    a = np.deg2rad(angle)
    c = np.array([600.0, 500.0])
    corners = np.float32([[300, 100], [880, 130], [850, 900], [330, 880]])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = ((corners - c) @ R.T + c).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, corners)
    photo = np.where(cv2.warpPerspective(np.full_like(page, 255), m, (1200, 1000)) > 0, cv2.warpPerspective(page, m, (1200, 1000)), photo)
    if shade:
        photo = (photo * np.linspace(0.45, 1.0, 1200)[None, :, None]).astype(np.uint8)
    return photo, corners, m


def test_clean_scan_is_left_alone():
    page = np.full((1100, 800, 3), 250, np.uint8)
    cv2.putText(page, "INVOICE", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    c = canonicalize(page, "invoice", orient=False)
    assert c.identity and c.image is page and c.to_original([10, 20, 30, 40]) == [10.0, 20.0, 30.0, 40.0]


def test_photo_page_is_cut_out_and_boxes_map_back():
    photo, corners, m = page_on_table(angle=8)
    q = find_page(photo)
    assert q is not None and float(np.abs(q - corners).max()) < 12
    c = canonicalize(photo, "receipt", orient=False, light=False)
    assert "page" in c.steps and abs(c.image.shape[1] - 600) < 25 and abs(c.image.shape[0] - 800) < 25
    # a box drawn on the flat page (page coordinates) lands where the same region of the photo is
    box_page = [100, 200, 300, 260]
    pts = cv2.perspectiveTransform(np.float32([[[100, 200], [300, 200], [300, 260], [100, 260]]]), m)[0]
    back = c.to_original([v * (c.image.shape[1] / 600) if i % 2 == 0 else v * (c.image.shape[0] / 800) for i, v in enumerate(box_page)])
    assert abs(back[0] - pts[:, 0].min()) < 25 and abs(back[2] - pts[:, 0].max()) < 25 and abs(back[1] - pts[:, 1].min()) < 25 and abs(back[3] - pts[:, 1].max()) < 25


def test_rotation_matrices_agree_with_the_image():
    img = np.zeros((100, 200, 3), np.uint8)
    cv2.rectangle(img, (20, 10), (60, 30), (255, 255, 255), -1)                         # box [20, 10, 60, 30]
    for deg in (90, 180, 270):
        out, m = rotate_page(img, deg)
        canon = Canon(image=out, matrix=m, steps=[f"rotate{deg}"], size=(200, 100))
        ys, xs = np.nonzero(out[:, :, 0])
        got = canon.to_original([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        assert all(abs(a - b) <= 1.5 for a, b in zip(got, [20, 10, 61, 31])), (deg, got)


def test_flatten_light_only_when_uneven():
    flat = np.full((400, 600, 3), 240, np.uint8)
    assert flatten_light(flat)[1] is False
    shaded = (flat * np.linspace(0.4, 1.0, 600)[None, :, None]).astype(np.uint8)
    out, changed = flatten_light(shaded)
    assert changed and out[:, :50].mean() > shaded[:, :50].mean() + 40 and abs(float(out[:, :50].mean()) - float(out[:, -50:].mean())) < 25


# ------------------------------------------------------------------------------------------------ Stage C
def test_render_clean_draws_each_word_in_its_box():
    page = render_clean([w("TOTAL", 20, 20, 120, 50), w("12,50", 300, 20, 380, 50)], (500, 100))
    a = np.asarray(page.convert("L"))
    assert a.shape == (100, 500) and a[20:50, 20:120].min() < 100 and a[20:50, 300:380].min() < 100 and a[60:, :].min() == 255 and a[:, 150:290].min() == 255


# ------------------------------------------------------------------------------------------------ the recognizer
def test_pretrained_printed_model_is_the_fallback_when_nothing_was_fine_tuned(monkeypatch, tmp_path):
    from documind.ocr import trocr
    made = []

    class FakeRec:
        def __init__(self, model_dir, *a, **k):
            made.append(str(model_dir))

    monkeypatch.setattr(trocr, "TrOCRRecognizer", FakeRec)
    monkeypatch.setattr(trocr, "_engine", None); monkeypatch.setattr(trocr, "_tried", False)
    monkeypatch.setattr(trocr, "_pre_engine", None); monkeypatch.setattr(trocr, "_pre_tried", False)
    monkeypatch.delenv("DOCUMIND_OCR", raising=False); monkeypatch.delenv("DOCUMIND_PRETRAINED_OCR", raising=False)
    eng = trocr.get_engine(tmp_path / "missing")
    assert isinstance(eng, FakeRec) and made == [trocr.PRETRAINED] and trocr.get_engine(tmp_path / "missing") is eng      # loaded once
    monkeypatch.setattr(trocr, "_pre_engine", None); monkeypatch.setattr(trocr, "_pre_tried", False)
    monkeypatch.setenv("DOCUMIND_PRETRAINED_OCR", "off")
    assert trocr.get_engine(tmp_path / "missing") is None
    monkeypatch.setenv("DOCUMIND_OCR", "tesseract")
    assert trocr.get_engine(tmp_path / "missing") is None


def test_missing_pretrained_model_falls_back_to_tesseract_with_a_warning(monkeypatch, tmp_path):
    from documind.ocr import trocr

    def boom(*a, **k):
        raise OSError("no internet")

    monkeypatch.setattr(trocr, "TrOCRRecognizer", boom)
    monkeypatch.setattr(trocr, "_engine", None); monkeypatch.setattr(trocr, "_tried", False)
    monkeypatch.setattr(trocr, "_pre_engine", None); monkeypatch.setattr(trocr, "_pre_tried", False)
    monkeypatch.delenv("DOCUMIND_OCR", raising=False); monkeypatch.delenv("DOCUMIND_PRETRAINED_OCR", raising=False)
    with pytest.warns(UserWarning, match="Tesseract alone"):
        assert trocr.get_engine(tmp_path / "missing") is None


# ------------------------------------------------------------------------------------------------ the A/B script
def test_compare_scoring_and_photo_simulation():
    from documind.training.compare_reconstruct import score, similarity, to_photo
    assert similarity("Total: 12,50", "total:12,50") == 1.0 and similarity("12,5O", "12,50") < 1.0
    res = score([{"label": "TOTAL", "text": "$ 1,583.04"}, {"label": "DATE", "text": "2O-Mar-2008"}], {"TOTAL": ["$1,583.04"], "DATE": ["20-Mar-2008"], "BUYER": ["Bob"]})
    assert res["TOTAL"] == (1, 1.0) and res["DATE"][0] == 0 and 0.8 < res["DATE"][1] < 1 and res["BUYER"] == (0, 0.0)
    page = np.full((400, 300, 3), 245, np.uint8)
    photo = to_photo(page, np.random.default_rng(1))
    assert photo.shape[0] > 400 and photo.shape[1] > 300 and find_page(photo) is not None          # a table around a tilted page: Stage A can find it


def test_contract_scoring_handles_dates_numbers_and_accents():
    from documind.training.compare_reconstruct import score_contract
    gt = {"title": "CONTRAT DE CRÉDIT", "contract_date": "26/07/2025", "amount_value": 252700, "rate_value": 6.6, "months": 48, "monthly_value": 6004.43, "iban": "MA64 9489"}
    row = {"TITLE": "CONTRAT DE CREDIT", "CONTRACT_DATE": "2025-07-26", "AMOUNT": "383000000.00", "RATE": "6.60", "DURATION_MONTHS": "48", "MONTHLY_PAYMENT": "6004.43", "IBAN": ""}
    r = score_contract(row, gt)
    assert r["TITLE"] == (1, 1.0) and r["CONTRACT_DATE"] == (1, 1.0) and r["RATE_PCT"][0] == 1 and r["DURATION_MONTHS"][0] == 1 and r["MONTHLY_PAYMENT"][0] == 1
    assert r["AMOUNT"][0] == 0 and r["IBAN"][0] == 0


def test_contract_text_reader_applies_stage_b_before_parsing(monkeypatch):
    from PIL import Image
    from documind.inference import contract_text as ct
    seen = {}
    monkeypatch.setattr(ct, "run_ocr", lambda *a, **k: [w("Montant", 10, 10, 90, 30, 40), w("2O000", 100, 10, 160, 30, 30)])
    def fake_reconstruct(gray, words, scale, engine=None, **k):
        seen["called"] = True
        return [{**x, "text": "20000", "refined": True} if x["text"] == "2O000" else x for x in words]
    monkeypatch.setattr(ct.reconstruct, "reconstruct_words", fake_reconstruct)
    monkeypatch.setattr(ct, "parse_terms", lambda words, W, H: (seen.setdefault("texts", [x["text"] for x in words]) and {}, set()))
    monkeypatch.setattr(ct, "find_signatures", lambda gray, words: {})
    monkeypatch.setenv("DOCUMIND_RECONSTRUCT", "on")
    res = ct.read_contract_text(Image.new("RGB", (400, 300), "white"))
    assert seen["called"] and seen["texts"] == ["Montant", "20000"] and res["reconstructed_words"] == 1
    seen.clear()
    monkeypatch.setenv("DOCUMIND_RECONSTRUCT", "off")
    ct.read_contract_text(Image.new("RGB", (400, 300), "white"))
    assert "called" not in seen
