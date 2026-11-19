"""Inference: image / PDF  ->  fields  ->  one CSV row."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from . import tableprep
from ..config import OCR_LANG, OCR_PSM, PDF_DPI
from ..ocr import reconstruct, run_ocr
from ..preprocessing import canonicalize, remove_table_lines, decode_image, pdf_to_images, preprocess
from .export import csv_columns, fields_to_row, rows_to_csv
from . import fieldrules, invoice_checks, invoice_yolo
from .postprocess import _score_items, apply_rules, items_to_text


class ModelNotFound(RuntimeError):
    pass


class FieldExtractor:
    def __init__(self, lang: str | None = None, psm: int | None = None) -> None:
        """The invoice fields come from the YOLO detector (models/invoices_yolo.pt), like receipts, cheques and contracts; the line items from the rules reader."""
        if not invoice_yolo.model_available():
            raise ModelNotFound(f"Invoice detector not found: {invoice_yolo.MODEL_PATH}. Train it: python -m documind.data.prepare_invoices --fatura data/raw/FATURA, "
                                "then python -m documind.training.train_invoices")
        self.yolo = invoice_yolo.load_model()
        self.classes: list[str] = [str(n).upper() for _, n in sorted(self.yolo.names.items())]
        self.meta: dict = {"backend": "yolo", "model": str(invoice_yolo.MODEL_PATH), "imgsz": int(self.yolo.overrides.get("imgsz", 1280))}
        self.lang = lang or OCR_LANG        # not the language saved at training time (FATURA = "eng"): French invoices need "eng+fra"
        self.psm = psm or OCR_PSM
        self.rules_psm = int(os.getenv("RULES_PSM", "4"))   # the table / totals rules read the page with psm 4 (columns)
        self.use_rules = os.getenv("USE_RULES", "1") != "0"
        self.rebuild_words = reconstruct.words_active()               # Stage B (words re-read by the OCR model) only with DOCUMIND_RECONSTRUCT=on

    @property
    def columns(self) -> list[str]:
        return csv_columns(self.classes)

    def _ocr_padded(self, img: np.ndarray, scale: float, psm: int) -> list[dict]:
        """OCR with a white margin: Tesseract loses the text that touches the border of the image (last table column)."""
        pad = int(0.04 * max(img.shape[:2]))
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        out = run_ocr(big, scale=scale, lang=self.lang, psm=psm)
        d = pad / scale
        for w in out:
            w["x0"] -= d; w["x1"] -= d; w["y0"] -= d; w["y1"] -= d
        return out

    def _read_cells(self, orig: np.ndarray, page_words: list[dict]) -> list[dict] | None:
        """Fully ruled table: read every cell on its own (as a small block, zoomed) instead of the whole page at once. Tesseract on a
        crowded grid mixes up or drops cells; alone, a cell is read reliably. The text outside the table comes from the page reading."""
        g, s, _ = tableprep.flat(orig)
        found = tableprep.grid(g)
        if not found:
            return None
        cells, (bx0, by0, bx1, by1), m = found
        out = [w for w in page_words if not (bx0 / s - 3 <= (w["x0"] + w["x1"]) / 2 <= bx1 / s + 3 and by0 / s - 3 <= (w["y0"] + w["y1"]) / 2 <= by1 / s + 3)]
        for x0, y0, x1, y1 in cells:
            crop = g[y0 + m:y1 - m, x0 + m:x1 - m]
            if crop.size == 0:
                continue
            if float(crop.mean()) < 110 and float((crop > 170).mean()) > 0.002:         # white text on a dark band (the header row): read it as dark text on white
                crop = 255 - crop
            if float((crop < 150).mean()) < 0.002:
                continue
            cf = float(np.clip(70.0 / crop.shape[0], 1.0, 3.0))
            if cf > 1.05:
                crop = cv2.resize(crop, None, fx=cf, fy=cf, interpolation=cv2.INTER_CUBIC)
            p = 12
            crop = cv2.copyMakeBorder(crop, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
            cw = run_ocr(crop, scale=cf * s, lang=self.lang, psm=6)
            if not cw:
                continue
            lh = float(np.median([w["y1"] - w["y0"] for w in cw]))
            text = " ".join(w["text"] for w in sorted(cw, key=lambda w: (round(((w["y0"] + w["y1"]) / 2) / max(lh, 1e-6)), w["x0"])))
            dx, dy = (x0 + m) / s - p / (cf * s), (y0 + m) / s - p / (cf * s)
            cy = (min(w["y0"] for w in cw) + max(w["y1"] for w in cw)) / 2 + dy      # a value wrapped onto two lines ("1,583.04" / "MAD") stays one cell
            out.append({**cw[0], "text": text, "conf": float(np.mean([w["conf"] for w in cw])), "x0": min(w["x0"] for w in cw) + dx,
                        "x1": max(w["x1"] for w in cw) + dx, "y0": cy - lh / 2, "y1": cy + lh / 2})
        return out

    def _reading_makers(self, prep, words: list[dict], orig: np.ndarray) -> list:
        """Ways to read the page again for the rules, cheapest first (the OCR is fragile on tables). They are tried one
        after the other and the search stops as soon as the table and the totals are consistent."""
        img, scale = prep.image, prep.scale
        clean, has_lines = remove_table_lines(img)
        base = clean if has_lines else img
        med = float(np.median([(w["y1"] - w["y0"]) * scale for w in words]))
        cache: dict = {}
        def tp(fn):
            def read():
                g, s, _ = fn(orig)
                cache[fn.__name__] = self._ocr_padded(g, s, self.rules_psm)
                return cache[fn.__name__]
            return read
        def cells():
            page = cache.get("clean") or tp(tableprep.clean)()
            return self._read_cells(orig, page)
        makers = [tp(tableprep.clean), cells,                                                  # grid rules erased, dark header bands inverted
                  tp(tableprep.denoised),                                               # + denoise / shading / sharpen (phone photos)
                  tp(tableprep.binarized),                                              # + adaptive binarisation (tiny or low-contrast scans)
                  tp(tableprep.light),                                                  # plain grey + upscale
                  lambda: self._ocr_padded(img, scale, self.rules_psm),                  # the model's preprocessing, columns mode
                  lambda: words,                                                        # the reading the model used (free)
                  lambda: run_ocr(img, scale=scale, lang=self.lang, psm=self.rules_psm)]
        if has_lines:
            makers.append(lambda: self._ocr_padded(clean, scale, self.rules_psm))        # table borders erased (model preprocessing)
        f = min(2.0, 26.0 / max(med, 1.0), 7000.0 / max(img.shape))
        if med < 24 and f > 1.15:                                                        # small text: higher zoom
            zoomed = lambda im: cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            makers.append(lambda: run_ocr(zoomed(img), scale=scale * f, lang=self.lang, psm=self.rules_psm))
            makers.append(lambda: self._ocr_padded(zoomed(base), scale * f, self.rules_psm))
        return makers

    # ------------------------------------------------------------------ pages
    def extract_image(self, img: np.ndarray, page: int = 1) -> dict:
        ow, oh = img.shape[1], img.shape[0]                      # the ORIGINAL picture: the screen draws the boxes on it
        canon = canonicalize(img, "invoice") if reconstruct.pages_active() else None
        if canon is not None and not canon.identity:
            img = canon.image                                    # Stage A: the page cut out, upright, evenly lit (the form the model was trained on)
        else:
            canon = None
        h, w = img.shape[:2]
        prep = preprocess(img)                                   # full pre-processing (with deskew)
        words = run_ocr(prep.image, scale=prep.scale, lang=self.lang, psm=self.psm, handwriting=True)
        if self.rebuild_words:
            words = reconstruct.reconstruct_words(prep.image, words, prep.scale)      # Stage B: re-read by the trained recognizer, canonical spelling
        # one box per field (like the receipts, cheques, contracts), each box read by the OCR engines
        from PIL import Image
        fields = invoice_yolo.fields_from_dets(Image.fromarray(img[:, :, ::-1].copy()), invoice_yolo.detect(self.yolo, img), page, self.lang)
        if words:                                                # the labels printed on the page fill the fields the detector did not find
            fields = sorted(fields + fieldrules.fill_missing(fields, words, page, self.classes, (float(w), float(h))), key=lambda f: (f["bbox"][1], f["bbox"][0]))
        items, items_check = [], {}
        if self.use_rules and words:
            low = float(np.mean([x["conf"] for x in words])) < 50          # unreadable scan: more readings will not help
            readings: list[list[dict]] = []
            post = None
            for make in self._reading_makers(prep, words, img):
                rd = make()
                if not rd:
                    continue
                readings.append(rd)
                post = apply_rules(readings[0], fields, float(w), float(h), page, alt_readings=readings[1:])
                if post["confident"] or (len(readings) >= 3 and (low or not post["table_seen"])):
                    break
            if post and not post["confident"] and len(readings) > 1:
                # no reading was fully consistent: use the one whose table is the most self-consistent as the main reading
                # (the first one is not always the best: it depends on the image which cleanup reads it best)
                cands = []
                for k, rd in enumerate(readings):
                    cand = apply_rules(rd, fields, float(w), float(h), page, alt_readings=readings[:k] + readings[k + 1:])
                    cands.append((bool(cand["confident"]), _score_items(cand["items"]) if cand["items"] else -99.0, -k, cand))
                post = max(cands, key=lambda t: t[:3])[3]
            if post:
                fields, items, items_check = post["fields"], post["items"], post["items_check"]
        if canon is not None:
            canon.map_back(fields, items)                        # boxes back on the original picture
        return {"page": page, "n_words": len(words), "mean_ocr_conf": float(np.mean([x["conf"] for x in words])) if words else 0.0,
                "fields": fields, "items": items, "items_check": items_check, "deskew_angle": prep.angle,
                "size": [int(ow), int(oh)], **({"canonical_steps": canon.steps} if canon is not None else {}),
                "reconstructed_words": sum(1 for x in words if x.get("refined"))}

    def extract_bytes(self, data: bytes, filename: str = "document") -> dict:
        is_pdf = filename.lower().endswith(".pdf") or data[:5] == b"%PDF-"
        images = pdf_to_images(data, PDF_DPI) if is_pdf else [decode_image(data)]
        pages = [self.extract_image(img, page=i + 1) for i, img in enumerate(images)]
        fields = [f for p in pages for f in p["fields"]]
        items = [it for p in pages for it in p.get("items", [])]
        warnings = []
        for p in pages:
            if p["n_words"] and p["mean_ocr_conf"] < 60:
                warnings.append({"code": "low_quality", "page": p["page"], "message":
                                 f"Low scan quality (image {p['size'][0]}x{p['size'][1]} px, OCR confidence {p['mean_ocr_conf']:.0f}%): "
                                 "the text is too small or blurry to read reliably, so the table and some fields may be missing. "
                                 "Use the original PDF or a scan of at least 150 dpi (about 1200 px wide)."})
                break
        for p in pages:
            if max(p["size"]) < 1000 and not warnings:
                warnings.append({"code": "low_resolution", "page": p["page"], "message":
                                 f"Small image ({p['size'][0]}x{p['size'][1]} px): digits in the table can be misread at this size. "
                                 "A larger photo, or the original PDF, reads more reliably."})
                break
        chk = next((p["items_check"] for p in pages if p.get("items_check")), {})
        if items and chk.get("matches_subtotal") is False:
            warnings.append({"code": "items_mismatch", "message":
                             f"The line items add up to {chk.get('items_sum')} but the invoice prints a different subtotal: check the amounts."})
        inv = invoice_checks.run_checks(fields, items, chk)
        invoice_checks.adjust_confidence(fields, inv["checks"])          # the checks are evidence: confirmed fields go up, contradicted ones go down
        seen_items = any(w["code"] == "items_mismatch" for w in warnings)
        warnings += [w for w in inv["warnings"] if not (seen_items and w.get("check") == "items_sum")]
        return {
            "file": filename,
            "warnings": warnings,
            "checks": inv["checks"],
            "n_pages": len(pages),
            "pages": [{k: v for k, v in p.items() if k not in ("fields", "items")} for p in pages],
            "fields": fields,
            "items": items,
            "items_check": next((p["items_check"] for p in pages if p.get("items_check")), {}),
            "row": {"file": filename, **fields_to_row(fields), "N_ITEMS": str(len(items)), "CHECKS": inv["summary"], "ITEMS": items_to_text(items)},
        }

    def to_csv(self, results: list[dict]) -> bytes:
        return rows_to_csv([r["row"] for r in results], self.columns)
