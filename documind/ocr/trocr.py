"""The trained text-line recognizer (TrOCR fine-tuned on the crops of the three datasets, documind.training.train_ocr) as an OCR engine.

    from documind.ocr.trocr import get_engine
    engine = get_engine()                       # None when no trained model is installed (or DOCUMIND_OCR=tesseract)
    engine.read([crop1, crop2])                 # -> ["12,50", "Farine 1kg"]   one PIL image per text line, batched on the GPU

DOCUMIND_OCR (environment variable):
    auto (default)  the trained model AND Tesseract read every box, the better reading is kept (documind.ocr.ensemble); Tesseract alone when models/ocr_trocr
                    does not exist or torch / transformers are not installed
    both            the same, but an error if the trained model is missing
    trocr           the trained model only (an error if it is missing)
    tesseract       Tesseract only (to compare)
DOCUMIND_OCR_PRIMARY (trocr by default) is the engine that decides when both readings look equally good.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from ..config import MODELS_DIR
from .crops import clean_text

MODEL_DIR = MODELS_DIR / "ocr_trocr"
_engine = None
_tried = False


def available(model_dir: Path = MODEL_DIR) -> bool:
    return (Path(model_dir) / "config.json").is_file()


def filter_chars(text: str, allowed: str | None) -> str:
    """Keep only the characters a field can contain (a quantity is digits): the whitelist of the Tesseract path, applied after reading."""
    return text if not allowed else "".join(c for c in text if c in allowed).strip()


class TrOCRRecognizer:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None, num_beams: int = 4, max_new_tokens: int = 72):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(str(model_dir))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).to(self.device).eval()
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens

    def read(self, images: list[Image.Image], batch_size: int = 32) -> list[str]:
        out: list[str] = []
        torch = self.torch
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            pixel = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                ids = self.model.generate(pixel, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
            out += [clean_text(t) for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out


def get_engine(model_dir: Path = MODEL_DIR):
    """The shared recognizer, or None (-> Tesseract). Loaded once, on first use."""
    global _engine, _tried
    mode = os.getenv("DOCUMIND_OCR", "auto").lower()
    if mode == "tesseract":
        return None
    strict = mode in ("trocr", "both")                 # asked for explicitly: a missing model is an error, not a silent fallback
    if _engine is None and (not _tried or strict):
        _tried = True
        if not available(model_dir):
            if strict:
                raise SystemExit(f"DOCUMIND_OCR={mode} but {model_dir} does not exist. Train it: python -m documind.training.train_ocr")
            return None
        try:
            _engine = TrOCRRecognizer(model_dir)
        except ImportError as exc:
            if strict:
                raise SystemExit(f"DOCUMIND_OCR={mode} but a package is missing ({exc}). Run: pip install -r requirements-ocr.txt")
            return None
    return _engine
