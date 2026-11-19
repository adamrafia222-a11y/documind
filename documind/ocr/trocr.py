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
    if not allowed:
        return text
    if "-" in allowed:
        text = "".join("-" if c in "–—−" else c for c in text)             # 150–00 (en / em dash) is 150-00
    return "".join(c for c in text if c in allowed).strip()


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


def _read_nbest(self, image: Image.Image, n: int = 5) -> list[str]:
    """The n most likely readings of ONE crop (beam search), best first, duplicates removed. A handwritten "Quarante-huit" read as "Quarante" is often the 2nd or 3rd choice."""
    torch = self.torch
    pixel = self.processor(images=[image.convert("RGB")], return_tensors="pt").pixel_values.to(self.device)
    with torch.no_grad():
        ids = self.model.generate(pixel, num_beams=max(self.num_beams, n), num_return_sequences=n, max_new_tokens=self.max_new_tokens)
    texts = [clean_text(t) for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
    return list(dict.fromkeys(t for t in texts if t))


TrOCRRecognizer.read_nbest = _read_nbest


PRETRAINED = os.getenv("RECONSTRUCT_OCR_MODEL", "microsoft/trocr-base-printed")     # or microsoft/trocr-small-printed (3x faster), or any TrOCR folder / hub id
_pre_engine = None
_pre_tried = False


def pretrained_engine():
    """A PRETRAINED printed-text recognizer (no training needed; the first use downloads ~1.3 GB from Hugging Face, then it is cached), or None.

    Used when models/ocr_trocr (your fine-tuned one) does not exist, so a picture that is uploaded is still re-read by a real OCR model.
    DOCUMIND_PRETRAINED_OCR=off disables it; RECONSTRUCT_OCR_MODEL chooses the model.
    """
    global _pre_engine, _pre_tried
    if os.getenv("DOCUMIND_PRETRAINED_OCR", "on").lower() in ("0", "off", "false", "no"):
        return None
    if _pre_engine is None and not _pre_tried:
        _pre_tried = True
        try:
            _pre_engine = TrOCRRecognizer(PRETRAINED)                              # a hub id: kept as a string (a Path would break it on Windows)
        except Exception as exc:                                                   # no torch / transformers, no internet and no cache: Tesseract alone
            import warnings
            warnings.warn(f"pretrained OCR model {PRETRAINED!r} not available ({type(exc).__name__}: {exc}); Tesseract alone. "
                          "Install: pip install -r requirements.txt (in the environment with PyTorch)", stacklevel=2)
    return _pre_engine


def get_engine(model_dir: Path = MODEL_DIR):
    """The shared recognizer: your fine-tuned one (models/ocr_trocr) if it exists, else the pretrained printed one, else None (-> Tesseract). Loaded once."""
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
        else:
            try:
                _engine = TrOCRRecognizer(model_dir)
            except ImportError as exc:
                if strict:
                    raise SystemExit(f"DOCUMIND_OCR={mode} but a package is missing ({exc}). Run: pip install -r requirements-ocr.txt")
    return _engine if _engine is not None else pretrained_engine()
