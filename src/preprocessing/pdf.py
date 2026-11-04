"""PDF -> list of page images (PyMuPDF, no external poppler needed)."""
from __future__ import annotations

import cv2
import numpy as np


def pdf_to_images(data: bytes, dpi: int = 200) -> list[np.ndarray]:
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz
    pages: list[np.ndarray] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            pages.append(arr)
    return pages
