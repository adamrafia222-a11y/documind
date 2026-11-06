"""Training-time augmentation: make clean documents look like phone photos."""
from __future__ import annotations

import cv2
import numpy as np


def augment_photo_like(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make a clean document look like a phone photo (no geometric change, so labels stay valid)."""
    out = img.astype(np.float32)
    out = out * rng.uniform(0.7, 1.2) + rng.uniform(-30, 20)               # brightness / contrast
    h, w = out.shape[:2]
    gradient = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.1), w, dtype=np.float32)
    out = out * gradient[None, :, None]                                     # uneven lighting
    out = out + rng.normal(0, rng.uniform(2, 8), out.shape)                 # sensor noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    k = int(rng.choice([1, 3, 3, 5]))
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)                              # slight blur
    quality = int(rng.integers(40, 90))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])  # compression artefacts
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else out
