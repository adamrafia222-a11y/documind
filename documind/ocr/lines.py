"""Reading order of boxes on a tilted page, and how a value that wraps over two lines is split between its boxes.

A scanned page is a little rotated (up to a couple of degrees): the right end of a line is higher or lower than its left end, so sorting boxes by their
y alone can put the second line of a wrapped value before the first. The tilt is measured on the picture (documind.preprocessing.image.estimate_skew)
and removed from the box centres before sorting.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def skew_angle(gray: np.ndarray, max_angle: float = 3.0, step: float = 0.25) -> float:
    """Degrees to rotate the picture by so that its text lines are horizontal (same convention as preprocessing.estimate_skew)."""
    from ..preprocessing.image import estimate_skew
    return estimate_skew(gray, max_angle=max_angle, step=step)


def reading_order(boxes: list, angle: float, size: tuple[int, int]) -> list[int]:
    """Indices of `boxes` (each [x0, y0, x1, y1]) in reading order: top to bottom after removing the tilt, left to right on the same line."""
    w, h = size
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    keyed = []
    for i, b in enumerate(boxes):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x, y = m[0, 0] * cx + m[0, 1] * cy + m[0, 2], m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
        keyed.append((y, x, i))
    typical = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 1.0
    keyed.sort()
    out, line = [], []
    for k in keyed:                                    # boxes less than half a line apart are on the same line: left to right
        if line and k[0] - line[0][0] > 0.5 * typical:
            out += [i for _, _, i in sorted(line, key=lambda t: t[1])]
            line = []
        line.append(k)
    out += [i for _, _, i in sorted(line, key=lambda t: t[1])]
    return out


def split_value(text: str, widths: list[float], max_error: float = 0.08, margin: float = 0.5) -> list[str] | None:
    """The words of `text` cut into len(widths) consecutive parts whose lengths in characters best follow the box widths (given in reading order).
    Returns None when no cut is clearly better than the others (the part texts would be a guess). Only 2 or 3 boxes."""
    words = text.split()
    n = len(widths)
    if n < 2 or n > 3 or len(words) < n:
        return None
    total_w = float(sum(widths))
    lens = [len(w) for w in words]
    total_c = sum(lens) + len(words) - 1

    def err(cuts: tuple[int, ...]) -> float:
        bounds, e = (0, *cuts, len(words)), 0.0
        for k in range(n):
            part = words[bounds[k]:bounds[k + 1]]
            chars = sum(len(w) for w in part) + len(part) - 1
            e += abs(chars / total_c - widths[k] / total_w)
        return e
    if n == 2:
        cands = [((c,), err((c,))) for c in range(1, len(words))]
    else:
        cands = [((a, b), err((a, b))) for a in range(1, len(words) - 1) for b in range(a + 1, len(words))]
    cands.sort(key=lambda t: t[1])
    best = cands[0]
    if best[1] > max_error * n or (len(cands) > 1 and best[1] > margin * cands[1][1]):
        return None
    bounds = (0, *best[0], len(words))
    return [" ".join(words[bounds[k]:bounds[k + 1]]) for k in range(n)]
