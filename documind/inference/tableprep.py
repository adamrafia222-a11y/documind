"""Image cleanups that make tables readable for Tesseract. Each returns (grey image, scale vs. the original, skew angle removed). Word boxes stay in the straightened page frame
(rows horizontal), which is what the line grouping needs. They are used as alternative readings: no single cleanup suits every document (ruled grids, dark header
bands, phone photos, tiny scans), so the extractor tries several and keeps the most consistent table.
"""
import cv2
import numpy as np

def up(img, cap=4.0, target=2400):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = float(np.clip(target / max(g.shape[:2]), 1.0, cap))
    return (cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1.01 else g), (s if s > 1.01 else 1.0)
def invert_bands(g):
    m = (g < 140).astype(np.uint8); H, W = g.shape
    c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 50) | 1, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(c); out = g.copy()
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if w > 0.25 * W and 0.008 * H < h < 0.09 * H and m[y:y + h, x:x + w].mean() > 0.55:
            out[y:y + h, x:x + w] = 255 - g[y:y + h, x:x + w]
    return out
def _rules(g):
    """(horizontal, vertical) masks of thin ruled lines. Filled areas (header bands, text blobs) are left out."""
    blk = (max(31, (min(g.shape) // 40) | 1))
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 15); L = max(g.shape) // 30
    hm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    t = max(7, int(0.007 * max(g.shape)))                       # whatever survives a t x t opening is a filled area, not a line
    thick = cv2.dilate(cv2.morphologyEx(cv2.add(hm, vm), cv2.MORPH_OPEN, np.ones((t, t), np.uint8)), np.ones((5, 5), np.uint8))
    nt = cv2.bitwise_not(thick)
    return cv2.bitwise_and(hm, nt), cv2.bitwise_and(vm, nt)


def strip_lines(g):
    hm, vm = _rules(g)
    keep = cv2.add(hm, vm)
    frac = cv2.countNonZero(keep) / keep.size
    if frac < 0.0004 or frac > 0.05: return g, False
    out = g.copy(); out[cv2.dilate(keep, np.ones((3, 3), np.uint8)) > 0] = 255; return out, True


def _grid_global(g, hm, vm):
    """First attempt (unchanged): lines are the rows / columns of the whole picture whose ruled length is at least half of the longest one."""
    H, W = g.shape
    def pos(m, axis, minlen):
        prof = m.sum(axis=axis) / 255.0
        idx = np.where(prof >= max(minlen, 0.5 * prof.max()))[0]
        if not len(idx): return [], []
        groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        return [int(gp.mean()) for gp in groups], [len(gp) for gp in groups]
    ys, yt = pos(hm, 1, 0.15 * W)
    xs, xt = pos(vm, 0, 0.05 * H)
    if len(ys) < 3 or len(xs) < 3: return None
    return ys, xs, int(max(yt + xt)) // 2 + 3


def _grid_local(g, hm, vm):
    """Second attempt, local to the table: the page border, a frame or a shadow can be the longest line of the picture and make every inner
    rule of the table look short. Here the table is found from its own horizontal rules, and its vertical rules are looked for inside that box only."""
    H, W = g.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(hm, connectivity=8)
    segs = [(int(x), int(x + w), y + h / 2.0, int(h)) for x, y, w, h, _a in st[1:]
            if 0.2 * W <= w < 0.97 * W and h <= max(10, 0.02 * H) and 0.02 * H < y < 0.98 * H]          # not a page frame / edge: those lines span the picture or hug its border
    if len(segs) < 3: return None
    widest = max(b - a for a, b, _y, _h in segs)
    main = [sg for sg in segs if sg[1] - sg[0] >= 0.6 * widest]
    if len(main) < 3: return None
    x0, x1 = int(np.median([a for a, _b, _y, _h in main])), int(np.median([b for _a, b, _y, _h in main]))
    main = [sg for sg in main if min(sg[1], x1) - max(sg[0], x0) >= 0.7 * (x1 - x0)]        # the rules that span the same width = this table
    ys = sorted(sg[2] for sg in main)
    merged = [[ys[0]]]
    for y in ys[1:]:
        merged[-1].append(y) if y - merged[-1][-1] <= 4 else merged.append([y])
    ys = [int(np.mean(m)) for m in merged]
    if len(ys) < 3: return None
    y0, y1 = ys[0], ys[-1]
    sub = vm[max(0, y0 - 2): y1 + 3, max(0, x0 - 4): x1 + 5]
    prof = sub.sum(axis=0) / 255.0
    idx = np.where(prof >= 0.5 * (y1 - y0))[0]
    xs = []
    if len(idx):
        groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        xs = [int(gp.mean()) + max(0, x0 - 4) for gp in groups]
    for edge in (x0, x1):                                                                  # an open table (no outer vertical rule): its width gives the edge
        if not any(abs(edge - x) <= 8 for x in xs): xs.append(edge)
    xs = sorted(xs)
    if len(xs) < 3: return None
    thick = float(np.median([sg[3] for sg in main]))
    return ys, xs, int(thick) // 2 + 3


def grid(g):
    """Cells of a fully ruled table: ([(x0, y0, x1, y1), ...], (table box), inset) in the pixels of g, or None."""
    hm, vm = _rules(g)
    found = _grid_global(g, hm, vm) or _grid_local(g, hm, vm)
    if not found: return None
    ys, xs, inset = found
    cells = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(len(ys) - 1) for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > 14 and ys[j + 1] - ys[j] > 14]
    if not 6 <= len(cells) <= 400: return None
    return cells, (xs[0], ys[0], xs[-1], ys[-1]), inset


def denoise(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.fastNlMeansDenoising(g, None, h=9, templateWindowSize=5, searchWindowSize=15)
def normalize(g):
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((max(15, g.shape[0] // 40) | 1,) * 2, np.uint8)), (0, 0), max(10, g.shape[0] // 60))
    return np.clip(cv2.divide(g, bg, scale=255), 0, 255).astype(np.uint8)
def sharpen(g): return cv2.addWeighted(g, 1.6, cv2.GaussianBlur(g, (0, 0), 2.0), -0.6, 0)
def skew(g: np.ndarray) -> float:
    """Skew angle in degrees (counter-clockwise to apply to straighten the page), from the sharpness of the row profile."""
    s = 700.0 / max(g.shape[1], 1)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    ink = (cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0).astype(np.float32)
    h, w = ink.shape
    def sharp(a: float) -> float:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        return float(np.var(cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST).sum(axis=1)))
    coarse = max(np.arange(-4.0, 4.01, 0.5), key=sharp)
    best = max(np.arange(coarse - 0.45, coarse + 0.46, 0.15), key=sharp)
    return float(best) if abs(best) >= 0.3 and sharp(best) > 1.03 * sharp(0.0) else 0.0


def straighten(img: np.ndarray) -> tuple[np.ndarray, float]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    a = skew(g)
    if not a:
        return g, 0.0
    h, w = g.shape
    return cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h), flags=cv2.INTER_CUBIC, borderValue=255), a


def light(img): g, a = straighten(img); g, s = up(g); return g, s, a
def flat(img):
    """Straightened, upscaled, shading flattened, header bands inverted (rules still in place)."""
    g, a = straighten(img); g, s = up(g); return invert_bands(normalize(g)), s, a


def clean(img): g, a = straighten(img); g, s = up(g); g2, _ = strip_lines(invert_bands(normalize(g))); return g2, s, a
def denoised(img):
    g, a = straighten(denoise(img)); g, s = up(g); g = sharpen(normalize(invert_bands(g))); g, _ = strip_lines(g); return g, s, a
def binarized(img):
    g, a = straighten(denoise(img)); g, s = up(g); g = sharpen(normalize(invert_bands(g))); g, _ = strip_lines(g)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, (max(31, g.shape[0] // 35) | 1), 12); return b, s, a
