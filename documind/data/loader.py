"""Finds images + annotation files in a raw dataset folder and reads the annotations.

Understands COCO, HuggingFace/LayoutLM, "class -> {text, bbox}" JSON (FATURA "original" format) and
most other JSON layouts that contain boxes. The exact coordinate convention (x,y,w,h or x0,y0,x1,y1,
pixels or normalised) is NOT guessed here: `labeling.align_labels` tests all of them against the OCR
words and keeps the one that fits.

Check yours with:  python -m documind.data.inspect_dataset --data data/raw/FATURA
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

Box = tuple[float, float, float, float]


@dataclass
class RawBox:
    label: str
    vals: tuple[float, float, float, float]   # as written in the file
    hint: str | None = None                   # "xywh" / "xyxy" when the format tells us, else None
    text: str | None = None                   # transcription of the box when the file has one (used by the audit)


@dataclass
class Annotation:
    boxes: list[RawBox] = field(default_factory=list)
    ref_size: tuple[float, float] | None = None   # (width, height) the file says the image has


# ------------------------------------------------------------------ discovery
_COLOR_TOKENS = re.compile(r"(?i)[_\-\s]*(colou?red?|white|bg|background)[_\-\s]*")


def _pair_key(stem: str) -> str:
    """Stem without 'color' / 'white' / 'bg' tokens, so Template1_Instance0_color finds Template1_Instance0.json."""
    return _COLOR_TOKENS.sub("", stem).lower()


def _is_colored(path: Path) -> bool:
    return bool(re.search(r"(?i)colou?r", "/".join(path.parts[-4:])))


def find_samples(root: str | Path, *, include_colored: bool = False) -> list[dict]:
    """Pair every image with its annotation JSON(s) (same file stem).

    FATURA ships every invoice twice (white and coloured background, same file name). By default one
    copy per invoice is kept (the white one); `include_colored=True` keeps both.
    """
    root = Path(root)
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    ann_by_stem: dict[str, list[Path]] = defaultdict(list)
    ann_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*.json"):
        ann_by_stem[p.stem].append(p)
        ann_by_key[_pair_key(p.stem)].append(p)

    samples = []
    for img in sorted(images):
        cands = ann_by_stem.get(img.stem) or ann_by_key.get(_pair_key(img.stem), [])
        if not cands:
            continue
        # COCO copy first (it has category names), then the others
        cands = sorted(cands, key=lambda p: (0 if "coco" in str(p).lower() else 1, str(p)))
        samples.append({"image": img, "annotations": cands, "stem": img.stem,
                        "template": template_of(img.stem), "colored": _is_colored(img)})

    if not include_colored:
        best: dict[str, dict] = {}
        for s in samples:
            k = _pair_key(s["stem"])
            if k not in best or (best[k]["colored"] and not s["colored"]):
                best[k] = s
        samples = sorted(best.values(), key=lambda s: str(s["image"]))
    return samples


def template_of(stem: str) -> str:
    """FATURA names are like Template12_Instance34 -> template 'Template12'."""
    m = re.match(r"(?i)^(template[\s_-]*\d+)", stem)
    if m:
        return m.group(1).lower()
    return re.sub(r"(?i)[_-]?(instance|img|image|inv|invoice)?[_-]?\d+$", "", stem) or stem


def select_samples(samples: list[dict], per_template: int | None, max_images: int | None, seed: int) -> list[dict]:
    """per_template None / 0 = use EVERY image. max_images None / 0 = no global cap."""
    rng = np.random.default_rng(seed)
    if per_template and per_template > 0:
        by_t: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_t[s["template"]].append(s)
        picked = []
        for t in sorted(by_t):
            items = by_t[t]
            take = rng.permutation(len(items))[:per_template]
            picked.extend(items[i] for i in take)
        samples = picked
    if max_images and max_images > 0 and len(samples) > max_images:
        samples = [samples[i] for i in rng.permutation(len(samples))[:max_images]]
    return samples


# ----------------------------------------------------------- annotation parsing
def normalize_label(label) -> str:
    """'Seller Name' / 'B-SELLER_NAME' / 'seller-name' -> 'SELLER_NAME'."""
    s = str(label).strip()
    if re.match(r"^[BIESL]-\S", s):
        s = s[2:]
    return re.sub(r"[\s\-]+", "_", s).upper()


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_box(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(_num(x) for x in v)


_GENERIC_KEYS = {"annotations", "annotation", "objects", "items", "fields", "shapes", "regions", "labels",
                 "data", "form", "entities", "boxes", "bboxes", "words", "tokens", "results", "result",
                 "output", "document", "documents", "images", "categories", "info", "license", "licenses",
                 "ocr", "lines", "content", "pages", "page"}
_BOX_KEYS = ("bbox", "box", "bounding_box", "boundingbox", "bndbox", "bounding_poly", "polygon", "poly",
             "points", "coordinates", "rect", "rectangle", "position", "location", "geometry", "segmentation")
_LABEL_KEYS = ("label", "class", "category", "category_name", "class_name", "field", "tag", "entity", "type", "name")


def _to_box(v) -> tuple[tuple[float, float, float, float], str | None] | None:
    """Anything that looks like a rectangle / polygon -> (4 numbers, hint)."""
    if _is_box(v):
        return tuple(float(x) for x in v), None
    if isinstance(v, (list, tuple)):
        if len(v) >= 6 and len(v) % 2 == 0 and all(_num(x) for x in v):            # flat polygon
            xs, ys = v[0::2], v[1::2]
            return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))), "xyxy"
        if len(v) >= 2 and all(isinstance(p, (list, tuple)) and len(p) >= 2 and _num(p[0]) and _num(p[1]) for p in v):
            xs, ys = [p[0] for p in v], [p[1] for p in v]                           # [[x, y], ...]
            return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))), "xyxy"
        return None
    if isinstance(v, dict):
        g = {str(k).lower(): x for k, x in v.items()}
        for names in (("xmin", "ymin", "xmax", "ymax"), ("x0", "y0", "x1", "y1"),
                      ("left", "top", "right", "bottom"), ("x1", "y1", "x2", "y2")):
            if all(_num(g.get(n)) for n in names):
                return tuple(float(g[n]) for n in names), "xyxy"
        for wn, hn in (("w", "h"), ("width", "height")):
            if all(_num(g.get(n)) for n in ("x", wn, "y", hn)):
                return (float(g["x"]), float(g["y"]), float(g[wn]), float(g[hn])), "xywh"
        for k in _BOX_KEYS:
            if k in g:
                return _to_box(g[k])
    return None


def _label_of(d: dict, fallback: str | None):
    for k in _LABEL_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return fallback


def read_annotation(path: str | Path) -> Annotation:
    """Read every labelled box of one annotation file, without guessing coordinate conventions."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    ann = Annotation()

    # ---- COCO: {"images": [...], "annotations": [{"bbox": [x, y, w, h], "category_id": n}], "categories": [...]}
    if isinstance(data, dict) and isinstance(data.get("annotations"), list) and (
            "categories" in data or "images" in data):
        cats = {c.get("id"): c.get("name") for c in data.get("categories", []) if isinstance(c, dict)}
        imgs = data.get("images") or []
        if imgs and isinstance(imgs[0], dict) and _num(imgs[0].get("width")) and _num(imgs[0].get("height")):
            ann.ref_size = (float(imgs[0]["width"]), float(imgs[0]["height"]))
        for a in data["annotations"]:
            if not isinstance(a, dict):
                continue
            b = _to_box(a.get("bbox"))
            if b is None:
                continue
            cid = a.get("category_id")
            name = cats.get(cid) or _label_of(a, None) or f"class_{cid}"
            ann.boxes.append(RawBox(normalize_label(name), b[0], "xywh" if _is_box(a.get("bbox")) else b[1]))
        return ann

    # ---- HuggingFace / LayoutLM: {"words"|"tokens": [...], "bboxes"|"boxes": [...], "ner_tags"|"labels": [...]}
    if isinstance(data, dict):
        g = {str(k).lower(): v for k, v in data.items()}
        words = g.get("words") or g.get("tokens")
        boxes = g.get("bboxes") or g.get("boxes") or g.get("bbox")
        tags = g.get("ner_tags") or g.get("labels") or g.get("tags") or g.get("label")
        if isinstance(words, list) and isinstance(boxes, list) and boxes and len(boxes) == len(words):
            names = g.get("label_names") or g.get("id2label") or g.get("ner_tag_names") or {}
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}
            for i, b in enumerate(boxes):
                tb = _to_box(b)
                if tb is None:
                    continue
                t = tags[i] if isinstance(tags, list) and i < len(tags) else 0
                label = names.get(t, names.get(str(t), t)) if isinstance(names, dict) else t
                label = f"class_{label}" if isinstance(label, int) else str(label)
                ann.boxes.append(RawBox(normalize_label(label), tb[0], tb[1]))
            if ann.boxes:
                return ann

    # ---- anything else: walk the JSON and collect every dict/list that carries a box
    def walk(obj, key_label: str | None, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(obj, dict):
            lower = {str(k).lower(): v for k, v in obj.items()}
            if ann.ref_size is None and _num(lower.get("width")) and _num(lower.get("height")) and depth <= 1:
                ann.ref_size = (float(lower["width"]), float(lower["height"]))
            b = None
            for k in _BOX_KEYS:
                if k in lower:
                    b = _to_box(lower[k])
                    if b:
                        break
            if b is None:
                b = _to_box(obj)
            if b is not None:
                label = _label_of(lower, key_label)
                if label is not None:
                    txt = lower.get("text")
                    ann.boxes.append(RawBox(normalize_label(label), b[0], b[1], txt if isinstance(txt, str) else None))
                return
            for k, v in obj.items():
                ks = str(k)
                walk(v, None if ks.lower() in _GENERIC_KEYS else ks, depth + 1)
        elif isinstance(obj, list):
            if _is_box(obj) and key_label is not None:
                ann.boxes.append(RawBox(normalize_label(key_label), tuple(float(x) for x in obj), None))
                return
            for it in obj:
                walk(it, key_label, depth + 1)

    walk(data, None)
    return ann


def to_pixels(vals, mode: str, W: float, H: float, ref: tuple[float, float] | None = None) -> np.ndarray:
    """Convert raw boxes (n, 4) to pixel x0,y0,x1,y1 under one coordinate convention.

    mode = "<shape>[_<unit>]"; shape: xyxy | xywh; unit: (pixels) | n1 (0-1) | n1000 (0-1000) | ref
    (pixels of the image size written in the annotation file, rescaled to the real image) | flipy
    (pixels, but the origin is the BOTTOM-left corner, as in PDF files: y is counted from the bottom edge).

    FATURA's "Original_Format" annotations use that bottom-left origin: for an 841 px high image the TABLE box
    is written y = 328.9 -> 220.9, which is y = 512 -> 620 once flipped (the same value as in the HF file).
    """
    v = np.asarray(vals, dtype=np.float64).reshape(-1, 4)
    shape, _, unit = mode.partition("_")
    if shape == "xywh":
        v = np.stack([v[:, 0], v[:, 1], v[:, 0] + v[:, 2], v[:, 1] + v[:, 3]], axis=1)
    if unit == "flipy":
        top, bottom = H - v[:, 3], H - v[:, 1]
        v = np.stack([v[:, 0], np.minimum(top, bottom), v[:, 2], np.maximum(top, bottom)], axis=1)
    if unit == "n1":
        v = v * np.array([W, H, W, H])
    elif unit == "n1000":
        v = v * np.array([W, H, W, H]) / 1000.0
    elif unit == "ref" and ref:
        v = v * np.array([W / ref[0], H / ref[1], W / ref[0], H / ref[1]])
    return v


def parse_annotation(path: Path, image_size: tuple[int, int]) -> list[tuple[str, Box]]:
    """Return [(label, (x0, y0, x1, y1))] using the format's own hint (COCO = x,y,w,h) and simple
    normalisation detection. For robust alignment use `labeling.align_labels` instead."""
    W, H = image_size
    ann = read_annotation(path)
    out: list[tuple[str, Box]] = []
    top = max((max(b.vals) for b in ann.boxes), default=0.0)
    for b in ann.boxes:
        shape = "xywh" if b.hint == "xywh" else "xyxy"
        unit = ""
        if top <= 1.5:
            unit = "_n1"
        elif top <= 1000 and max(W, H) > 1100 and b.hint is None:
            unit = "_n1000"
        x0, y0, x1, y1 = to_pixels([b.vals], shape + unit, W, H)[0]
        out.append((b.label, (float(x0), float(y0), float(x1), float(y1))))
    return out
