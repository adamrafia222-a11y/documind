"""Merge neighbouring words that share a predicted label into fields."""
from __future__ import annotations

import numpy as np

from ..config import BACKGROUND_LABEL
from ..features import assign_lines


def group_fields(words: list[dict], labels: list[str], conf: np.ndarray, page: int) -> list[dict]:
    """Words with the same label on a row become one field; vertically adjacent ones (addresses) are joined."""
    layout = assign_lines(words)
    segments: list[dict] = []
    for ids in layout.lines:
        cur: dict | None = None
        for i in ids:
            lab = labels[i]
            if lab == BACKGROUND_LABEL:
                cur = None
                continue
            w = words[i]
            h = max(1.0, w["y1"] - w["y0"])
            if cur and cur["label"] == lab and w["x0"] - cur["x1"] <= 2.0 * max(h, cur["h"]):
                cur["texts"].append(w["text"]); cur["confs"].append(float(conf[i]))
                cur["x1"] = max(cur["x1"], w["x1"]); cur["y0"] = min(cur["y0"], w["y0"]); cur["y1"] = max(cur["y1"], w["y1"])
            else:
                cur = {"label": lab, "texts": [w["text"]], "confs": [float(conf[i])], "x0": w["x0"], "x1": w["x1"],
                       "y0": w["y0"], "y1": w["y1"], "h": h}
                segments.append(cur)

    # multi-line fields (addresses...): join vertically adjacent segments with the same label
    merged: list[dict] = []
    for seg in sorted(segments, key=lambda s: (s["label"], s["y0"], s["x0"])):
        prev = merged[-1] if merged else None
        if (prev and prev["label"] == seg["label"] and 0 <= seg["y0"] - prev["y1"] <= 0.8 * max(seg["h"], prev["h"])
                and abs(seg["x0"] - prev["x0"]) <= 2.0 * max(seg["h"], prev["h"])):
            prev["texts"] += seg["texts"]; prev["confs"] += seg["confs"]
            prev["x0"] = min(prev["x0"], seg["x0"]); prev["x1"] = max(prev["x1"], seg["x1"]); prev["y1"] = seg["y1"]
        else:
            merged.append(seg)

    return [
        {"label": s["label"], "text": " ".join(s["texts"]), "confidence": round(float(np.mean(s["confs"])), 3),
         "bbox": [round(s["x0"], 1), round(s["y0"], 1), round(s["x1"], 1), round(s["y1"], 1)], "page": page}
        for s in sorted(merged, key=lambda s: (s["y0"], s["x0"]))
    ]
