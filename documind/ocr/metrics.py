"""Character / word error rates (no dependency): used to compare OCR engines on the same crops."""
from __future__ import annotations


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists of words)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate of one line: edits / characters of the reference (can exceed 1.0 if the hypothesis is much longer)."""
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    return edit_distance(r, hyp.split()) / max(1, len(r))


def summarize(pairs: list[tuple[str, str]]) -> dict:
    """[(reference, hypothesis)] -> {"n", "cer" (total edits / total characters), "wer", "exact" (share of lines read exactly)}."""
    if not pairs:
        return {"n": 0, "cer": None, "wer": None, "exact": None}
    ce = sum(edit_distance(r, h) for r, h in pairs)
    cn = sum(len(r) for r, _ in pairs) or 1
    we = sum(edit_distance(r.split(), h.split()) for r, h in pairs)
    wn = sum(len(r.split()) for r, _ in pairs) or 1
    return {"n": len(pairs), "cer": round(ce / cn, 4), "wer": round(we / wn, 4), "exact": round(sum(r == h for r, h in pairs) / len(pairs), 4)}
