"""Page layout: group OCR words into rows (independent of Tesseract's own lines)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Layout:
    line_of: list[int]            # word index -> line index
    lines: list[list[int]]        # line index -> word indices sorted left to right


def assign_lines(words: list[dict]) -> Layout:
    """Group words in rows using their vertical centres (independent of Tesseract's own lines)."""
    n = len(words)
    order = sorted(range(n), key=lambda i: (words[i]["y0"] + words[i]["y1"]) / 2)
    groups: list[dict] = []
    for i in order:
        w = words[i]
        cy = (w["y0"] + w["y1"]) / 2
        h = max(1.0, w["y1"] - w["y0"])
        if groups and abs(cy - groups[-1]["cy"]) <= 0.5 * max(h, groups[-1]["h"]):
            g = groups[-1]
            g["idx"].append(i)
            k = len(g["idx"])
            g["cy"] = g["cy"] * (k - 1) / k + cy / k
            g["h"] = max(g["h"], h)
        else:
            groups.append({"cy": cy, "h": h, "idx": [i]})
    lines = [sorted(g["idx"], key=lambda j: words[j]["x0"]) for g in groups]
    line_of = [0] * n
    for li, idxs in enumerate(lines):
        for j in idxs:
            line_of[j] = li
    return Layout(line_of=line_of, lines=lines)


# ------------------------------------------------------------------ cells and blocks
@dataclass
class Structure:
    """Cells = groups of words on one row that are close together (a key, a value, a table cell...).
    Blocks = cells stacked on top of each other and aligned (an address, a paragraph, a table)."""
    cell_of: list[int]               # word index -> cell index
    cells: list[dict]                # {"ids": [...], "line": int, "x0","y0","x1","y1","h": float, "pos": int (index in its line)}
    cells_in_line: list[list[int]]   # line index -> cell indices, left to right
    block_of_cell: list[int]         # cell index -> block index
    blocks: list[list[int]]          # block index -> cell indices, top to bottom (then left to right)


def build_structure(words: list[dict], layout: Layout, gap_factor: float = 1.2,
                    v_gap_factor: float = 1.0) -> Structure:
    cells: list[dict] = []
    cells_in_line: list[list[int]] = []
    cell_of = [0] * len(words)
    for li, ids in enumerate(layout.lines):
        hs = sorted(max(1.0, words[j]["y1"] - words[j]["y0"]) for j in ids)
        h = hs[len(hs) // 2]
        cur: list[int] = []
        groups: list[list[int]] = []
        for j in ids:
            if cur and words[j]["x0"] - words[cur[-1]]["x1"] > gap_factor * h:
                groups.append(cur)
                cur = []
            cur.append(j)
        if cur:
            groups.append(cur)
        line_cells = []
        for pos, g in enumerate(groups):
            ci = len(cells)
            for j in g:
                cell_of[j] = ci
            cells.append({"ids": g, "line": li, "pos": pos, "h": h,
                          "x0": min(words[j]["x0"] for j in g), "x1": max(words[j]["x1"] for j in g),
                          "y0": min(words[j]["y0"] for j in g), "y1": max(words[j]["y1"] for j in g)})
            line_cells.append(ci)
        cells_in_line.append(line_cells)

    parent = list(range(len(cells)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for li in range(len(cells_in_line) - 1):
        for a in cells_in_line[li]:
            ca = cells[a]
            for b in cells_in_line[li + 1]:
                cb = cells[b]
                h = max(ca["h"], cb["h"])
                if cb["y0"] - ca["y1"] > v_gap_factor * h:
                    continue
                overlap = min(ca["x1"], cb["x1"]) - max(ca["x0"], cb["x0"])
                narrow = min(ca["x1"] - ca["x0"], cb["x1"] - cb["x0"])
                if overlap >= 0.2 * max(narrow, 1.0) or abs(ca["x0"] - cb["x0"]) <= 1.5 * h:
                    parent[find(b)] = find(a)
    groups: dict[int, list[int]] = {}
    for ci in range(len(cells)):
        groups.setdefault(find(ci), []).append(ci)
    blocks = [sorted(g, key=lambda c: (cells[c]["line"], cells[c]["x0"])) for g in groups.values()]
    blocks.sort(key=lambda g: (cells[g[0]]["line"], cells[g[0]]["x0"]))
    block_of_cell = [0] * len(cells)
    for bi, g in enumerate(blocks):
        for c in g:
            block_of_cell[c] = bi
    return Structure(cell_of, cells, cells_in_line, block_of_cell, blocks)


def cell_above(struct: Structure, ci: int, max_lines: int = 6) -> int | None:
    """Nearest cell above `ci` (within max_lines rows) that overlaps it horizontally."""
    c = struct.cells[ci]
    for lj in range(c["line"] - 1, max(-1, c["line"] - 1 - max_lines), -1):
        best, best_ov = None, 0.0
        for cj in struct.cells_in_line[lj]:
            d = struct.cells[cj]
            ov = min(c["x1"], d["x1"]) - max(c["x0"], d["x0"])
            if ov > 0.2 * max(min(c["x1"] - c["x0"], d["x1"] - d["x0"]), 1.0) and ov > best_ov:
                best, best_ov = cj, ov
        if best is not None:
            return best
    return None
