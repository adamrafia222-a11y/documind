"""Animated charts for the dashboard, drawn as inline SVG / HTML + CSS (no chart library, no script): they animate when they appear and work offline.
Every function returns a string; every text that comes from data goes through html.escape. The CSS they need is in documind.ui.theme.CSS.

  sparkline_svg   a small line that draws itself            donut_html      a ring whose segments sweep round, with its legend
  hbars_html      horizontal bars that grow                 histogram_html  vertical bars that rise (red / amber / green by value)
  heatmap_html    weekday x hour grid, cells fade in        activity_html   a list of recent items that slide in
  kpi_card_html   a number that counts up + a delta + a sparkline
"""
from __future__ import annotations

import hashlib
import math
from html import escape

from . import theme as T

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _delay_class(delay: int) -> str:
    return f" dm-d{min(int(delay), 4)}" if delay and delay > 0 else ""


def _uid(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8]


def sparkline_svg(values: list[float], color: str = "#8b5cf6", w: int = 130, h: int = 38) -> str:
    """A line through `values` that draws itself from left to right, with a soft area under it. A flat line when there is nothing to show."""
    vals = [float(v) for v in values] or [0.0]
    top = max(vals) or 1.0
    n = len(vals)
    pts = [(2 + (i * (w - 4) / (n - 1) if n > 1 else (w - 4) / 2), h - 4 - (v / top) * (h - 10)) for i, v in enumerate(vals)]
    line = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    area = line + f" L {pts[-1][0]:.1f} {h} L {pts[0][0]:.1f} {h} Z"
    gid = "sg" + _uid(color, *vals)
    return (f'<svg class="dm-spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".45"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<path class="area" d="{area}" fill="url(#{gid})"/><path class="line" d="{line}" pathLength="1" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle class="dot" cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3.2" fill="{color}"/></svg>')


def kpi_card_html(icon: str, label: str, value: int | str, sub: str = "", spark: list[float] | None = None, delta: tuple[str, str] | None = None, color: str = "#8b5cf6", delay: int = 0) -> str:
    """A big number (it counts up when it is a whole number), a small caption, an optional delta chip ("▲ 4 vs previous 30 days", kind up / down / flat) and an optional sparkline."""
    val = T.count_html(value) if isinstance(value, int) else escape(str(value))
    chip = f'<span class="dm-delta {escape(delta[1])}">{escape(delta[0])}</span>' if delta else ""
    return (f'<div class="dm-card dm-kpi dm-up{_delay_class(delay)}"><div class="top"><div class="ico">{escape(icon)}</div><div class="lab">{escape(label)}</div></div>'
            f'<div class="val">{val}</div><div class="foot"><div>{chip}{f"<small class=dm-muted>{escape(sub)}</small>" if sub else ""}</div>{sparkline_svg(spark, color) if spark is not None else ""}</div></div>')


def donut_html(parts: list[tuple[str, float, str]], center_top: str = "", center_bottom: str = "", size: int = 176) -> str:
    """parts: [(label, value, colour)]. A ring whose segments sweep round one after the other; the legend lists value and share."""
    total = float(sum(max(0.0, v) for _, v, _ in parts))
    r, sw = 70.0, 22.0
    circ = 2 * math.pi * r
    segs, offset = [], 0.0
    for i, (label, v, color) in enumerate(parts):
        if total <= 0 or v <= 0:
            continue
        length = max(0.0, v / total * circ - 3.0)
        segs.append(f'<circle class="seg" cx="88" cy="88" r="{r:.0f}" fill="none" stroke="{escape(color)}" stroke-width="{sw:.0f}" stroke-dasharray="{length:.1f} {circ:.1f}" '
                    f'stroke-dashoffset="{-offset:.1f}" style="--len:{length:.1f};--c:{circ:.1f};animation-delay:{i * 0.18:.2f}s"><title>{escape(label)}: {v:g}</title></circle>')
        offset += v / total * circ
    track = f'<circle cx="88" cy="88" r="{r:.0f}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="{sw:.0f}"/>'
    legend = "".join(f'<div class="row"><i style="background:{escape(color)}"></i><span>{escape(label)}</span><b>{v:g}</b><small>{(v / total * 100 if total else 0):.0f}%</small></div>' for label, v, color in parts)
    return (f'<div class="dm-donut"><div class="ring" style="width:{size}px;height:{size}px"><svg viewBox="0 0 176 176" width="{size}" height="{size}"><g transform="rotate(-90 88 88)">{track}{"".join(segs)}</g></svg>'
            f'<div class="mid"><b>{escape(str(center_top))}</b><small>{escape(center_bottom)}</small></div></div><div class="dm-legend">{legend}</div></div>')


def hbars_html(items: list[tuple[str, float, str]], max_value: float | None = None, fmt: str = "{:g}", empty: str = "Nothing to show yet.") -> str:
    """items: [(label, value, colour)] -> horizontal bars that grow from the left (the longest = max_value, default the largest value)."""
    if not items:
        return f'<p class="dm-muted">{escape(empty)}</p>'
    top = max_value if max_value else (max(v for _, v, _ in items) or 1.0)
    rows = "".join(f'<div class="dm-hbar"><span class="l" title="{escape(label)}">{escape(label)}</span><div class="t"><i style="--w:{min(100.0, max(0.0, v / top * 100)):.1f};background:{escape(c)};animation-delay:{k * 0.07:.2f}s"></i></div>'
                   f'<b>{escape(fmt.format(v))}</b></div>' for k, (label, v, c) in enumerate(items))
    return f'<div class="dm-hbars">{rows}</div>'


def histogram_html(counts: list[int], labels: list[str] | None = None, height: int = 150) -> str:
    """One vertical bar per bin that rises from the bottom; the colour follows the position (the lower bins red, the middle amber, the top green)."""
    n = len(counts)
    top = max(counts) or 1
    labels = labels or [f"{i * 100 // n}" for i in range(n)]
    bars = []
    for i, c in enumerate(counts):
        color = "var(--bad)" if i < n * 0.6 else "var(--warn)" if i < n * 0.85 else "var(--ok)"
        bars.append(f'<div class="col" title="{escape(labels[i])}: {c}"><small>{c if c else ""}</small><div class="bar"><i style="height:{c / top * 100:.1f}%;background:{color};animation-delay:{i * 0.06:.2f}s"></i></div>'
                    f'<span>{escape(labels[i])}</span></div>')
    return f'<div class="dm-hist" style="height:{height}px">{"".join(bars)}</div>'


def heatmap_html(matrix: list[list[int]]) -> str:
    """7 rows (Mon-Sun) x 24 columns (hours): the more documents, the brighter the cell. Cells fade in diagonally."""
    peak = max((max(r) for r in matrix), default=0) or 1
    head = '<div class="h"></div>' + "".join(f'<div class="h">{h if h % 3 == 0 else ""}</div>' for h in range(24))
    body = []
    for d, row in enumerate(matrix):
        body.append(f'<div class="d">{DAYS[d]}</div>')
        for h, v in enumerate(row):
            a = 0.06 + 0.94 * (v / peak) if v else 0.05
            body.append(f'<div class="c" title="{DAYS[d]} {h:02d}:00 · {v} document{"s" if v != 1 else ""}" style="--a:{a:.2f};animation-delay:{min(1.6, (d + h) * 0.035):.2f}s"></div>')
    return f'<div class="dm-heat">{head}{"".join(body)}</div>'


def activity_html(rows: list[dict]) -> str:
    """rows: [{file, doc_type, when (text), conf (0-1)}]: the latest documents, each sliding in after the one before."""
    if not rows:
        return '<p class="dm-muted">No document yet.</p>'
    items = "".join(f'<div class="item" style="animation-delay:{i * 0.09:.2f}s">{T.type_badge_html(r.get("doc_type", "invoice"))}<div class="nm"><b>{escape(str(r.get("file", "")))}</b>'
                    f'<small>{escape(str(r.get("when", "")))}</small></div><span class="cf">{round(float(r.get("conf") or 0) * 100)}%</span></div>' for i, r in enumerate(rows))
    return f'<div class="dm-feed">{items}</div>'


def card_html(title: str, body: str, subtitle: str = "", delay: int = 0) -> str:
    """A titled glass card around a chart. `body` is HTML made by the functions of this module (the text in it is already escaped); title and subtitle are escaped here."""
    return (f'<div class="dm-card dm-chart dm-up{_delay_class(delay)}"><h4>{escape(title)}</h4>{f"<small class=dm-muted>{escape(subtitle)}</small>" if subtitle else ""}<div class="body">{body}</div></div>')
