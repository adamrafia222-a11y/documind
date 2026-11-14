"""The look of the Streamlit front end, checked without a browser: the HTML pieces are well formed, nothing typed by a person can inject HTML, the CSS is
self-contained and every animation it uses is defined. (How it LOOKS can only be judged in a browser.)
    pytest -q tests/test_ui_theme.py
"""
import re
from html.parser import HTMLParser

import pytest

from documind.ui import theme as T

VOID = {"br", "hr", "img", "input", "meta", "link", "path", "circle", "rect", "stop"}          # path / circle / rect / stop are self-closing in the SVG we write


class Balanced(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.tags = [], [], set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.add(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


def check(html: str) -> Balanced:
    p = Balanced()
    p.feed(html)
    p.close()
    assert not p.errors and not p.stack, (p.errors, p.stack)
    return p


EVIL = '<img src=x onerror="alert(1)"><script>alert(2)</script>'


def test_every_html_piece_is_well_formed():
    pieces = [T.brand_html(), T.avatar_html("Gogo Dev"), T.user_chip_html("Gogo", "g@x.com"), T.page_header_html("History", "sub", "🕘"), T.type_badge_html("cheque"),
              T.chip_html("12 fields", "ok"), T.count_html(7), T.stat_card_html("📚", "Documents", 12, "all", 2), T.stat_card_html("🎯", "Confidence", "91 %"),
              T.stat_grid_html([("📚", "Documents", 3, ""), ("⚠️", "Warnings", "2", "look")]), T.ring_html(91), T.ring_html(40, "40"), T.hero_html(), T.reading_html("a.png", 3, 2),
              T.done_html("Read 2 of 3"), T.skeleton_html(), T.empty_html("🗂️", "No history", "Read a document")]
    for p in pieces:
        check(p)
    assert "svg" in check(T.brand_html()).tags and "svg" in check(T.done_html("x")).tags


def test_nothing_typed_by_a_person_can_inject_html():
    pieces = [T.avatar_html(EVIL), T.user_chip_html(EVIL, EVIL), T.page_header_html(EVIL, EVIL, EVIL), T.type_badge_html(EVIL), T.chip_html(EVIL), T.stat_card_html(EVIL, EVIL, EVIL, EVIL),
              T.ring_html(50, EVIL), T.reading_html(EVIL), T.done_html(EVIL), T.empty_html(EVIL, EVIL, EVIL), T.brand_html(EVIL)]
    for p in pieces:
        assert "<script" not in p and "<img" not in p and 'onerror="' not in p, p[:120]
        check(p)
    assert "&lt;script&gt;" in T.user_chip_html(EVIL, EVIL)


def test_css_is_self_contained_and_animations_are_defined():
    css = T.CSS
    assert css.count("{") == css.count("}") and "http://" not in css and "https://" not in css and "@import" not in css and "url(" not in css          # no CDN, no font to download
    defined = set(re.findall(r"@keyframes\s+([\w-]+)", css))
    used = {m for decl in re.findall(r"animation:([^;}]+)", css) for m in re.findall(r"\b(dm-[\w-]+)", decl)}
    assert used and used <= defined, used - defined
    assert "prefers-reduced-motion" in css and "@property --n" in css and "@property --p" in css
    assert T.AUTH_CSS.count("{") == T.AUTH_CSS.count("}") and "stSidebar" in T.AUTH_CSS


def test_classes_used_by_the_pieces_exist_in_the_css():
    used = set()
    for p in (T.brand_html(), T.user_chip_html("a", "b"), T.page_header_html("t"), T.type_badge_html("invoice"), T.stat_grid_html([("x", "y", 1, "z")]), T.ring_html(50), T.hero_html(),
              T.reading_html("a"), T.done_html("x"), T.skeleton_html(), T.empty_html("x", "y", "z"), T.chip_html("x", "warn")):
        used |= {c for cls in re.findall(r'class="([^"]+)"', p) for c in cls.split()}
    missing = {c for c in used if f".{c}" not in T.CSS}
    assert not missing, missing


def test_small_helpers():
    assert T.initials("Gogo Dev") == "GD" and T.initials("gogo@example.com") == "GE" and T.initials("") == "?" and T.initials("  ") == "?"
    assert T.avatar_html("Gogo") == T.avatar_html("Gogo") and T.avatar_html("Gogo") != T.avatar_html("Omar")                     # a stable colour per person
    assert "--to:85" in T.ring_html(85) and "var(--ok)" in T.ring_html(90) and "var(--warn)" in T.ring_html(70) and "var(--bad)" in T.ring_html(30)
    assert "--to:100" in T.ring_html(180) and "--to:0" in T.ring_html(-5)                                                          # always between 0 and 100
    assert 'style="--to:42"' in T.count_html(42) and "<i>42</i>" in T.count_html(42)                                                 # the real number stays in the page
    assert T.stat_card_html("x", "y", 5).count("dm-count") == 1 and "dm-count" not in T.stat_card_html("x", "y", "5 s")                # only whole numbers count up
    assert T.type_badge_html("contract").count("📑") == 1 and T.TYPE_COLOR.keys() == T.TYPE_LABEL.keys() == T.TYPE_ICON.keys()
    assert T.reading_html("a.png", 3, 2).count("dm-step cyc") == 4 and "(2/3)" in T.reading_html("a.png", 3, 2)
    assert T.hero_html().count('class="bx"') == 5 and "dm-scan" not in T.hero_html() and 'class="scan"' in T.hero_html()
