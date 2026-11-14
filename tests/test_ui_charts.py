"""The animated charts of the dashboard and the dashboard helpers, checked without a browser: well-formed, escaped, the numbers drawn are the numbers given.
    pytest -q tests/test_ui_charts.py
"""
import datetime
import re

from documind.ui import charts as C
from documind.ui import helpers as H
from documind.ui import theme as T
from test_ui_theme import EVIL, check


# ------------------------------------------------------------------------------------------------ the charts
def pieces():
    return [C.sparkline_svg([1, 3, 2, 5]), C.sparkline_svg([]), C.sparkline_svg([0, 0, 0]), C.sparkline_svg([4]),
            C.kpi_card_html("📚", "Documents", 12, "of 40", [1, 2, 3], ("▲ 4 vs the previous 30 days", "up")), C.kpi_card_html("🎯", "Confidence", "91 %"),
            C.donut_html([("Invoice", 3, "#60a5fa"), ("Cheque", 1, "#34d399")], "4", "documents"), C.donut_html([], "0", "documents"),
            C.hbars_html([("DATE", 40, "var(--bad)"), ("TOTAL", 90, "var(--ok)")], 100, "{:.0f} %"), C.hbars_html([]),
            C.histogram_html([0, 1, 0, 2, 5, 3, 0, 1, 0, 2]), C.heatmap_html([[0] * 24 for _ in range(7)]), C.activity_html([{"file": "a.png", "doc_type": "cheque", "when": "now", "conf": 0.9}]),
            C.activity_html([]), C.card_html("Title", "<p>x</p>", "sub")]


def test_every_chart_is_well_formed():
    for p in pieces():
        check(p)


def test_every_class_a_chart_uses_exists_in_the_css():
    used = {c for p in pieces() for cls in re.findall(r'class="([^"]+)"', p) for c in cls.split()}
    missing = {c for c in used if f".{c}" not in T.CSS}
    assert not missing, missing


def test_text_from_the_data_cannot_inject_html():
    evil = [C.kpi_card_html(EVIL, EVIL, EVIL, EVIL, [1, 2], (EVIL, EVIL)), C.donut_html([(EVIL, 2, "#fff")], EVIL, EVIL), C.hbars_html([(EVIL, 5, "red")], None, "{:g}"),
            C.histogram_html([1, 2], [EVIL, EVIL]), C.activity_html([{"file": EVIL, "doc_type": "invoice", "when": EVIL, "conf": 0.5}]), C.card_html(EVIL, "<p>fine</p>", EVIL)]
    for p in evil:
        assert "<script" not in p and "<img" not in p and 'onerror="' not in p, p[:160]
        check(p)


def test_the_numbers_drawn_are_the_numbers_given():
    bars = C.hbars_html([("A", 50, "red"), ("B", 100, "blue"), ("C", 0, "green")], 100)
    assert re.findall(r"--w:([\d.]+)", bars) == ["50.0", "100.0", "0.0"] and "--w:100" in C.hbars_html([("A", 5, "red")])                 # the longest bar is full by default
    assert [float(v) for v in re.findall(r"--w:([\d.]+)", C.hbars_html([("A", 500, "red")], 100))] == [100.0]                                # never longer than the track
    h = C.histogram_html([0, 5, 10], ["a", "b", "c"])
    assert re.findall(r"height:([\d.]+)%", h) == ["0.0", "50.0", "100.0"]
    ten = C.histogram_html([1] * 10)
    assert re.findall(r"background:(var\(--\w+\))", ten) == ["var(--bad)"] * 6 + ["var(--warn)"] * 3 + ["var(--ok)"]                          # below 60 red, 60-90 amber, the top bin green
    d = C.donut_html([("A", 3, "#111"), ("B", 1, "#222")], "4", "docs")
    assert "75%" in d and "25%" in d and d.count('class="seg"') == 2 and "<b>4</b>" in d
    assert C.donut_html([("A", 0, "#111")], "0", "docs").count('class="seg"') == 0                                                            # nothing to draw: only the empty ring
    heat = C.heatmap_html([[0] * 24, [3] + [0] * 23, [6] + [0] * 23] + [[0] * 24] * 4)
    a = [float(x) for x in re.findall(r"--a:([\d.]+)", heat)]
    assert len(a) == 7 * 24 and max(a) == 1.0 and a[24] < a[48] and min(a) == 0.05 and 'title="Tue 00:00 · 3 documents"' in heat and 'title="Wed 00:00 · 6 documents"' in heat
    assert 'title="Mon 05:00 · 0 documents"' in heat and heat.count('class="c"') == 168


def test_kpi_counts_whole_numbers_and_shows_the_delta_and_the_sparkline():
    k = C.kpi_card_html("📚", "Documents", 12, "of 40", [1, 2, 3], ("▲ 4 vs the previous 30 days", "up"))
    assert 'style="--to:12"' in k and "dm-delta up" in k and "▲ 4 vs the previous 30 days" in k and "dm-spark" in k and "of 40" in k
    plain = C.kpi_card_html("🎯", "Confidence", "91 %")
    assert "dm-count" not in plain and "dm-spark" not in plain and "dm-delta" not in plain and "91 %" in plain


def test_a_sparkline_with_nothing_in_it_is_still_a_valid_flat_line():
    for values in ([], [0, 0], [7]):
        svg = C.sparkline_svg(values)
        assert "nan" not in svg.lower() and "inf" not in svg.lower() and 'class="line"' in svg
    assert C.sparkline_svg([1, 2]) == C.sparkline_svg([1, 2]) and C.sparkline_svg([1, 2]) != C.sparkline_svg([2, 1])                       # same data = same picture (and its own gradient id)


# ------------------------------------------------------------------------------------------------ the helpers
def stats_fixture():
    return {"per_day_type": {"2026-11-12": {"invoice": 2, "cheque": 1}, "2026-11-11": {"invoice": 1}, "2026-11-05": {"receipt": 4}, "2026-10-20": {"invoice": 3}}}


def test_range_summary():
    today = datetime.date(2026, 11, 12)
    s = H.range_summary(stats_fixture(), 7, today)
    assert s["spark"] == [0, 0, 0, 0, 0, 1, 3] and s["docs"] == 4 and s["by_type"] == {"invoice": 3, "cheque": 1}
    assert [r for r in s["rows"] if r["date"] == "2026-11-12"] == [{"date": "2026-11-12", "type": "Invoice", "documents": 2}, {"date": "2026-11-12", "type": "Cheque", "documents": 1}]
    assert s["prev"] == 4                                                                                                                    # 5 Nov = the day before the 7-day window began
    month = H.range_summary(stats_fixture(), 30, today)
    assert month["docs"] == 11 and len(month["spark"]) == 30 and month["prev"] == 0 and H.range_summary(stats_fixture(), 90, today)["prev"] is None


def test_delta_chip():
    assert H.delta_chip(10, 6, 30) == ("▲ 4 vs the previous 30 days", "up") and H.delta_chip(2, 5, 7) == ("▼ 3 vs the previous 7 days", "down")
    assert H.delta_chip(4, 4, 30) == ("= same vs the previous 30 days", "flat") and H.delta_chip(0, 0, 30) is None and H.delta_chip(3, None, 90) is None


def test_warning_labels_are_readable():
    assert H.warning_label("amount_mismatch") == "Amount in figures ≠ amount in words" and H.warning_label("brand_new_code") == "Brand new code"
    assert H.warning_label("no_signature_lender") != H.warning_label("no_signature_borrower")
