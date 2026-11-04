"""Rule layer: line items + amounts, on the words of a real (modern) invoice that FATURA does not look like."""
from documind.inference.postprocess import apply_rules, extract_line_items, items_to_text


def W(text, x0, cy, h=9.0):
    return {"text": text, "x0": x0, "x1": x0 + 5.5 * len(text), "y0": cy - h / 2, "y1": cy + h / 2, "conf": 90.0}


def invoice_words():
    rows = [
        (221, [("ary", 45), ("DESCRIPTION", 140), ("UNIT", 282), ("PRICE", 302), ("AMOUNT", 393)]),   # "QTY" misread by OCR
        (242, [("1", 52), ("Front", 80), ("and", 100), ("rear", 115), ("brake", 133), ("cables", 153), ("100.00", 300), ("100.00", 402)]),
        (262, [("2", 51), ("New", 79), ("set", 97), ("of", 110), ("pedal", 120), ("arms", 141), ("15.00", 305), ("20.00", 406)]),  # damaged: 30.00
        (282, [("3", 51), ("Labor", 80), ("3hrs", 102), ("5.00", 309), ("15.00", 406)]),
        (310, [("Subtotal", 294), ("145.00", 402)]),
        (331, [("Sales", 265), ("Tax", 286), ("6.25%", 302), ("9.06", 410)]),
        (351, [("TOTAL", 298), ("$154.06", 389)]),
        (161, [("PO#", 287)]), (168, [("2312/2019", 395)]),           # OCR split the row in two lines
        (183, [("DUE", 287), ("DATE", 303), ("26/02/2019", 393)]),
    ]
    return [W(t, x, cy) for cy, ws in rows for t, x in ws]


def test_line_items_columns_and_repair():
    r = apply_rules(invoice_words(), [], 465, 659)
    it = r["items"]
    assert [i["qty"] for i in it] == ["1", "2", "3"]
    assert [i["description"] for i in it] == ["Front and rear brake cables", "New set of pedal arms", "Labor 3hrs"]
    assert [i["unit_price"] for i in it] == ["100.00", "15.00", "5.00"]
    assert [i["amount"] for i in it] == ["100.00", "30.00", "15.00"]            # 20.00 repaired = 2 x 15.00, sum == subtotal
    assert [i["corrected"] for i in it] == [False, True, False]
    assert (r["items_check"]["items_sum"], r["items_check"]["matches_subtotal"]) == (145.0, True)
    assert "2 x New set of pedal arms @ 15.00 = 30.00" in items_to_text(it)


def test_amounts_and_pairs_are_read_by_their_label():
    f = {x["label"]: x["text"] for x in apply_rules(invoice_words(), [], 465, 659)["fields"]}
    assert f["SUB_TOTAL"] == "145.00" and f["TAX"] == "9.06" and f["TOTAL"] == "$154.06"
    assert f["PO_NUMBER"] == "2312/2019" and f["DUE_DATE"] == "26/02/2019"


def test_model_values_are_kept_and_only_gaps_are_filled():
    model = [{"label": "TOTAL", "text": "Total: 999.99", "confidence": 0.9, "bbox": [0, 0, 1, 1], "page": 1},
             {"label": "SUB_TOTAL", "text": "Subtotal", "confidence": 0.9, "bbox": [0, 0, 1, 1], "page": 1}]
    f = {x["label"]: x["text"] for x in apply_rules(invoice_words(), model, 465, 659)["fields"]}
    assert f["TOTAL"] == "Total: 999.99"          # has an amount -> the model's answer stays
    assert f["SUB_TOTAL"] == "145.00"             # no amount in the model's text -> replaced by the rule


def test_no_table_no_items():
    assert extract_line_items([W("Hello", 10, 20), W("world", 50, 20)], 465, 659)[0] == []


def totals_block(tax1="14.76", total="USS$633.36", tax2="27.60"):
    rows = [(357, [("Subtotal", 404), ("591.00", 530)]), (373, [("Sales", 380), ("Tax", 410), ("6%", 430), (tax1, 535)]),
            (389, [("Tax", 412), ("2", 430), ("%", 440), (tax2, 530)]), (417, [("Total", 406), (total, 520)])]
    return [W(t, x, cy) for cy, ws in rows for t, x in ws]


def test_currency_glued_to_the_total_and_two_tax_lines():
    f = {x["label"]: x["text"] for x in apply_rules(totals_block(), [], 595, 842)["fields"]}
    assert f["TOTAL"] == "USS$633.36"            # "US$" read as "USS$" by the OCR is still an amount
    assert f["TAX"] == "42.36"                   # 14.76 + 27.60: every tax line between Subtotal and Total counts
    assert f["SUB_TOTAL"] == "591.00"


def test_damaged_digits_are_fixed_by_the_other_reading_or_the_arithmetic():
    # reading 1 lost the decimal point of the first tax line; reading 2 has it right -> the consistent pair wins
    f = {x["label"]: x["text"] for x in apply_rules(totals_block(tax1="1476"), [], 595, 842, alt_words=totals_block())["fields"]}
    assert f["TAX"] == "42.36"
    # a single damaged reading: total - subtotal gives the tax and it is flagged as computed
    r = apply_rules(totals_block(tax1="1476", tax2="27.40"), [], 595, 842)
    tax = next(x for x in r["fields"] if x["label"] == "TAX")
    assert tax["text"] == "42.36" and tax["source"] == "computed"


def test_left_label_on_the_same_row_as_the_subtotal_does_not_become_an_item():
    words = [w for w in invoice_words() if w["y0"] < 290]                       # header + the 3 rows
    rows = [(310, [("Stamp", 40), ("and", 70), ("signature", 90), ("Subtotal", 294), ("145.00", 402)]),
            (331, [("Sales", 265), ("Tax", 286), ("6.25%", 302), ("9.06", 410)]), (351, [("TOTAL", 298), ("$154.06", 389)])]
    words += [W(t, x, cy) for cy, ws in rows for t, x in ws]
    r = apply_rules(words, [], 465, 659)
    assert len(r["items"]) == 3 and r["items_check"]["matches_subtotal"] is True
    assert {x["label"]: x["text"] for x in r["fields"]}["SUB_TOTAL"] == "145.00"


def grid_invoice_words():
    """Table with grid lines, an ID column and amounts written with a space as thousands separator."""
    def row(cy, items):
        return [W(t, x, cy) for t, x in items]
    return (row(500, [("ID", 70), ("|", 95), ("Description", 108), ("|", 300), ("Price", 381), ("QTY", 464), ("Total", 794)])
            + row(534, [("1", 72), ("Logo", 112), ("design", 148), ("$2", 364), ("500", 378), ("1", 482), ("$2", 745), ("500.00", 759), ("|", 840)])
            + row(760, [("SUBTOTAL", 670), ("$2", 745), ("500.00", 759)])
            + row(786, [("DISCOUNT", 670), ("$0.00", 770)])
            + row(842, [("INVOICE", 650), ("TOTAL", 700), ("$2", 745), ("500.00", 759)]))


def test_space_as_thousands_separator_grid_noise_and_id_column():
    r = apply_rules(grid_invoice_words(), [], 900, 1273)
    it = r["items"]
    assert len(it) == 1
    assert (it[0]["id"], it[0]["qty"], it[0]["description"]) == ("1", "1", "Logo design")
    assert (it[0]["unit_price"], it[0]["amount"]) == ("$2 500", "$2 500.00")
    assert r["items_check"]["matches_subtotal"] is True
    f = {x["label"]: x["text"] for x in r["fields"]}
    assert f["SUB_TOTAL"] == "$2 500.00" and f["TOTAL"] == "$2 500.00" and f["DISCOUNT"] == "$0.00"


def test_column_gaps_are_not_mistaken_for_a_thousands_space():
    # qty 2 then a price 500.00 in the next column: far apart -> two separate numbers
    from documind.inference.postprocess import _lines
    ln = _lines([W("2", 100, 50), W("500.00", 160, 50)])[0]
    assert [w["text"] for w in ln["words"]] == ["2", "500.00"]


def gst_words():
    """GST invoice: 3-line stacked headers, 11 columns, a Total row inside the table, OCR noise "(96)" for "(%)"."""
    cx = [39, 155, 280, 382, 463, 541, 619, 700, 782, 864, 945]
    heads = [["Sr.", "No."], ["Description of", "Goods/Services"], ["HSN/SAC", "Code"], ["Quantity"], ["Unit", "Price", "(R)"],
             ["Taxable", "Value", "(R)"], ["CGST", "(96)"], ["CCGST", "Amount", "(R)"], ["SGST", "(%)"], ["SGST", "Amount", "(R)"], ["Total", "(R)"]]
    ws = []
    for c, hd in zip(cx, heads):
        for k, t in enumerate(hd):
            ws.append(W(t, c - 3 * len(t), 480 + (3 - len(hd) + k) * 18, h=11))
    rows = [["1", "Product A", "1234", "10", "500", "5000", "9%", "450", "9%", "450", "5900"],
            ["2", "Product B", "5678", "5", "1000", "5000", "9%", "450", "9%", "450", "5900"]]
    for y, r in zip((560, 600), rows):
        for i, (c, t) in enumerate(zip(cx, r)):
            ws.append(W(t, (c - 50) if i == 1 else c - 3 * len(t), y, h=11))
    ws.append(W("Total", 14, 640, h=11))
    for c, t in ((541, "10000"), (700, "900"), (864, "900"), (945, "11800")):
        ws.append(W(t, c - 3 * len(t), 640, h=11))
    ws += [W("Invoice", 20, 430, h=11), W("Details:", 70, 430, h=11)]
    return ws


def test_gst_table_with_stacked_headers_and_a_total_row():
    r = apply_rules(gst_words(), [], 1000, 1414)
    it = r["items"]
    assert len(it) == 2
    assert [(i["id"], i["description"], i["hsn_sac_code"], i["qty"], i["unit_price"], i["taxable_value"], i["amount"]) for i in it] == [
        ("1", "Product A", "1234", "10", "500", "5000", "5900"), ("2", "Product B", "5678", "5", "1000", "5000", "5900")]
    assert (it[0]["cgst"], it[0]["sgst_pct"]) == ("9%", "9%")       # "CGST (96)": the OCR lost the % of the header
    assert r["items_check"]["footer"] == {"taxable_value": "10000", "ccgst_amount": "900", "sgst_amount": "900", "amount": "11800"}
    f = {x["label"]: x["text"] for x in r["fields"]}
    assert (f["SUB_TOTAL"], f["TAX"], f["TOTAL"]) == ("10000.00", "1800.00", "11800.00")      # tax = CGST + SGST
    assert not any(i["corrected"] for i in it)          # amount != qty x price here (taxes): never "repaired"
