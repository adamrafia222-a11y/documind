"""A ruled invoice like the ones that gave trouble: dark header band with white text, 7 columns, wrapped description cells, totals box. Pillow only."""
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


ROWS = [("1", ["Software Development", "(Knowvault Platform)"], "1", "25 000,00", "20%", "5 000,00", "30 000,00"),
        ("2", ["Technical Support", "(1 Month)"], "1", "5 000,00", "20%", "1 000,00", "6 000,00"),
        ("3", ["Consulting Services"], "10", "500,00", "20%", "1 000,00", "6 000,00")]
HEAD = ["#", "Description", "Quantity", "Unit Price (HT)", "VAT Rate", "VAT Amount", "Total (TTC)"]
XS = [40, 100, 350, 460, 590, 700, 855, 1010]


def make(width=1074, height=857, page_border=False) -> Image.Image:
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    navy, line = (11, 45, 100), (40, 55, 90)
    d.text((660, 75), "Invoice No.", font=font(26), fill=(20, 20, 30)); d.text((835, 75), ": INV-2025-0876", font=font(26), fill=(20, 20, 30))
    d.text((660, 118), "Issue Date", font=font(26), fill=(20, 20, 30)); d.text((835, 118), ": 18/09/2025", font=font(26), fill=(20, 20, 30))
    d.text((66, 75), "Bill To :", font=font(26, True), fill=navy); d.text((66, 112), "Knowvault Corp.", font=font(26, True), fill=(20, 20, 30))
    d.rectangle([XS[0], 295, XS[-1], 365], fill=navy)
    for i, h in enumerate(HEAD):
        parts = h.split(" (") if h.startswith("Unit") else [h]
        for k, part in enumerate(parts):
            t = part if k == 0 else "(" + part
            tw = d.textlength(t, font=font(20, True))
            d.text(((XS[i] + XS[i + 1]) / 2 - tw / 2, 316 + 22 * k - (11 if len(parts) > 1 else 0) + (0 if len(parts) > 1 else 0)), t, font=font(20, True), fill="white")
    ys = [365, 440, 510, 570]
    for j, (n, desc, q, up, vr, va, tt) in enumerate(ROWS):
        y0, y1 = ys[j], ys[j + 1]
        cells = [n, None, q, up, vr, va, tt]
        for i, c in enumerate(cells):
            if i == 1:
                for k, ln in enumerate(desc):
                    d.text((XS[1] + 16, (y0 + y1) / 2 - 12 * len(desc) + 24 * k), ln, font=font(21), fill=(20, 20, 30))
            elif i in (3, 5, 6):
                tw = d.textlength(c, font=font(21)); d.text((XS[i + 1] - 18 - tw, (y0 + y1) / 2 - 12), c, font=font(21), fill=(20, 20, 30))
            else:
                tw = d.textlength(c, font=font(21)); d.text(((XS[i] + XS[i + 1]) / 2 - tw / 2, (y0 + y1) / 2 - 12), c, font=font(21), fill=(20, 20, 30))
    for y in [365] + ys[1:]:
        d.line([XS[0], y, XS[-1], y], fill=line, width=2)
    for x in XS:
        d.line([x, 295, x, 570], fill=line if x in (XS[0], XS[-1]) else line, width=2)
    d.line([XS[0], 295, XS[-1], 295], fill=line, width=2)
    d.rectangle([575, 593, 1010, 640], outline=line, width=2); d.text((592, 604), "Subtotal (HT)", font=font(22, True), fill=(20, 20, 30)); d.text((900, 604), "30 000,00", font=font(22, True), fill=(20, 20, 30))
    d.rectangle([575, 640, 1010, 683], outline=line, width=2); d.text((592, 650), "VAT (20%)", font=font(22, True), fill=(20, 20, 30)); d.text((905, 650), "6 000,00", font=font(22, True), fill=(20, 20, 30))
    d.rectangle([575, 683, 1010, 730], fill=navy); d.text((592, 695), "Total (TTC)", font=font(22, True), fill="white"); d.text((825, 695), "36 000,00 MAD", font=font(22, True), fill="white")
    if page_border:                                                                   # a screenshot / scan with a dark frame: the longest lines of the picture are NOT the table
        d.rectangle([0, 0, width - 1, height - 1], outline=(30, 30, 40), width=12)
    return im
