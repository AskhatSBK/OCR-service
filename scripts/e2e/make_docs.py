"""Generate test documents with known ground truth (runs inside the OCR service image)."""
import io
import json
import random
import sys

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = sys.argv[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INVOICE = [
    ("b", "INVOICE No. INV-2026-0417"),
    ("r", "Date: 03.11.2026    Due date: 03.12.2026"),
    ("r", "Supplier: Nordwind Logistics GmbH, Hafenstrasse 12, 20457 Hamburg"),
    ("r", "VAT ID: DE811907980"),
    ("r", "IBAN: DE89 3704 0044 0532 0130 00    SWIFT: COBADEFFXXX"),
    ("b", "Счёт-фактура № 417 от 03 ноября 2026 г."),
    ("r", "Покупатель: ТОО «Казахстан Трейд», г. Алматы, пр. Абая 150"),
    ("r", "Назначение платежа: оплата за транспортные услуги"),
    ("b", "Τιμολόγιο Παροχής Υπηρεσιών"),
    ("r", "Πελάτης: Αιγαίο Ναυτιλιακή Α.Ε., Πειραιάς"),
    ("r", "Item            Qty    Unit price    Amount"),
    ("r", "Freight HAM-ALA   2      1,250.00    2,500.00"),
    ("r", "Customs handling  1        340.50      340.50"),
    ("r", "Insurance         1        115.00      115.00"),
    ("b", "Total due: 2,955.50 EUR"),
]
STATEMENT = [
    ("b", "Bank Statement 01.10.2026 - 31.10.2026"),
    ("r", "Account: KZ86 125K ZT50 0410 0100"),
    ("r", "Date        Description                 Debit      Credit"),
    ("r", "02.10.2026  Card payment AMAZON EU      129.99"),
    ("r", "05.10.2026  Salary October                          4,800.00"),
    ("r", "11.10.2026  Transfer to Nordwind GmbH   2,955.50"),
    ("r", "19.10.2026  Комиссия банка              12.00"),
    ("b", "Closing balance: 18,402.51"),
]
KEY_FIELDS = {
    "invoice": ["INV-2026-0417", "DE89 3704 0044 0532 0130 00", "COBADEFFXXX", "DE811907980",
                "2,955.50", "Казахстан Трейд", "транспортные услуги", "Αιγαίο Ναυτιλιακή", "Πειραιάς"],
    "statement": ["KZ86 125K ZT50 0410 0100", "4,800.00", "2,955.50", "Комиссия банка", "18,402.51"],
}


def render(lines, scale=2.0):
    w, h = int(1240 * scale / 2), int((140 + 62 * len(lines)) * scale / 2)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y = 60 * scale / 2
    for style, text in lines:
        f = ImageFont.truetype(FONT_B if style == "b" else FONT, int(26 * scale / 2))
        d.text((70 * scale / 2, y), text, font=f, fill="black")
        y += 62 * scale / 2
    return img


def scanify(img, seed):
    rnd = random.Random(seed)
    img = img.convert("L").rotate(rnd.uniform(-1.5, 1.5), expand=True, fillcolor=245)
    px = img.load()
    for _ in range(img.width * img.height // 150):
        x, y = rnd.randrange(img.width), rnd.randrange(img.height)
        px[x, y] = rnd.choice((90, 160, 255))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=55)  # JPEG artefacts like a cheap scanner
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def digital_pdf(lines, path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for style, text in lines:
        page.insert_text((40, y), text, fontsize=10.5, fontfile=FONT_B if style == "b" else FONT,
                         fontname="DJB" if style == "b" else "DJ")
        y += 24
    doc.save(path)


def image_pdf(images, path):
    doc = pymupdf.open()
    for im in images:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        page = doc.new_page(width=595, height=595 * im.height / im.width)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)


truth = {}
digital_pdf(INVOICE, f"{OUT}/invoice_digital.pdf")
truth["invoice_digital.pdf"] = ("invoice", [t for _, t in INVOICE])

inv_scan, st_scan = scanify(render(INVOICE), 1), scanify(render(STATEMENT), 2)
image_pdf([inv_scan, st_scan], f"{OUT}/scan_2pages.pdf")
truth["scan_2pages.pdf"] = ("invoice+statement", [t for _, t in INVOICE] + [t for _, t in STATEMENT])

photo = scanify(render(INVOICE, scale=1.6), 3)
photo.save(f"{OUT}/invoice_photo.webp", "WEBP", quality=60)
truth["invoice_photo.webp"] = ("invoice", [t for _, t in INVOICE])

st_scan.save(f"{OUT}/statement_scan.png")
truth["statement_scan.png"] = ("statement", [t for _, t in STATEMENT])

json.dump({"truth": truth, "key_fields": KEY_FIELDS}, open(f"{OUT}/truth.json", "w"), ensure_ascii=False, indent=1)
print("generated", sorted(truth))
