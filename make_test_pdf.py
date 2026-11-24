from PIL import Image, ImageDraw, ImageFont

font_lg = ImageFont.load_default(size=28)
font_md = ImageFont.load_default(size=20)

# Page 1: Header, metadata, and initial line items (with one deliberate math error)
p1 = Image.new("RGB", (1240, 1754), "white")
d1 = ImageDraw.Draw(p1)
d1.text((80, 70), "ACME INDUSTRIAL SUPPLY CO.", fill="black", font=font_lg)
d1.text((80, 115), "Invoice Date: 2026-11-15   |   Account: #8841", fill="black", font=font_md)
d1.line((80, 165, 1160, 165), fill="black", width=3)

headers = ["Code", "Description", "Qty", "Unit Price", "Total"]
xs = [80, 230, 680, 810, 990]
for x, h in zip(xs, headers):
    d1.text((x, 185), h, fill="black", font=font_md)
d1.line((80, 220, 1160, 220), fill="black", width=2)

d1.text((80, 240), "[Hardware]", fill="black", font=font_md)
page1_rows = [
    ("HW-101", "Heavy Duty Steel Bracket", "4", "$25.00", "$100.00"),
    ("HW-204", "Titanium Hex Bolt Pack", "3", "$50.00", "$160.00"),  # Deliberate math error!
    ("HW-309", "Industrial Torque Wrench", "1", "$145.50", "$145.50"),
]
for idx, row in enumerate(page1_rows):
    y = 285 + idx * 48
    for x, val in zip(xs, row):
        d1.text((x, y), val, fill="black", font=font_md)

# Page 2: Headerless continuation table (tests cross-page state carry-over)
p2 = Image.new("RGB", (1240, 1754), "white")
d2 = ImageDraw.Draw(p2)
d2.text((80, 70), "Page 2 - Continued", fill="black", font=font_md)
page2_rows = [
    ("HW-412", "Neoprene Gasket Roll", "2", "$32.00", "$64.00"),
    ("HW-518", "Safety Lockout Kit", "5", "$18.00", "$90.00"),
]
for idx, row in enumerate(page2_rows):
    y = 140 + idx * 48
    for x, val in zip(xs, row):
        d2.text((x, y), val, fill="black", font=font_md)

p1.save("sample_document.pdf", "PDF", resolution=150.0, save_all=True, append_images=[p2])
print("Created sample_document.pdf")