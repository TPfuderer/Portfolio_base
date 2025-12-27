# quick_test.py
from pathlib import Path
from portfolio_base.tegut_ocr.yolo_detect import detect_products

pdf = Path("data/input/pdf_new/tegut_test.pdf")
crops = detect_products(pdf)

print(f"{len(crops)} Produkte erkannt")
