# ===============================
# 🚀 Flyer Text-Matching Pipeline – KW45_V5 (Seiten 1–2)
# ===============================

import fitz, json
from pathlib import Path

# === 📂 Pfade anpassen ===
pdf_path = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Flyer_pdf\tegut... Flugblatt KW 45_2025 Unterfranken, Mittel- & Oberfranken.pdf")
label_dir = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2\KW42_V4\labels")
out_dir   = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2\KW45_V5")
(out_dir / "matched_json").mkdir(parents=True, exist_ok=True)
(out_dir / "textlayers").mkdir(exist_ok=True)

# === ⚙️ Hilfsfunktionen ===
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

# === 🧩 1️⃣ Textlayer-Blöcke extrahieren (Seiten 1 & 2) ===
print("📄 Extrahiere Textlayer für Seiten 1 & 2 ...")
doc = fitz.open(pdf_path)
text_layers = {}

for i in [1, 2]:
    page = doc[i-1]
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[:5]
        text = text.strip()
        if not text:
            continue
        blocks.append({"x1": x0, "y1": y0, "x2": x1, "y2": y1, "text": text})
    text_layers[i] = blocks
    (out_dir / "textlayers" / f"page_{i:02d}_text.json").write_text(json.dumps(blocks, indent=2))
    print(f"✅ Seite {i}: {len(blocks)} Textboxen gespeichert.")

# === 🧠 2️⃣ Produktboxen aus YOLO-Labels laden ===
def load_yolo_labels(label_path):
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls, conf, x1, y1, x2, y2 = parts
        boxes.append({
            "cls": int(float(cls)),
            "conf": float(conf),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        })
    return boxes

label_files = {
    1: label_dir / "page_01.txt",
    2: label_dir / "page_02.txt"
}

# === 📦 3️⃣ Matching Produkt-Box ↔ Text-Box (IoU ≥ 0.2) ===
print("🔍 Starte Matching zwischen Produkt- und Text-Boxen ...")
for page_no, label_path in label_files.items():
    product_boxes = load_yolo_labels(label_path)
    text_boxes = text_layers[page_no]
    matches = []

    for prod in product_boxes:
        prod_box = [prod["x1"], prod["y1"], prod["x2"], prod["y2"]]
        overlapping = []
        for txt in text_boxes:
            txt_box = [txt["x1"], txt["y1"], txt["x2"], txt["y2"]]
            if iou(prod_box, txt_box) >= 0.2:
                overlapping.append(txt["text"])
        matched_text = " ".join(overlapping).strip()
        matches.append({
            "page": page_no,
            "product_box": prod_box,
            "conf": prod["conf"],
            "cls": prod["cls"],
            "matched_text": matched_text
        })

    out_path = out_dir / "matched_json" / f"page_{page_no:02d}_matches.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    print(f"✅ Seite {page_no}: {len(matches)} Produkt-Text-Matches gespeichert → {out_path}")

print("🎉 Matching abgeschlossen – Ergebnisse liegen in:", out_dir / "matched_json")
