# ===============================
# 🧮 Textlayer + YOLO-Koordinaten an Seiten anpassen (KW-dynamisch)
# ===============================

import fitz
import json
import cv2
import numpy as np
from pathlib import Path

# === 📅 Kalenderwoche definieren ===
KW = 42   # einfach hier anpassen (z. B. 38, 45, …)

# === 📂 Pfade automatisch ableiten ===
base_dir = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2")
pdf_path  = Path(fr"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Flyer_pdf\tegut... Flugblatt KW {KW}_2025 Unterfranken, Mittel- & Oberfranken.pdf")
pages_dir = base_dir / f"KW{KW}_V4" / "pages"
label_dir = base_dir / f"KW{KW}_V4" / "labels"
out_dir   = base_dir / f"KW{KW}_V5" / "aligned"
out_vis   = out_dir / "visual"

out_dir.mkdir(parents=True, exist_ok=True)
out_vis.mkdir(parents=True, exist_ok=True)

print(f"📄 PDF:       {pdf_path}")
print(f"🖼️ Pages:     {pages_dir}")
print(f"🔵 Labels:    {label_dir}")
print(f"💾 Output:    {out_dir}")

# === 🧠 Hilfsfunktionen ===
def load_yolo_labels(path):
    boxes = []
    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        print(f"❌ Fehler beim Lesen {path.name}: {e}")
        return boxes
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            cls, conf, x1, y1, x2, y2 = parts
            boxes.append(dict(
                cls=int(float(cls)), conf=float(conf),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
            ))
    print(f"   ✅ {len(boxes)} Boxen aus {path.name}")
    return boxes

# === PDF öffnen ===
if not pdf_path.exists():
    print(f"❌ PDF nicht gefunden: {pdf_path}")
    raise SystemExit
doc = fitz.open(pdf_path)
print(f"✅ PDF geladen ({len(doc)} Seiten)\n")

# === 🔄 Verarbeitung (alle vorhandenen PNGs automatisch) ===
page_files = sorted(pages_dir.glob("page_*.png"))
for page_img_path in page_files:
    page_no = int(page_img_path.stem.split("_")[-1])
    print(f"🟩 === Seite {page_no} ===")

    if not (label_dir / f"page_{page_no:02d}.txt").exists():
        print(f"⚠️ Keine Labeldatei für Seite {page_no}")
        continue

    # --- Seitenbild laden ---
    img = cv2.imread(str(page_img_path))
    if img is None:
        print(f"❌ Konnte Bild nicht laden: {page_img_path}")
        continue
    new_h, new_w = img.shape[:2]
    print(f"   ✅ Seitenbild: {new_w}×{new_h}")

    # --- Textlayer aus PDF ---
    page = doc[page_no - 1]
    text_blocks = page.get_text("blocks")
    print(f"   ✅ Textlayer: {len(text_blocks)} Blöcke")

    # --- PDF-Seitenmaße & Matrix von der Pixmap-Erstellung ---
    page_w_pt, page_h_pt = page.rect.width, page.rect.height

    # 🟢 Nutze dieselbe Matrix wie beim Rendern der PDF-Seite
    zoom_x = new_w / page_w_pt
    zoom_y = new_h / page_h_pt
    mat = fitz.Matrix(zoom_x, zoom_y)

    # --- Textlayer direkt mit Matrix transformieren (keine Invertierung nötig) ---
    text_aligned = []
    for b in text_blocks:
        try:
            x0, y0, x1, y1, text = b[:5]
        except Exception:
            continue
        if not str(text).strip():
            continue
        rect = fitz.Rect(x0, y0, x1, y1) * mat
        text_aligned.append({
            "x1": rect.x0,
            "y1": rect.y0,
            "x2": rect.x1,
            "y2": rect.y1,
            "text": text.strip()
        })
    print(f"   ✅ {len(text_aligned)} Textboxen korrekt transformiert (Matrix-Methode)")

    # --- YOLO Labels ---
    yolo_path = label_dir / f"page_{page_no:02d}.txt"
    yolo_boxes = load_yolo_labels(yolo_path)

    # --- Speichern ---
    json_path = out_dir / f"page_{page_no:02d}_text_aligned.json"
    label_out = out_dir / f"page_{page_no:02d}_labels_aligned.txt"
    json_path.write_text(json.dumps(text_aligned, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(label_out, "w", encoding="utf-8") as f:
        for b in yolo_boxes:
            f.write(f"{b['cls']} {b['conf']} {b['x1']} {b['y1']} {b['x2']} {b['y2']}\n")
    print(f"   💾 Gespeichert in: {out_dir}")

    # --- Visualisierung ---
    img_vis = img.copy()

    # 🔵 YOLO-Boxen (Produkte) – dicke blaue Linien
    for b in yolo_boxes:
        x1, y1, x2, y2 = map(int, [b["x1"], b["y1"], b["x2"], b["y2"]])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(img_vis, f"{b['conf']:.2f}", (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    # 🔴 Textlayer-Boxen – dünnere rote Linien, aber gut sichtbar
    for t in text_aligned:
        x1, y1, x2, y2 = map(int, [t["x1"], t["y1"], t["x2"], t["y2"]])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        txt_short = t["text"][:20].replace("\n", " ")
        cv2.putText(img_vis, txt_short, (x1 + 5, min(y2, y1 + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Speichern
    out_img = out_vis / f"page_{page_no:02d}_compare.jpg"
    cv2.imwrite(str(out_img), img_vis)
    print(f"   🖼️ Visual gespeichert: {out_img.name}\n")

print(f"🎉 Fertig: KW{KW}_V5 → {out_dir}")
