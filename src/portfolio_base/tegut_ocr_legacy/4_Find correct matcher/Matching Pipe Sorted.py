# ===============================
# 🧩 Produkt-Text-Matching mit eindeutiger Zuordnung
# ===============================

import fitz
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# === 📅 Kalenderwoche definieren ===
KW = 42  # anpassen nach Bedarf

# === 📂 Pfade automatisch ableiten ===
base_dir = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2")
pdf_path = Path(fr"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Flyer_pdf\tegut... Flugblatt KW {KW}_2025 Unterfranken, Mittel- & Oberfranken.pdf")
pages_dir = base_dir / f"KW{KW}_V4" / "pages"
label_dir = base_dir / f"KW{KW}_V4" / "labels"
out_dir = base_dir / f"KW{KW}_V5" / "matched"
out_vis = out_dir / "visual"

out_dir.mkdir(parents=True, exist_ok=True)
out_vis.mkdir(parents=True, exist_ok=True)

print(f"📄 PDF:       {pdf_path}")
print(f"🖼️ Pages:     {pages_dir}")
print(f"🔵 Labels:    {label_dir}")
print(f"💾 Output:    {out_dir}\n")

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
    return boxes

def rect_intersection(a, b):
    """Berechne den Anteil der Textbox, der in der Produktbox liegt"""
    x_left = max(a["x1"], b["x1"])
    y_top = max(a["y1"], b["y1"])
    x_right = min(a["x2"], b["x2"])
    y_bottom = min(a["y2"], b["y2"])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    text_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter_area / text_area


# === PDF öffnen ===
doc = fitz.open(pdf_path)
print(f"✅ PDF geladen ({len(doc)} Seiten)\n")

# === 🔄 Verarbeitung aller Seiten ===
page_files = sorted(pages_dir.glob("page_*.png"))
all_matches = []

for page_img_path in page_files:
    page_no = int(page_img_path.stem.split("_")[-1])
    print(f"🟩 === Seite {page_no} ===")

    # --- Lade Bild ---
    img = cv2.imread(str(page_img_path))
    if img is None:
        print(f"❌ Konnte Bild nicht laden: {page_img_path}")
        continue
    new_h, new_w = img.shape[:2]

    # --- Textlayer aus PDF ---
    page = doc[page_no - 1]
    text_blocks = page.get_text("blocks")
    page_w_pt, page_h_pt = page.rect.width, page.rect.height
    mat = fitz.Matrix(new_w / page_w_pt, new_h / page_h_pt)

    text_boxes = []
    for b in text_blocks:
        try:
            x0, y0, x1, y1, text = b[:5]
        except Exception:
            continue
        if not str(text).strip():
            continue
        rect = fitz.Rect(x0, y0, x1, y1) * mat
        text_boxes.append({
            "x1": rect.x0, "y1": rect.y0, "x2": rect.x1, "y2": rect.y1, "text": text.strip()
        })
    print(f"   ✅ {len(text_boxes)} Textboxen geladen")

    # --- YOLO-Labels laden ---
    yolo_path = label_dir / f"page_{page_no:02d}.txt"
    if not yolo_path.exists():
        print(f"⚠️ Keine Labeldatei für Seite {page_no}")
        continue
    yolo_boxes = load_yolo_labels(yolo_path)
    print(f"   ✅ {len(yolo_boxes)} YOLO-Boxen geladen")

    # --- Overlap-Matching ---
    matches = []
    assigned_texts = set()
    for i, yb in enumerate(yolo_boxes):
        yb_rect = yb.copy()
        matched_texts = []
        for j, tb in enumerate(text_boxes):
            overlap = rect_intersection(yb_rect, tb)
            if overlap >= 0.95:  # mindestens 95% der Textbox liegt in der Produktbox
                matched_texts.append((j, overlap))
        # Sortiere Textmatches nach Overlap absteigend
        matched_texts.sort(key=lambda x: x[1], reverse=True)
        for j, overlap in matched_texts:
            if j not in assigned_texts:
                assigned_texts.add(j)
                matches.append({
                    "page": page_no,
                    "product_id": i + 1,
                    "conf": yb["conf"],
                    "yolo_box": [yb["x1"], yb["y1"], yb["x2"], yb["y2"]],
                    "text": text_boxes[j]["text"],
                    "text_box": [text_boxes[j]["x1"], text_boxes[j]["y1"], text_boxes[j]["x2"], text_boxes[j]["y2"]],
                    "overlap": round(overlap, 3)
                })
        # falls keine Textbox matched, trotzdem Produkt speichern
        if not matched_texts:
            matches.append({
                "page": page_no,
                "product_id": i + 1,
                "conf": yb["conf"],
                "yolo_box": [yb["x1"], yb["y1"], yb["x2"], yb["y2"]],
                "text": None,
                "text_box": None,
                "overlap": 0.0
            })

    all_matches.extend(matches)
    print(f"   ✅ {len(matches)} Produkt-Text-Kombinationen")

    # --- Debug-Bild ---
    img_vis = img.copy()
    for m in matches:
        # Blaue Box = YOLO-Produkt
        x1, y1, x2, y2 = map(int, m["yolo_box"])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

        if m["text_box"]:
            tx1, ty1, tx2, ty2 = map(int, m["text_box"])
            cv2.rectangle(img_vis, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.putText(img_vis, f"{m['overlap']:.2f}", (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_img = out_vis / f"page_{page_no:02d}_matched.jpg"
    cv2.imwrite(str(out_img), img_vis)
    print(f"   🖼️ Debug-Image gespeichert: {out_img.name}\n")

# === 💾 CSV + JSON-Ausgabe ===
csv_path = out_dir / f"KW{KW}_matches.csv"
json_path = out_dir / f"KW{KW}_matches.json"
pd.DataFrame(all_matches).to_csv(csv_path, index=False, encoding="utf-8-sig")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_matches, f, indent=2, ensure_ascii=False)

print(f"✅ Alle Seiten verarbeitet.")
print(f"💾 CSV gespeichert: {csv_path}")
print(f"💾 JSON gespeichert: {json_path}")
print(f"🎉 Fertig: Eindeutiges Produkt-Text-Matching abgeschlossen!")

# ===============================
# 🧩 Kombiniere alle Produkt-Texte (gleiche Product IDs)
# ===============================

import pandas as pd
from pathlib import Path
import json

KW = 42
base_dir = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2") / f"KW{KW}_V5" / "matched"
csv_path = base_dir / f"KW{KW}_matches.csv"
out_csv  = base_dir / f"KW{KW}_matches_grouped.csv"
out_json = base_dir / f"KW{KW}_matches_grouped.json"

print(f"📂 Lade Matches aus: {csv_path}")
df = pd.read_csv(csv_path)

# Nur Produkte mit Text behalten (optional)
df_text = df.copy()

# Gruppierung nach Seite und Produkt-ID
grouped = (
    df_text.groupby(["page", "product_id"], as_index=False)
    .agg({
        "conf": "first",
        "yolo_box": "first",
        "text": lambda x: "\n".join([t for t in x.dropna().astype(str) if t.strip()]),
        "overlap": "max"
    })
)

grouped.rename(columns={"text": "combined_text"}, inplace=True)

# Reihenfolge anpassen
grouped = grouped[["page", "product_id", "conf", "yolo_box", "overlap", "combined_text"]]

# Speichern
grouped.to_csv(out_csv, index=False, encoding="utf-8-sig")
grouped.to_json(out_json, orient="records", indent=2, force_ascii=False)

print(f"✅ Gruppierte Matches gespeichert:")
print(f"   - CSV:  {out_csv}")
print(f"   - JSON: {out_json}")
print(f"📊 {len(grouped)} eindeutige Produktgruppen erstellt.")
