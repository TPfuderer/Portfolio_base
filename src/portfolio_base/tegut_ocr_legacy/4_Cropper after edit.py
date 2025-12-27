# ===============================
# 🖼️ Produktbilder aus YOLO-Boxen ausschneiden (TEGUT OCR Version)
# ===============================

import cv2
import ast
import pandas as pd
from pathlib import Path

# === 📅 Kalenderwoche definieren ===
KW = 46   # << HIER ÄNDERN

# === 📂 Ordnerstruktur ===
BASE = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")
base_dir = BASE / f"KW{KW:02d}"

pages_dir = base_dir / "pages"
csv_path  = base_dir / "matched" / f"KW{KW}_matches_grouped.csv"
crops_dir = base_dir / "crops"

crops_dir.mkdir(parents=True, exist_ok=True)

print(f"📄 Lade CSV: {csv_path}")
df = pd.read_csv(csv_path)

# ===============================
# Cropping
# ===============================

crop_paths = []

for idx, row in df.iterrows():
    try:
        # --- PNG-Seitenbild auswählen ---
        page_no = int(row["page"])
        img_path = pages_dir / f"page_{page_no:02d}.png"

        if not img_path.exists():
            print(f"⚠️ Seite nicht gefunden: {img_path}")
            crop_paths.append(None)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Fehler beim Laden des Bildes: {img_path}")
            crop_paths.append(None)
            continue

        # --- YOLO-Box in Pixel ---
        box = ast.literal_eval(row["yolo_box"])
        x1, y1, x2, y2 = map(int, box)

        # --- Crop ausschneiden ---
        crop = img[y1:y2, x1:x2]

        # --- Dateiname ---
        crop_name = f"KW{KW}_page{page_no:02d}_id{int(row['product_id']):03d}.jpg"
        crop_path = crops_dir / crop_name

        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))

        print(f"✅ Crop gespeichert: {crop_name}")

    except Exception as e:
        print(f"❌ Fehler in Zeile {idx}: {e}")
        crop_paths.append(None)

# ===============================
# CSV erweitern und speichern
# ===============================

df["crop_path"] = crop_paths

df["crop_link"] = df["crop_path"].apply(
    lambda p: f'<a href="file:///{p}" target="_blank">🖼️ öffnen</a>' if isinstance(p, str) else None
)

out_csv = base_dir / "matched" / f"KW{KW}_matches_grouped_crops.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\n🎯 Cropping abgeschlossen!")
print(f"💾 Neue CSV: {out_csv}")
print(f"📁 Crops gespeichert unter: {crops_dir}")
