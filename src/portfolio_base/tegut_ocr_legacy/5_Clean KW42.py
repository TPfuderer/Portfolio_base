# ===============================
# 🧹 Filtering + GPT Prompt Generator (TEGUT OCR Version)
# ===============================

import pandas as pd
from pathlib import Path

# === Kalenderwoche einstellen ===
KW = 46

# === Basisordner ===
BASE = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")

# === Input (kommt direkt aus deinem Cropping-Skript) ===
csv_path = BASE / f"KW{KW:02d}" / "matched" / f"KW{KW}_matches_grouped_crops.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"❌ Cropped CSV nicht gefunden:\n{csv_path}")

print(f"📂 Lade Cropping-Datei:\n{csv_path}")

# === Output-Pfade ===
output_filtered = csv_path.with_name(csv_path.stem + "_filtered.csv")
output_prompt   = csv_path.with_name(csv_path.stem + "_prompt_input.txt")

# === CSV einlesen (alles als String) ===
df = pd.read_csv(csv_path, dtype=str)

# === Spalten behalten ===
cols_keep = ["page", "product_id", "combined_text", "crop_path"]
df = df[cols_keep]

# === Entferne NULL/leer ===
df = df[df["combined_text"].notna()]
df = df[df["combined_text"].str.strip().ne("")]
df = df[df["combined_text"].str.upper() != "NULL"]

# === Gefilterte CSV speichern ===
df.to_csv(output_filtered, index=False, encoding="utf-8-sig")

print(f"✅ Gefilterte Datei gespeichert:\n{output_filtered}")
print(f"📊 Verbleibende Zeilen: {len(df)}")

# ===============================
# 📝 GPT Prompt-Datei erzeugen
# ===============================

lines = []

for _, row in df.iterrows():
    page = row["page"].strip()
    pid = row["product_id"].strip()
    text = row["combined_text"].strip()

    block = f"§§ page: {page} | product_id: {pid}\n{text}\n@@\n"
    lines.append(block)

with open(output_prompt, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"📝 Prompt-Input Datei gespeichert:\n{output_prompt}")
print(f"📦 Gesamtprodukte exportiert: {len(lines)}")
