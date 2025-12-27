import json
import pandas as pd
from pathlib import Path

# ============================
# 📅 Kalenderwoche definieren
# ============================
KW = 46  # << hier einstellen

# ============================
# 🔧 Pfade (NEU für TEGUT OCR)
# ============================
BASE = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")
gpt_dir = BASE / f"KW{KW:02d}" / "matched" / "GPT_batches"
csv_path = BASE / f"KW{KW:02d}" / "matched" / f"KW{KW}_matches_grouped_crops_filtered.csv"

print(f"📁 GPT JSON Ordner: {gpt_dir}")
print(f"📄 CSV crop_path Datei: {csv_path}")

# ============================
# 📄 Alle JSON-Dateien laden
# ============================
all_records = []

for json_file in gpt_dir.glob("page_*.json"):
    try:
        raw = json_file.read_text(encoding="utf-8")
        data = json.loads(raw)

        # Falls GPT nur ein Dict statt Liste ausgegeben hat
        if isinstance(data, dict):
            data = [data]

        all_records.extend(data)
        print(f"✅ {json_file.name}: {len(data)} Einträge geladen")

    except Exception as e:
        print(f"⚠️ Fehler in {json_file.name}: {e}")

# ============================
# 🧠 In DataFrame umwandeln
# ============================
gpt_df = pd.DataFrame(all_records)

print("\n📋 Spalten im GPT-DataFrame:")
print(list(gpt_df.columns))

print("\n🧾 GPT-Output-Vorschau:")
print(gpt_df.head(10).to_string(index=False))


# ============================
# ➕ crop_path CSV anhängen
# ============================
df_crop = pd.read_csv(csv_path, usecols=["page", "product_id", "crop_path"], dtype=str)
print(f"\n📄 crop_path-Datei geladen: {len(df_crop)} Zeilen")

# 🔄 Reihen angleichen
min_len = min(len(gpt_df), len(df_crop))
gpt_df_trimmed = gpt_df.iloc[:min_len].reset_index(drop=True)
df_crop_trimmed = df_crop.iloc[:min_len].reset_index(drop=True)

# ➕ Nebeneinander kombinieren
combined_df = pd.concat([gpt_df_trimmed, df_crop_trimmed], axis=1)

print(f"✅ Kombiniert (spaltenweise): {len(combined_df)} Zeilen, {len(combined_df.columns)} Spalten")

print("\n📋 Kombinierte Spalten:")
print(list(combined_df.columns))

print("\n🧾 Vorschau kombinierter DataFrame:")
print(combined_df.head(15).to_string(index=False))


# ============================
# 💾 Speichern
# ============================
out_path = csv_path.with_name(csv_path.stem + "_GPT_columns.csv")
combined_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\n💾 Datei gespeichert unter:\n{out_path}")
