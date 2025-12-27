import json
import pandas as pd
from pathlib import Path

# ============================
# 🔧 Pfade
# ============================
gpt_dir = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2\KW42_V6\matched\GPT_batches")

# ============================
# 📄 Alle JSON-Dateien laden
# ============================
all_records = []

for json_file in gpt_dir.glob("page_*.json"):
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
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

# Optional: alle Keys anzeigen
print("\n📋 Spalten im GPT-DataFrame:")
print(list(gpt_df.columns))

# ============================
# 📊 Vorschau
# ============================
print("\n🧾 GPT-Output-Vorschau:")
print(gpt_df.head(10).to_string(index=False))


# ============================
# ➕ CSV-Spalten (rechts) anhängen
# ============================
from itertools import islice

csv_path = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\Run2\KW42_V6\matched\KW42_matches_grouped_crops_filtered.csv")

# Nur relevante Spalten laden
df_crop = pd.read_csv(csv_path, usecols=["page", "product_id", "crop_path"], dtype=str)

print(f"\n📄 crop_path-Datei geladen: {len(df_crop)} Zeilen")

# Falls CSV länger ist als GPT-Ergebnis → kürzen
min_len = min(len(gpt_df), len(df_crop))
gpt_df_trimmed = gpt_df.iloc[:min_len].reset_index(drop=True)
df_crop_trimmed = df_crop.iloc[:min_len].reset_index(drop=True)

# Rechts anhängen (nebeneinander)
combined_df = pd.concat([gpt_df_trimmed, df_crop_trimmed], axis=1)

print(f"✅ Kombiniert (spaltenweise): {len(combined_df)} Zeilen, {len(combined_df.columns)} Spalten")

# Spaltenübersicht
print("\n📋 Kombinierte Spalten:")
print(list(combined_df.columns))

# Vorschau
print("\n🧾 Vorschau kombinierter DataFrame:")
print(combined_df.head(15).to_string(index=False))

# Optional: Speichern
out_path = csv_path.with_name(csv_path.stem + "_GPT_columns.csv")
combined_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\n💾 Datei gespeichert unter:\n{out_path}")
