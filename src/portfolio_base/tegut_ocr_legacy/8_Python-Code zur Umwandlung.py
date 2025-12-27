import pandas as pd
from pathlib import Path
import shutil
import numpy as np

# =======================================
# 📅 Kalenderwoche einstellen
# =======================================
KW = 46

# =======================================
# 📂 Pfade einrichten
# =======================================
BASE = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")
MATCHED = BASE / f"KW{KW:02d}" / "matched"

INPUT = MATCHED / f"KW{KW}_matches_grouped_crops_filtered_GPT_columns.csv"

# Finale Ausgabe
STREAMLIT_READY = BASE / f"KW{KW:02d}" / "streamlit_ready"
IMG_OUT = STREAMLIT_READY / "images_tegut"

STREAMLIT_READY.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)

# Datumsbereich (bei Bedarf anpassen)
GUELTIG_VON = "2025-11-10"
GUELTIG_BIS = "2025-11-16"
GUELTIG_RAW = "10.11.2025 bis 16.11.2025"

OUTPUT = STREAMLIT_READY / f"tegut_KW{KW}_{GUELTIG_VON}_bis_{GUELTIG_BIS}.csv"

print("📄 Input CSV:", INPUT)
print("📁 Output CSV:", OUTPUT)
print("🖼️ Ziel-Image-Ordner:", IMG_OUT)

# =======================================
# 📄 CSV laden (als String, damit keine NaN entstehen)
# =======================================
df = pd.read_csv(INPUT, dtype=str)

# =======================================
# 🔧 ALLE Arten von NaN / None / 'nan' entfernen
# =======================================
df = df.replace({
    np.nan: "",
    "nan": "",
    "NaN": "",
    "None": "",
    "NONE": "",
    "null": "",
    "NULL": "",
})

# =======================================
# Spalten normalisieren
# =======================================
df = df.rename(columns={
    "Vorheriger_Preis": "Vorheriger Preis",
    "crop_path": "Bildpfad"
})

# =======================================
# Gültigkeitsinfos
# =======================================
df["Nur_online"] = "Nein"
df["Gueltig_raw"] = GUELTIG_RAW
df["Gueltig_von"] = GUELTIG_VON
df["Gueltig_bis"] = GUELTIG_BIS

# =======================================
# 🖼️ Bilder kopieren + Pfade reparieren
# =======================================
def process_img(path):
    if not isinstance(path, str) or not path.strip():
        return ""
    src = Path(path)
    if not src.exists():
        print(f"⚠️ Bild fehlt: {src}")
        return ""
    dst = IMG_OUT / src.name
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f"❌ Fehler beim Kopieren {src} → {dst}: {e}")
    return f"images_tegut/{src.name}"

df["Bildpfad"] = df["Bildpfad"].apply(process_img)

# =======================================
# Preis-Spalten als String (und nan-frei)
# =======================================
df["Preis"] = df["Preis"].astype(str).replace("nan", "")
df["Vorheriger Preis"] = df["Vorheriger Preis"].astype(str).replace("nan", "")

# =======================================
# Zielspalten für Streamlit
# =======================================
cols = [
    "Produkt", "Marke", "Preis", "Vorheriger Preis", "Preis_kg",
    "Hinweis", "Nur_online", "Gueltig_raw", "Gueltig_von",
    "Gueltig_bis", "Bildpfad"
]

for c in cols:
    if c not in df.columns:
        df[c] = ""

df = df[cols]

# =======================================
# 💾 Speichern
# =======================================
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n🎉 Fertig! Streamlit-Ready Export erstellt:")
print(OUTPUT)
print("🖼️ Bilder kopiert nach:", IMG_OUT)
