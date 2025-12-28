import sys

import numpy as np
from pathlib import Path
from datetime import datetime

# -------------------------------------------------
# src/ zum Python-Pfad hinzufügen (GANZ AM ANFANG!)
# -------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[2]  # -> .../Portfolio_base/src
sys.path.insert(0, str(SRC_ROOT))

# -------------------------------------------------
# JETZT normale Imports
# -------------------------------------------------
import streamlit as st
from PIL import Image
import tempfile
import re
import zipfile
import shutil


from portfolio_base.tegut_ocr.yolo_detect import detect_products
from portfolio_base.tegut_ocr.ocr_easy import extract_text_easyocr

import shutil

def recrop_from_yolo_labels(run_dir: Path) -> list[dict]:
    pages_dir = run_dir / "pages"
    labels_dir = run_dir / "labels"
    crops_dir = run_dir / "crops"

    raw_dir = crops_dir / "raw"
    ocr_dir = crops_dir / "ocr"

    # alte Crops löschen
    if crops_dir.exists():
        import shutil
        shutil.rmtree(crops_dir)

    raw_dir.mkdir(parents=True)
    ocr_dir.mkdir(parents=True)

    crop_infos = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        page_stem = label_file.stem
        img_path = pages_dir / f"{page_stem}.png"

        if not img_path.exists():
            continue

        img = Image.open(img_path)
        W, H = img.size
        img_np = np.array(img)

        lines = label_file.read_text().splitlines()

        for i, line in enumerate(lines):
            cls, xc, yc, w, h = map(float, line.split())

            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)

            crop = img_np[y1:y2, x1:x2]

            base = f"{page_stem}_box{i+1:03d}_cls{int(cls)}"

            raw_path = raw_dir / f"{base}.jpg"
            ocr_path = ocr_dir / f"{base}.jpg"

            Image.fromarray(crop).save(raw_path)
            Image.fromarray(crop).save(ocr_path)

            crop_infos.append({
                "raw_path": raw_path,
                "ocr_path": ocr_path,
                "bbox": (x1, y1, x2, y2),
                "cls": int(cls),
                "page": page_stem
            })

    return crop_infos


def strip_confidence(lines: list[str]) -> list[str]:
    clean = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            clean.append(" ".join(parts[:5]))
    return clean


def zip_directory(dir_path: Path) -> Path:
    zip_path = dir_path.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(dir_path), "zip", dir_path)
    return zip_path

def export_to_makesense_image_first(run_dir: Path) -> Path:
    """
    Export für MakeSense (image-first, Bug-kompatibel)
    labels.txt liegt IM labels-Ordner
    """
    pages_dir  = run_dir / "pages"
    labels_dir = run_dir / "labels"

    out_dir = run_dir / "makesense_export"
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"

    # clean
    if out_dir.exists():
        shutil.rmtree(out_dir)

    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    # Images kopieren
    for img in pages_dir.glob("*.png"):
        shutil.copy(img, images_out / img.name)

    # YOLO-Labels kopieren + Confidence strippen
    for lbl in labels_dir.glob("*.txt"):
        raw_lines = lbl.read_text(encoding="utf-8").splitlines()
        clean_lines = strip_confidence(raw_lines)

        (labels_out / lbl.name).write_text(
            "\n".join(clean_lines),
            encoding="utf-8"
        )

    # 🔥 WICHTIG: labels.txt IM labels-Ordner
    (labels_out / "labels.txt").write_text(
        "product\n",
        encoding="utf-8"
    )

    return out_dir

def import_from_makesense_image_first(uploaded_files, run_dir: Path):
    """
    Importiert YOLO-Labels (cls xc yc w h) aus MakeSense
    """
    labels_dir = run_dir / "labels"

    # alte Labels löschen
    for p in labels_dir.glob("*.txt"):
        p.unlink()

    # neue Labels speichern
    for f in uploaded_files:
        target = labels_dir / f.name
        target.write_bytes(f.read())



def get_yolo_page_images(run_dir: Path) -> list[Path]:
    yolo_dir = run_dir / "yolo" / "detect"

    if not yolo_dir.exists():
        return []

    pages = {}
    for img in sorted(yolo_dir.glob("*.jpg")):
        m = re.search(r"_page_(\d+)", img.name)
        if not m:
            continue

        page_idx = int(m.group(1))
        if page_idx not in pages:
            pages[page_idx] = img

    return [pages[k] for k in sorted(pages)]


# ======================================================
# 1️⃣ PDF Upload
# ======================================================

uploaded_file = st.file_uploader(
    "📄 PDF-Seite hochladen",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Bitte eine einzelne PDF-Seite hochladen.")
    st.stop()

# ------------------------------------------------------
# PDF temporär speichern
# ------------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = Path(tmp.name)

st.success("PDF hochgeladen")

# ======================================================
# 2️⃣ Object Detection
# ======================================================

if st.button("🔍 Produkte erkennen"):
    with st.spinner("YOLO erkennt Produkte …"):
        run_dir, crop_infos = detect_products(pdf_path)

        st.session_state["RUN_DIR"] = run_dir
        st.session_state["crop_paths"] = crop_infos

        st.success(f"{len(crop_infos)} Produkte erkannt")

# ======================================================
# 2️⃣a YOLO-Seitenansicht (GANZE SEITE + BOXEN)
# ======================================================

if "crop_paths" in st.session_state:
    st.markdown("## 🖼️ YOLO-Ergebnis – Seite mit Bounding Boxes")

    RUN_DIR = st.session_state["RUN_DIR"]
    yolo_pages = get_yolo_page_images(RUN_DIR)

    if not yolo_pages:
        st.warning("⚠️ Keine YOLO-Seitenbilder gefunden.")
    else:
        for page_img in yolo_pages:
            st.image(
                Image.open(page_img),
                caption=f"YOLO Detection: {page_img.name}",
                use_container_width=True
            )

        st.info(
            "🔍 **Hinweis**\n\n"
            "- Bounding Boxes stammen direkt aus YOLO\n"
            "- Confidence ist in den Boxen enthalten\n"
            "- Boxen sind aktuell **read-only**\n\n"
            "➡️ Interaktive Bearbeitung folgt im nächsten Schritt"
        )

# ======================================================
# 🔵 3️⃣a MakeSense – Export (image-first)
# ======================================================

if "RUN_DIR" in st.session_state:
    if st.button("⬇️ Für MakeSense vorbereiten"):
        out_dir = export_to_makesense_image_first(st.session_state["RUN_DIR"])
        zip_path = zip_directory(out_dir)

        with open(zip_path, "rb") as f:
            st.download_button(
                "📦 MakeSense-Export herunterladen (ZIP)",
                f,
                file_name="makesense_export.zip",
                mime="application/zip"
            )

# ======================================================
# 🔵 3️⃣b MakeSense – Import (image-first)
# ======================================================

st.markdown("## ⬆️ MakeSense-Labels importieren")

uploaded_labels = st.file_uploader(
    "YOLO-Labels aus MakeSense hochladen (*.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_labels and st.button("📥 Labels importieren"):
    import_from_makesense_image_first(
        uploaded_labels,
        st.session_state["RUN_DIR"]
    )

    # 🔁 NEU: Crops aus neuen Labels erzeugen
    new_crops = recrop_from_yolo_labels(st.session_state["RUN_DIR"])
    st.session_state["crop_paths"] = new_crops

    st.success(f"✔ {len(new_crops)} Crops neu erzeugt – bereit für OCR")


# ======================================================
# 4️⃣ Crops anzeigen & auswählen
# ======================================================

if "crop_paths" in st.session_state:
    st.markdown("## 🧩 Erkannte Produkt-Crops")

    selected = []
    cols = st.columns(4)

    for i, crop in enumerate(st.session_state["crop_paths"]):
        # ✅ crop ist jetzt ein dict → raw_path verwenden
        img_path = crop["raw_path"]

        with cols[i % 4]:
            img = Image.open(img_path)
            st.image(img, use_container_width=True)

            if st.checkbox("OCR anwenden", key=f"ocr_{i}"):
                selected.append(crop)

    st.session_state["selected_crops"] = selected


# ======================================================
# 5️⃣ OCR auf ausgewählte Crops
# ======================================================

if st.button("🔤 OCR auf Auswahl anwenden"):
    if not st.session_state.get("selected_crops"):
        st.warning("Bitte mindestens einen Crop auswählen.")
        st.stop()

    st.markdown("## 📑 OCR-Ergebnisse")

    for crop in st.session_state["selected_crops"]:
        # OCR immer auf ocr_path anwenden
        res = extract_text_easyocr(crop["ocr_path"])

        with st.expander(crop["raw_path"].name):
            st.image(Image.open(crop["raw_path"]), width=300)
            st.text(res["text"])
            st.caption(f"OCR-Confidence: {res['mean_confidence']:.2f}")
