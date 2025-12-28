from pathlib import Path

# ======================================================
# 📦 Package root
# ======================================================
# src/portfolio_base/tegut_ocr/paths.py
# parents[1] -> src/portfolio_base
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ======================================================
# 📁 Statische Projektverzeichnisse
# ======================================================
DATA_DIR = PACKAGE_ROOT / "data"
MODELS_DIR = PACKAGE_ROOT / "models"

# ======================================================
# 🧠 YOLO model
# ======================================================
YOLO_MODEL = MODELS_DIR / "tegut_yolo.pt"

# ======================================================
# 🧪 Safety check
# ======================================================
if not YOLO_MODEL.exists():
    raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL}")
