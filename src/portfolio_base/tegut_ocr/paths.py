from pathlib import Path

# Root = Portfolio_base
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_PDF_NEW     = DATA_DIR / "input" / "pdf_new"
INPUT_PDF_ARCHIVE = DATA_DIR / "input" / "pdf_archive"

PAGES_DIR     = DATA_DIR / "pages"
DETECTIONS_DIR = DATA_DIR / "detections"
LABELS_DIR     = DATA_DIR / "labels"
CROPS_DIR      = DATA_DIR / "crops"
FILTERED_DIR   = DATA_DIR / "filtered"

MODEL_DIR = PROJECT_ROOT / "models"
YOLO_MODEL = MODEL_DIR / "tegut_yolo.pt"

for p in [
    INPUT_PDF_NEW, INPUT_PDF_ARCHIVE,
    PAGES_DIR, DETECTIONS_DIR, LABELS_DIR,
    CROPS_DIR, FILTERED_DIR
]:
    p.mkdir(parents=True, exist_ok=True)
