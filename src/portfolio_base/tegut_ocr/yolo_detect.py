from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import fitz
import cv2
import numpy as np
from uuid import uuid4

from portfolio_base.tegut_ocr.paths import DATA_DIR, YOLO_MODEL

# ======================================================
# 🧠 Public API
# ======================================================
def _cleanup_crops(crops_dir: Path):
    for p in crops_dir.glob("*.jpg"):
        p.unlink()


def detect_products(pdf_path: Path, dpi: int = 450):
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # =========================================
    # 🆕 Neuer isolierter Run
    # =========================================
    RUN_DIR = _create_run_dir(DATA_DIR)

    PAGES_DIR = RUN_DIR / "pages"
    YOLO_DIR = RUN_DIR / "yolo"
    CROPS_DIR = RUN_DIR / "crops"
    LABELS_DIR = RUN_DIR / "labels"
    FILTERED_DIR = RUN_DIR / "filtered"

    for d in [
        PAGES_DIR,
        YOLO_DIR,
        CROPS_DIR,
        LABELS_DIR,
        FILTERED_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================
    # Pipeline
    # =========================================
    page_images = _pdf_to_images(pdf_path, dpi, PAGES_DIR)
    results = _run_yolo(page_images, YOLO_DIR)
    crop_infos = _extract_crops(results, CROPS_DIR)

    _save_labels(results, LABELS_DIR)
    _apply_iou_filter(results, FILTERED_DIR)

    return RUN_DIR, crop_infos




# ======================================================
# 🔧 Internals
# ======================================================

def _pdf_to_images(pdf_path: Path, dpi: int, pages_dir: Path) -> list[Path]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        out = pages_dir / f"{pdf_path.stem}_page_{i:02d}.png"
        pix.save(out)
        pages.append(out)

    doc.close()
    return pages

def _create_run_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"run_{ts}_{uuid4().hex[:6]}"
    run_dir = base_dir / "output" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _run_yolo(page_images: list[Path], yolo_dir: Path, min_conf: float = 0.8):
    model = YOLO(YOLO_MODEL)

    return model.predict(
        source=[str(p) for p in page_images],
        conf=0.8,          # ✅ HIER: filtert Boxen schon im YOLO Predict
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(yolo_dir),
        name="detect",
        exist_ok=True
    )



def _extract_crops(results, crops_dir: Path, min_conf: float = 0.8) -> list[dict]:
    crop_infos = []

    raw_dir = crops_dir / "raw"
    ocr_dir = crops_dir / "ocr"
    raw_dir.mkdir(exist_ok=True)
    ocr_dir.mkdir(exist_ok=True)

    for result in results:
        img_path = Path(result.path)
        img = Image.open(img_path)
        img_np = np.array(img)

        for j, box in enumerate(result.boxes):
            conf = float(box.conf[0])

            # =========================================
            # ❌ CONFIDENCE-FILTER
            # =========================================
            if conf < min_conf:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = map(int, xyxy)
            cls = int(box.cls[0])

            crop_raw = img_np[y1:y2, x1:x2]
            crop_ocr = _remove_lines(crop_raw.copy())

            base = f"{img_path.stem}_box{j+1:03d}_cls{cls}_conf{conf:.2f}"

            raw_path = raw_dir / f"{base}.jpg"
            ocr_path = ocr_dir / f"{base}.jpg"

            Image.fromarray(crop_raw).save(raw_path)
            Image.fromarray(crop_ocr).save(ocr_path)

            crop_infos.append({
                "raw_path": raw_path,
                "ocr_path": ocr_path,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "cls": cls,
                "page": img_path.name
            })

    return crop_infos

def _remove_lines(crop: np.ndarray) -> np.ndarray:
    return crop

def _save_labels(results, labels_dir: Path):
    for result in results:
        stem = Path(result.path).stem
        result.save_txt(labels_dir / f"{stem}.txt", save_conf=True)


def _apply_iou_filter(results, filtered_dir: Path, threshold: float = 0.9):
    def iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return inter / float(areaA + areaB - inter)

    for result in results:
        img = cv2.imread(result.path)
        boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in result.boxes]

        keep = []
        for i, boxA in enumerate(boxes):
            if any(iou(boxA, boxB) > threshold for j, boxB in enumerate(boxes) if i != j):
                continue
            keep.append(boxA)

        for x1, y1, x2, y2 in keep:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(str(filtered_dir / Path(result.path).name), img)
