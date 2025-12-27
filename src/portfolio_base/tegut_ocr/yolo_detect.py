from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import fitz
import cv2
import numpy as np

from portfolio_base.tegut_ocr.paths import (
    PAGES_DIR,
    DETECTIONS_DIR,
    LABELS_DIR,
    CROPS_DIR,
    FILTERED_DIR,
    YOLO_MODEL,
)

# ======================================================
# 🧠 Public API
# ======================================================

def detect_products(pdf_path: Path, dpi: int = 450) -> list[Path]:
    """
    Runs YOLO detection on a flyer PDF and returns product crop paths.

    Parameters
    ----------
    pdf_path : Path
        Path to input flyer PDF
    dpi : int
        Rendering DPI for PDF → PNG

    Returns
    -------
    list[Path]
        Paths to cropped product images
    """

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    kw = datetime.now().isocalendar()[1]
    page_images = _pdf_to_images(pdf_path, dpi)
    results = _run_yolo(page_images, kw)
    crop_paths = _extract_crops(results)
    _save_labels(results)
    _apply_iou_filter(results)

    return crop_paths


# ======================================================
# 🔧 Internals (private helpers)
# ======================================================

def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Path]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        out = PAGES_DIR / f"{pdf_path.stem}_page_{i:02d}.png"
        pix.save(out)
        pages.append(out)

    doc.close()
    return pages


def _run_yolo(page_images: list[Path], kw: int):
    model = YOLO(YOLO_MODEL)

    results = model.predict(
        source=[str(p) for p in page_images],
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(DETECTIONS_DIR),
        name=f"KW{kw:02d}_detect",
        exist_ok=True
    )

    return results


def _extract_crops(results) -> list[Path]:
    crop_paths = []

    for result in results:
        img_path = Path(result.path)
        img = Image.open(img_path)
        img_np = np.array(img)

        for j, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            crop = img_np[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

            crop = _remove_lines(crop)

            name = f"{img_path.stem}_box{j+1:03d}_cls{cls}_conf{conf:.2f}.jpg"
            out = CROPS_DIR / name
            Image.fromarray(crop).save(out)
            crop_paths.append(out)

    return crop_paths


def _remove_lines(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=5
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(crop, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return crop


def _save_labels(results):
    for result in results:
        stem = Path(result.path).stem
        result.save_txt(LABELS_DIR / f"{stem}.txt", save_conf=True)


def _apply_iou_filter(results, threshold: float = 0.9):
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

        cv2.imwrite(str(FILTERED_DIR / Path(result.path).name), img)
