"""
Object Detection using YOLOv8.

Detects objects in an input image and returns bounding boxes,
labels, and confidence scores.
"""
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO
from .config import YOLO_MODEL_NAME, YOLO_CONFIDENCE_THRESHOLD, DEVICE

_yolo_model = None


def get_yolo_model() -> YOLO:
    """Lazy-load YOLOv8 model (downloads on first use)."""
    global _yolo_model
    if _yolo_model is None:
        print(f"[Detection] Loading YOLOv8 model: {YOLO_MODEL_NAME}")
        _yolo_model = YOLO(YOLO_MODEL_NAME)
        print("[Detection] YOLOv8 model loaded.")
    return _yolo_model


def detect_objects(image_bytes: bytes) -> list[dict]:
    """
    Run YOLOv8 detection on an image.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG).

    Returns:
        List of detections, each containing:
            - label: COCO class name
            - confidence: float 0-1
            - bbox: [x1, y1, x2, y2] in pixels
            - class_id: COCO class index
    """
    model = get_yolo_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(
        source=image,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            detections.append({
                "label": model.names[cls_id],
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in xyxy],
                "class_id": cls_id,
            })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def crop_detection(image_bytes: bytes, bbox: list[float], padding: int = 10) -> Image.Image:
    """
    Crop the detected region from the image with optional padding.

    Args:
        image_bytes: Raw image bytes.
        bbox: [x1, y1, x2, y2] bounding box.
        padding: Extra pixels around the crop.

    Returns:
        Cropped PIL Image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    return image.crop((x1, y1, x2, y2))
