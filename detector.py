from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from visualization import render_yolo_result


MODEL_NAME = "yolov8n.pt"
_model = None
_model_error = None


def load_model():
    """Load the pretrained YOLO model once and reuse it."""
    global _model, _model_error

    if _model is not None:
        return _model

    if _model_error is not None:
        raise RuntimeError(_model_error)

    try:
        _model = YOLO(MODEL_NAME)
        return _model
    except Exception as exc:
        _model_error = f"Could not load YOLO model: {exc}"
        raise RuntimeError(_model_error) from exc


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    return Image.open(image).convert("RGB")


def detect_objects_in_image(
    image: Any,
    confidence_threshold: float = 0.40,
    show_labels: bool = True,
    show_confidence: bool = True,
    hidden_classes: list[str] | None = None,
):
    """
    Detect objects in an uploaded image or PIL image.

    Returns a processed PIL image and a beginner-friendly detection summary.
    """
    model = load_model()
    pil_image = _to_pil_image(image)
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(image_bgr, conf=confidence_threshold, verbose=False)
    annotated_bgr, summary = render_yolo_result(
        image_bgr,
        results[0],
        show_labels=show_labels,
        show_confidence=show_confidence,
        show_track_ids=False,
        hidden_classes=hidden_classes,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(annotated_rgb)

    return processed_image, summary
