from collections import Counter
import colorsys
from hashlib import md5
from typing import Iterable

import cv2


DEFAULT_HIDDEN_CLASSES = ["dining table"]
COMMON_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "cat",
    "dog",
    "backpack",
    "umbrella",
    "handbag",
    "bottle",
    "cup",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "cell phone",
    "book",
    "clock",
]


def color_for_class(class_name: str) -> tuple[int, int, int]:
    digest = md5(class_name.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.68
    value = 0.88
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(blue * 255), int(green * 255), int(red * 255)


def format_label(
    class_name: str,
    confidence: float,
    track_id: int | None = None,
    show_confidence: bool = True,
    show_track_id: bool = False,
) -> str:
    parts = [class_name]
    if show_track_id and track_id is not None:
        parts.append(f"#{track_id}")
    if show_confidence:
        parts.append(f"{confidence:.2f}")
    return " ".join(parts)


def _overlaps(rect: tuple[int, int, int, int], occupied: Iterable[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = rect
    for ox1, oy1, ox2, oy2 in occupied:
        if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
            return True
    return False


def _label_rect(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_width: int,
    text_height: int,
    baseline: int,
    image_width: int,
    image_height: int,
    occupied_labels: Iterable[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    padding_x = 5
    padding_y = 4
    label_width = text_width + padding_x * 2
    label_height = text_height + baseline + padding_y * 2
    label_x1 = max(0, min(x1, image_width - label_width))

    candidates = [
        y1 - label_height,
        y1 + 2,
        y2 - label_height - 2,
        y2 + 2,
        y1 + label_height + 4,
        y2 - label_height * 2 - 4,
    ]

    fallback = None
    for candidate_y in candidates:
        label_y1 = max(0, min(candidate_y, image_height - label_height))
        rect = (label_x1, label_y1, label_x1 + label_width, label_y1 + label_height)
        if rect[3] <= image_height and fallback is None:
            fallback = rect
        if rect[3] <= image_height and not _overlaps(rect, occupied_labels):
            return rect

    return fallback or (label_x1, 0, label_x1 + label_width, label_height)


def draw_clean_box(
    image_bgr,
    box_xyxy: tuple[int, int, int, int],
    label: str,
    class_name: str,
    occupied_labels: list[tuple[int, int, int, int]],
    show_labels: bool = True,
):
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    color = color_for_class(class_name)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)

    if not show_labels or not label:
        return image_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    rect = _label_rect(
        x1,
        y1,
        x2,
        y2,
        text_width,
        text_height,
        baseline,
        width,
        height,
        occupied_labels,
    )

    if _overlaps(rect, occupied_labels):
        return image_bgr

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    cv2.addWeighted(overlay, 0.86, image_bgr, 0.14, 0, image_bgr)
    text_x = rect[0] + 5
    text_y = rect[1] + text_height + 4
    cv2.putText(
        image_bgr,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    occupied_labels.append(rect)
    return image_bgr


def _box_track_id(box) -> int | None:
    if box.id is None:
        return None
    return int(box.id[0])


def render_yolo_result(
    image_bgr,
    result,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_track_ids: bool = False,
    hidden_classes: Iterable[str] | None = None,
):
    hidden = set(hidden_classes or [])
    rendered = image_bgr.copy()
    occupied_labels = []
    class_counts = Counter()
    detections = []
    tracked_ids = set()

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names.get(class_id, str(class_id))
        if class_name in hidden:
            continue

        confidence = float(box.conf[0])
        track_id = _box_track_id(box)
        if track_id is not None:
            tracked_ids.add(track_id)

        xyxy = box.xyxy[0].tolist()
        box_xyxy = tuple(int(round(value)) for value in xyxy)
        label = format_label(
            class_name,
            confidence,
            track_id=track_id,
            show_confidence=show_confidence,
            show_track_id=show_track_ids,
        )

        draw_clean_box(
            rendered,
            box_xyxy,
            label,
            class_name,
            occupied_labels,
            show_labels=show_labels,
        )
        class_counts[class_name] += 1
        detections.append(
            {
                "class": class_name,
                "confidence": round(confidence, 3),
                "track_id": track_id,
            }
        )

    return rendered, {
        "total_objects": len(detections),
        "class_counts": dict(class_counts),
        "detections": detections,
        "tracked_ids": tracked_ids,
    }
