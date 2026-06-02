from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import cv2
from PIL import Image

from detector import load_model
from visualization import render_yolo_result


OUTPUT_DIR = Path("outputs")
MAX_VIDEO_SIZE_MB = 100
MAX_VIDEO_SECONDS = 60


def _get_file_size_mb(video_file: Any) -> float:
    if hasattr(video_file, "size"):
        return video_file.size / (1024 * 1024)
    return 0


def _save_uploaded_video(video_file: Any) -> str:
    suffix = Path(video_file.name).suffix if hasattr(video_file, "name") else ".mp4"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(video_file.getbuffer())
        return temp_file.name


def _create_video_writer(output_path: Path, fps: float, size: tuple[int, int]):
    """Prefer browser-friendly MP4 codecs, then fall back to OpenCV's common MP4 codec."""
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError("Could not create a video writer for the processed MP4 output.")


def _save_preview_frame(preview_path: Path, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame_rgb).save(preview_path, quality=90)


def process_video(
    video_file: Any,
    confidence_threshold: float = 0.40,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_track_ids: bool = True,
    hidden_classes: list[str] | None = None,
):
    """
    Process a short uploaded video with YOLO tracking.

    Returns the saved output video path, preview image path, and a summary dictionary.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    size_mb = _get_file_size_mb(video_file)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise ValueError(
            f"Video is too large ({size_mb:.1f} MB). Please use a short demo video under "
            f"{MAX_VIDEO_SIZE_MB} MB."
        )

    model = load_model()
    input_path = _save_uploaded_video(video_file)
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        Path(input_path).unlink(missing_ok=True)
        raise ValueError("Could not read the uploaded video. Please try another file.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    if duration > MAX_VIDEO_SECONDS:
        capture.release()
        Path(input_path).unlink(missing_ok=True)
        raise ValueError(
            f"Video is {duration:.1f} seconds long. Please upload a short video under "
            f"{MAX_VIDEO_SECONDS} seconds for this demo app."
        )

    output_path = OUTPUT_DIR / f"processed_{uuid4().hex}.mp4"
    preview_path = OUTPUT_DIR / f"preview_{uuid4().hex}.jpg"
    writer, codec = _create_video_writer(output_path, fps, (width, height))

    class_counts = Counter()
    tracked_ids = set()
    processed_frames = 0
    preview_saved = False
    fallback_preview_frame = None

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            try:
                results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False)
            except Exception:
                results = model.predict(frame, conf=confidence_threshold, verbose=False)

            result = results[0]
            annotated_frame, frame_summary = render_yolo_result(
                frame,
                result,
                show_labels=show_labels,
                show_confidence=show_confidence,
                show_track_ids=show_track_ids,
                hidden_classes=hidden_classes,
            )
            frame_has_detections = frame_summary["total_objects"] > 0

            class_counts.update(frame_summary["class_counts"])
            tracked_ids.update(frame_summary["tracked_ids"])
            writer.write(annotated_frame)

            if fallback_preview_frame is None:
                fallback_preview_frame = annotated_frame.copy()

            if frame_has_detections and not preview_saved:
                _save_preview_frame(preview_path, annotated_frame)
                preview_saved = True

            processed_frames += 1
    finally:
        capture.release()
        writer.release()
        Path(input_path).unlink(missing_ok=True)

    if not preview_saved and fallback_preview_frame is not None:
        _save_preview_frame(preview_path, fallback_preview_frame)

    summary = {
        "processed_frames": processed_frames,
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "output_codec": codec,
        "confidence_threshold": confidence_threshold,
        "class_counts": dict(class_counts),
        "unique_tracking_ids": len(tracked_ids),
    }

    return str(output_path), str(preview_path), summary
