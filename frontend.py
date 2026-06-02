from pathlib import Path

import streamlit as st
from PIL import Image

from detector import detect_objects_in_image
from visualization import COMMON_CLASSES
from video_processor import process_video


IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4", "avi", "mov"]
MAX_IMAGE_PREVIEW_WIDTH = 850
MAX_VIDEO_PREVIEW_WIDTH = 960
DEFAULT_CONFIDENCE_THRESHOLD = 0.40


def _apply_page_style():
    st.set_page_config(
        page_title="CodeAlpha Object Detection and Tracking",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
            color: #1f2937;
        }
        .main .block-container {
            max-width: 1100px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        [data-testid="stHeader"] {
            background: #f7f9fc;
        }
        h1, h2, h3 {
            color: #111827;
            letter-spacing: 0;
        }
        h1 {
            text-align: center;
            margin-bottom: 0.15rem;
        }
        div[data-testid="stMarkdownContainer"] p {
            line-height: 1.6;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #e5e7eb;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.6rem 0.85rem;
        }
        div[data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            background: #2563EB !important;
            border-color: #2563EB !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
            color: #ffffff !important;
        }
        .subtitle {
            text-align: center;
            color: #4b5563;
            margin-bottom: 0.25rem;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .intro {
            text-align: center;
            color: #374151;
            margin-bottom: 1.5rem;
        }
        .preview-note {
            color: #4b5563;
            font-size: 0.95rem;
            margin-top: -0.25rem;
        }
        div[data-testid="stImage"] img {
            max-height: 620px;
            object-fit: contain;
            border-radius: 8px;
        }
        video {
            display: block;
            width: 100% !important;
            max-width: 960px;
            max-height: 520px;
            object-fit: contain;
            margin: 0 auto;
            border-radius: 8px;
            background: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _file_extension(uploaded_file) -> str:
    return Path(uploaded_file.name).suffix.lower().replace(".", "")


def _resize_for_preview(image: Image.Image, max_width: int = MAX_IMAGE_PREVIEW_WIDTH) -> Image.Image:
    preview = image.copy()
    if preview.width > max_width:
        ratio = max_width / preview.width
        new_size = (max_width, int(preview.height * ratio))
        preview = preview.resize(new_size, Image.LANCZOS)
    return preview


def _show_summary(summary: dict, is_video: bool = False):
    total_objects = summary.get("total_objects")
    if total_objects is not None:
        st.metric("Total objects detected", total_objects)

    if "processed_frames" in summary:
        col1, col2, col3 = st.columns(3)
        col1.metric("Processed frames", summary["processed_frames"])
        col2.metric("Video duration", f'{summary["duration_seconds"]}s')
        col3.metric("Tracking IDs", summary["unique_tracking_ids"])
        st.caption("Tracking IDs may increase when objects move, disappear, reappear, or overlap.")

    class_counts = summary.get("class_counts", {})
    if class_counts:
        if is_video:
            st.subheader("Detection summary across processed frames")
            st.caption("Counts represent detections across processed frames, not unique real-world objects.")
        else:
            st.subheader("Detection summary")
        st.table(
            [{"Object": class_name, "Count": count} for class_name, count in class_counts.items()]
        )
    else:
        st.info("No objects were detected with the current confidence setting.")


def _image_detection_view(confidence_threshold: float, display_options: dict):
    with st.container(border=True):
        st.subheader("Image upload")
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=IMAGE_TYPES,
            help="Supported formats: JPG, JPEG, PNG",
        )

    if uploaded_image is None:
        st.info("Upload an image to start object detection.")
        return

    if _file_extension(uploaded_image) not in IMAGE_TYPES:
        st.error("Unsupported image type. Please upload a JPG, JPEG, or PNG file.")
        return

    try:
        original_image = Image.open(uploaded_image).convert("RGB")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            col1.subheader("Original image")
            col1.image(_resize_for_preview(original_image), use_container_width=True)

            with st.spinner("Detecting objects in the image..."):
                processed_image, summary = detect_objects_in_image(
                    original_image,
                    confidence_threshold=confidence_threshold,
                    show_labels=display_options["show_labels"],
                    show_confidence=display_options["show_confidence"],
                    hidden_classes=display_options["hidden_classes"],
                )

            col2.subheader("Detected objects")
            col2.image(_resize_for_preview(processed_image), use_container_width=True)

        with st.container(border=True):
            _show_summary(summary)
    except Exception as exc:
        st.error(f"Image processing error: {exc}")


def _video_detection_view(confidence_threshold: float, display_options: dict):
    with st.container(border=True):
        st.subheader("Video upload")
        st.markdown('<p class="preview-note">Use short videos for faster processing.</p>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Upload a short video",
            type=VIDEO_TYPES,
            help="Supported formats: MP4, AVI, MOV. Short demo videos work best.",
        )

    if uploaded_video is None:
        st.info("Upload a short video to start object detection and tracking.")
        return

    if _file_extension(uploaded_video) not in VIDEO_TYPES:
        st.error("Unsupported video type. Please upload an MP4, AVI, or MOV file.")
        return

    with st.container(border=True):
        st.subheader("Original video")
        st.markdown('<p class="preview-note">Preview is limited to keep the demo layout readable.</p>', unsafe_allow_html=True)
        st.video(uploaded_video)

    if st.button("Process Video", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing video frames with YOLO. This may take a moment..."):
                output_path, preview_path, summary = process_video(
                    uploaded_video,
                    confidence_threshold=confidence_threshold,
                    show_labels=display_options["show_labels"],
                    show_confidence=display_options["show_confidence"],
                    show_track_ids=display_options["show_track_ids"],
                    hidden_classes=display_options["hidden_classes"],
                )

            st.success("Video processed successfully.")
            with st.container(border=True):
                st.subheader("Processed output video")
                st.markdown('<p class="preview-note">Preview is limited to a compact demo size.</p>', unsafe_allow_html=True)
                st.video(output_path)
                st.caption(
                    "Saved in outputs/. If browser playback has codec issues, use the download button."
                )

                with open(output_path, "rb") as video_file:
                    st.download_button(
                        "Download Processed Video",
                        data=video_file,
                        file_name=Path(output_path).name,
                        mime="video/mp4",
                        use_container_width=True,
                    )

                if Path(preview_path).exists():
                    st.subheader("Processed frame preview")
                    preview_image = Image.open(preview_path).convert("RGB")
                    st.image(
                        _resize_for_preview(preview_image, MAX_VIDEO_PREVIEW_WIDTH),
                        caption="Preview from the processed output video",
                        use_container_width=True,
                    )

            with st.container(border=True):
                _show_summary(summary, is_video=True)
        except Exception as exc:
            st.error(f"Video processing error: {exc}")


def run_app():
    _apply_page_style()

    st.title("CodeAlpha Object Detection and Tracking")
    st.markdown(
        '<div class="subtitle">Artificial Intelligence Internship - Task 4</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="intro">Upload an image or video to detect and track objects using a pretrained YOLO model.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        mode = st.radio(
            "Choose mode",
            ["Image Detection", "Video Detection and Tracking"],
            horizontal=True,
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Higher values hide weak detections. Lower it only if objects are being missed.",
        )
        control_col1, control_col2 = st.columns(2)
        show_labels = control_col1.checkbox("Show labels", value=True)
        show_confidence = control_col2.checkbox("Show confidence", value=True)

        show_track_ids = False
        if mode == "Video Detection and Tracking":
            show_track_ids = st.checkbox("Show track IDs", value=True)

        with st.expander("Advanced visualization settings", expanded=False):
            hidden_classes = st.multiselect(
                "Hide selected classes",
                options=COMMON_CLASSES,
                default=[],
                help="Choose classes manually if the overlay becomes visually noisy.",
            )

    display_options = {
        "show_labels": show_labels,
        "show_confidence": show_confidence,
        "show_track_ids": show_track_ids,
        "hidden_classes": hidden_classes,
    }

    if mode == "Image Detection":
        _image_detection_view(confidence_threshold, display_options)
    else:
        _video_detection_view(confidence_threshold, display_options)
