import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from model.pred_func import (
    df_face,
    face_rec,
    is_video,
    load_genconvit,
    pred_vid,
    preprocess_frame,
    real_or_fake,
    real_or_fake_thres,
)
from model.text_pred_func import load_text_model, predict_text

st.set_page_config(
    page_title="GenConViT Streamlined",
    page_icon="🎬",
    layout="wide",
)

SAMPLE_DIR = Path("sample_prediction_data")
WEIGHT_DIR = Path("weight")
SUPPORTED_VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mpg", ".mpeg")
SUPPORTED_IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _normalize_weight_name(raw_name: str) -> str:
    """Return a sanitized weight filename without directories or suffix."""
    clean_name = raw_name.strip()
    if not clean_name:
        return clean_name
    clean_name = Path(clean_name)
    if clean_name.suffix == ".pth":
        clean_name = clean_name.with_suffix("")
    return clean_name.name


@st.cache_resource(show_spinner=False)
def _load_cached_model(net: str, fp16: bool, ed_weight: str, vae_weight: str):
    """Cache the GenConViT model so it is only loaded once per configuration."""
    return load_genconvit(
        net=net,
        fp16=fp16,
        ed=_normalize_weight_name(ed_weight),
        vae=_normalize_weight_name(vae_weight),
    )


@st.cache_resource(show_spinner="Loading BERT model...")
def _load_cached_text_model():
    """Cache the BERT text model so it is only loaded once."""
    model, tokenizer = load_text_model()
    return model, tokenizer


def _save_uploaded_video(uploaded_file) -> Optional[str]:
    """Persist an uploaded file to a temporary location on disk."""
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def _run_video_inference(
    video_path: str,
    net: str,
    fp16: bool,
    num_frames: int,
    threshold: float,
    ed_weight: str,
    vae_weight: str,
):
    """Execute the GenConViT pipeline for a single video file."""
    model = _load_cached_model(net, fp16, ed_weight, vae_weight)
    frames = df_face(video_path, num_frames, net)

    if isinstance(frames, list) or len(frames) == 0:
        raise ValueError("No faces detected in the selected video.")

    if fp16:
        frames = frames.half()

    label_idx, probability = pred_vid(frames, model)
    predicted_label = real_or_fake(label_idx)
    threshold_label = real_or_fake_thres(probability, threshold)

    return {
        "predicted_label": predicted_label,
        "df_probability": float(probability),
        "threshold_label": threshold_label,
    }


def _prepare_image_tensor(image_bytes: bytes):
    """Convert raw image bytes into a normalized tensor ready for inference."""
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode the uploaded image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces, count = face_rec([image_rgb])
    if count == 0:
        raise ValueError("No faces detected in the uploaded image.")

    return preprocess_frame(faces)


def _run_image_inference(
    image_tensor,
    net: str,
    fp16: bool,
    threshold: float,
    ed_weight: str,
    vae_weight: str,
):
    """Execute the GenConViT pipeline for a single image."""
    model = _load_cached_model(net, fp16, ed_weight, vae_weight)

    if fp16:
        image_tensor = image_tensor.half()

    label_idx, probability = pred_vid(image_tensor, model)
    predicted_label = real_or_fake(label_idx)
    threshold_label = real_or_fake_thres(probability, threshold)

    return {
        "predicted_label": predicted_label,
        "df_probability": float(probability),
        "threshold_label": threshold_label,
    }


def _sidebar_controls():
    st.sidebar.title("⚙️ Configuration")
    
    # --- Visual Media Controls ---
    st.sidebar.header("📹 Visual Deepfake Settings")
    
    st.sidebar.info(
        "These settings control how the AI analyzes videos and images to detect face manipulations."
    )

    # 1. Network Selection (Simplified: Kept available but with better context)
    net = st.sidebar.selectbox(
        "Model Architecture", 
        options=["genconvit", "ed", "vae"],
        help="Select the internal architecture used for detection. 'genconvit' is usually the most robust."
    )

    # 2. Frame Count
    num_frames = st.sidebar.slider(
        "Frames to Analyze", 
        min_value=4, 
        max_value=32, 
        value=15,
        help="Higher values increase accuracy but take longer to process. 15 is a good balance."
    )

    # 3. FP16 (Renamed for clarity)
    fp16 = st.sidebar.toggle(
        "⚡ Enable Fast Mode (FP16)", 
        value=False,
        help="Uses Half-Precision (FP16). Faster and uses less memory, but requires a compatible GPU. Disable if you experience errors."
    )

    # 4. Visual Threshold
    threshold = st.sidebar.slider(
        "Fake Sensitivity Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.2,
        step=0.05,
        help="How strict should the AI be? Lower values (e.g., 0.1) flag fakes more easily but might cause false alarms. Higher values (e.g., 0.8) only flag obvious fakes."
    )

    # --- Internal Weight Configuration (Hidden from User) ---
    # Instead of asking the user to type filenames, we define them here.
    # If you need to change these, change them in the code, not the UI.
    default_ed = "genconvit_ed_inference"
    default_vae = "genconvit_vae_inference"

    # Check if weights exist silently
    ed_path = WEIGHT_DIR / f"{default_ed}.pth"
    vae_path = WEIGHT_DIR / f"{default_vae}.pth"
    
    if not ed_path.exists() or not vae_path.exists():
        st.sidebar.error(f"⚠️ System Error: Model weights missing in '{WEIGHT_DIR}'.")

    # --- Text Controls ---
    st.sidebar.markdown("---") # Visual separator
    st.sidebar.header("📝 Text Analysis Settings")
    
    text_threshold = st.sidebar.slider(
        "AI Text Sensitivity",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Threshold for classifying text as AI-generated. Values > 0.5 require high confidence to label as AI."
    )
    
    # Add an About section at the bottom
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About this App"):
        st.markdown(
            """
            **GenConViT Streamlined** uses a Vision Transformer for video deepfake detection 
            and DistilBERT for AI text detection.
            
            *Adjust sensitivity sliders if you are getting too many False Positives.*
            """
        )

    # Return the hardcoded weights along with user settings
    return net, num_frames, fp16, threshold, default_ed, default_vae, text_threshold


def _video_section(net, num_frames, fp16, threshold, ed_weight, vae_weight):
    st.subheader("Video Deepfake Detection")
    st.markdown(
        "Upload a short clip or pick one of the bundled samples to run inference with "
        "GenConViT. The model processes a fixed number of frames and looks for the most "
        "confident face detections before making a prediction."
    )

    source = st.radio(
        "Choose a video source",
        options=["Upload a file", "Use sample data"],
        horizontal=True,
        key="video_source_radio",
    )

    selected_video_path = None
    uploaded_file = None

    if source == "Upload a file":
        uploaded_file = st.file_uploader(
            "MP4, MOV, AVI, MPG, or MPEG files only",
            type=[ext.replace(".", "") for ext in SUPPORTED_VIDEO_TYPES],
            key="video_uploader",
        )
        if uploaded_file is not None:
            selected_video_path = _save_uploaded_video(uploaded_file)
            st.video(uploaded_file)
    else:
        sample_files = sorted(
            p.name for p in SAMPLE_DIR.glob("*") if p.suffix in SUPPORTED_VIDEO_TYPES
        )
        sample_choice = st.selectbox(
            "Sample videos",
            options=["Select a file"] + sample_files,
            index=0,
            key="video_sample_select",
        )
        if sample_choice != "Select a file":
            selected_video_path = str(SAMPLE_DIR / sample_choice)
            st.video(selected_video_path)

    run_button = st.button(
        "Run video analysis",
        type="primary",
        disabled=selected_video_path is None,
        key="video_run_button",
    )

    if run_button:
        if selected_video_path is None:
            st.error("Please upload or select a video before running the analysis.")
            return

        if not is_video(selected_video_path):
            st.error("The selected file is not a supported video format.")
            return

        with st.spinner("Analyzing video… this may take a minute."):
            try:
                result = _run_video_inference(
                    selected_video_path,
                    net,
                    fp16,
                    num_frames,
                    threshold,
                    ed_weight,
                    vae_weight,
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except ValueError as exc:
                st.warning(str(exc))
                return
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Inference failed: {exc}")
                return

        st.success("Inference complete!")
        st.metric("Raw prediction", result["predicted_label"])
        st.metric("Fake probability", f"{result['df_probability']:.3f}")
        st.metric(
            "Thresholded label",
            result["threshold_label"],
            help=f"Computed using the {threshold:.2f} threshold slider above.",
        )

        st.caption(
            "Raw prediction uses the model's internal logits, while the thresholded label "
            "compares the fake probability against your adjustable threshold."
        )

    if uploaded_file is not None and selected_video_path:
        st.caption(f"Uploaded file saved temporarily at `{selected_video_path}`.")


def _image_section(net, fp16, threshold, ed_weight, vae_weight):
    st.subheader("Image Deepfake Detection")
    st.markdown(
        "Upload a single frame or headshot and run GenConViT directly on the detected faces."
    )
    uploaded_image = st.file_uploader(
        "PNG, JPG, JPEG, BMP, or WEBP files only",
        type=[ext.replace(".", "") for ext in SUPPORTED_IMAGE_TYPES],
        key="image_uploader",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption=uploaded_image.name)

    run_button = st.button(
        "Run image analysis",
        type="primary",
        disabled=uploaded_image is None,
        key="image_run_button",
    )

    if run_button:
        if uploaded_image is None:
            st.error("Please upload an image before running the analysis.")
            return

        with st.spinner("Analyzing image…"):
            try:
                image_tensor = _prepare_image_tensor(uploaded_image.getvalue())
                result = _run_image_inference(
                    image_tensor,
                    net,
                    fp16,
                    threshold,
                    ed_weight,
                    vae_weight,
                )
            except ValueError as exc:
                st.warning(str(exc))
                return
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Inference failed: {exc}")
                return

        st.success("Inference complete!")
        st.metric("Raw prediction", result["predicted_label"])
        st.metric("Fake probability", f"{result['df_probability']:.3f}")
        st.metric(
            "Thresholded label",
            result["threshold_label"],
            help=f"Computed using the {threshold:.2f} threshold slider above.",
        )


def _text_tab(text_threshold: float):
    st.header("AI-Generated Text Detection")
    st.markdown(
        "Enter text below to analyze whether it was written by a human or generated by AI. "
        "The model uses DistilBERT (a memory-efficient BERT variant) to analyze linguistic patterns and features."
    )
    st.info(
        "💡 **Memory Optimized**: Using DistilBERT for lower memory usage. "
        "The model will automatically download on first use."
    )
    
    text_input = st.text_area(
        "Enter text to analyze",
        height=200,
        placeholder="Paste or type the text you want to analyze here...",
        key="text_input_area",
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button(
            "Analyze text",
            type="primary",
            disabled=not text_input.strip(),
            key="text_run_button",
        )
    
    if run_button:
        if not text_input.strip():
            st.error("Please enter some text before running the analysis.")
            return
        
        with st.spinner("Analyzing text with BERT model…"):
            try:
                model, tokenizer = _load_cached_text_model()
                result = predict_text(text_input, model, tokenizer)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Text analysis failed: {exc}")
                return
        
        st.success("Analysis complete!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", result["label"])
            st.metric(
                "AI Probability",
                f"{result['ai_probability']:.3f}",
                help="Probability that the text is AI-generated (0.0 = human, 1.0 = AI)",
            )
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.3f}")
            threshold_label = "AI-Generated" if result["ai_probability"] >= text_threshold else "Human-Written"
            st.metric(
                "Thresholded label",
                threshold_label,
                help=f"Computed using the {text_threshold:.2f} threshold.",
            )
        
        # Progress bar for visualization
        st.progress(result["ai_probability"])
        st.caption(
            f"AI-Generated: {result['ai_probability']:.1%} | "
            f"Human-Written: {result['human_probability']:.1%}"
        )
        
        # Additional info
        with st.expander("Analysis details"):
            st.write(f"**Text length:** {len(text_input)} characters")
            st.write(f"**Word count:** {len(text_input.split())} words")
            st.write(f"**Model confidence:** {result['confidence']:.1%}")
            st.write(
                "The model uses BERT (Bidirectional Encoder Representations from Transformers) "
                "to analyze linguistic patterns and features that distinguish AI-generated text "
                "from human-written content."
            )


def _visual_media_tab(net, num_frames, fp16, threshold, ed_weight, vae_weight):
    st.header("Visual Media Deepfake Detection")
    media_mode = st.radio(
        "Select media type",
        options=["Video", "Image"],
        horizontal=True,
        key="media_mode_selector",
    )

    if media_mode == "Video":
        _video_section(net, num_frames, fp16, threshold, ed_weight, vae_weight)
    else:
        _image_section(net, fp16, threshold, ed_weight, vae_weight)


def main():
    net, num_frames, fp16, threshold, ed_weight, vae_weight, text_threshold = _sidebar_controls()

    visual_tab, text_tab = st.tabs(["Visual Media", "Text"])

    with visual_tab:
        _visual_media_tab(net, num_frames, fp16, threshold, ed_weight, vae_weight)
    with text_tab:
        _text_tab(text_threshold)


if __name__ == "__main__":
    if not WEIGHT_DIR.exists():
        os.makedirs(WEIGHT_DIR, exist_ok=True)
    main()

