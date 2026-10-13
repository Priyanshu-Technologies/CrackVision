
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import time
from src.crackvision.inference_utils import predict_image
import threading

class CameraStream:
    """Reads frames continuously in a background thread so we always get the newest frame."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.lock = threading.Lock()
        self.running = self.cap.isOpened()
        self.thread = None
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.cap.release()

ROOT = Path(__file__).resolve().parent

MODELS = {
    "Balanced · 640px": ROOT / "models/crackvision-yolo11n-seg-best.pt",
    "Detail · 960px": ROOT / "models/crackvision-yolo11n-seg-960-best.pt",
}

st.set_page_config(
    page_title="CrackVision | AI Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1120 0%, #101a2d 55%, #111329 100%);
    color: #e5edf8;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
section[data-testid="stSidebar"] {
    background: #0d1525;
    border-right: 1px solid #25334b;
}
.hero {
    padding: 1.8rem;
    border: 1px solid #293a55;
    border-radius: 22px;
    background: linear-gradient(120deg, #14243b, #18203a 60%, #251d45);
    margin-bottom: 1.5rem;
}
.eyebrow {
    color: #22d3ee;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .18em;
}
.hero h1 {
    font-size: clamp(2rem, 4vw, 3.2rem);
    margin: .4rem 0;
    color: #f1f5f9;
}
.hero p {
    color: #aabbd1;
    margin-bottom: 0;
}
.panel {
    background: #111d30;
    border: 1px solid #293a55;
    border-radius: 17px;
    padding: 1.1rem;
    min-height: 100px;
}
.panel-label {
    color: #93a8c3;
    font-size: .8rem;
    margin-bottom: .5rem;
}
.panel-value {
    font-size: 1.6rem;
    font-weight: 750;
    color: #f1f5f9;
}
div[data-testid="stMetric"] {
    background: #111d30;
    border: 1px solid #293a55;
    border-radius: 14px;
    padding: 1rem;
}
div[data-testid="stMetricLabel"] {
    color: #aabbd1;
}
div.stButton > button {
    border-radius: 10px;
    border: 1px solid #22d3ee;
    min-height: 2.7rem;
    font-weight: 700;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #0891b2, #6366f1);
    color: white;
    border: none;
}
div[data-testid="stFileUploader"] {
    background: #111d30;
    border: 1px dashed #42617e;
    border-radius: 14px;
    padding: 1rem;
}
h2, h3 {
    color: #e6f1ff;
}
.stCaption {
    color: #93a8c3;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def run_inspection(model, image_rgb, image_size, confidence):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Analysing surface with CrackVision AI..."):
        result = predict_image(
            model=model,
            image=image_bgr,
            image_size=image_size,
            confidence=confidence,
        )

        # Extract prediction information
        detected_count = len(result.boxes) if result.boxes is not None else 0

        confidence_scores = (
            result.boxes.conf.tolist()
            if result.boxes is not None
            else []
        )

        # Build a summary from the model's predictions
        if detected_count > 0:
            average_confidence = sum(confidence_scores) / detected_count

            st.success(f"Detected {detected_count} potential crack instance(s).")
            st.write(
                f"Average prediction confidence: "
                f"{average_confidence:.1%}"
            )
        else:
            st.info("No cracks were detected at the selected confidence threshold.")


    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0

    st.markdown("### Inspection results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Defect instances", count)
    m2.metric(
        "Highest confidence",
        f"{max((float(b.conf[0]) for b in boxes), default=0):.1%}"
        if count else "—",
    )
    m3.metric("Input resolution", f"{image_rgb.shape[1]} × {image_rgb.shape[0]}")

    original_tab, result_tab = st.tabs(
        ["🖼️ Original surface", "🔎 AI segmentation"]
    )

    with original_tab:
        st.image(image_rgb, use_container_width=True)

    with result_tab:
        st.image(annotated, use_container_width=True)

        st.download_button(
            "⬇️ Download annotated image",
            data=Image.fromarray(annotated).tobytes()
            if False else _image_bytes(annotated),
            file_name="crackvision-result.png",
            mime="image/png",
            use_container_width=True,
        )

    if count:
        rows = []
        for i, box in enumerate(boxes):
            rows.append({
                "Instance": i + 1,
                "Defect class": model.names[int(box.cls[0])],
                "Confidence": round(float(box.conf[0]), 3),
            })

        st.markdown("### Detected instances")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No defects detected at this threshold. "
            "This does not guarantee that the surface is defect-free."
        )


def _image_bytes(image_rgb):
    from io import BytesIO
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


with st.sidebar:
    st.markdown("## 🔬 CrackVision")
    st.caption("INSPECTION CONTROL CENTER")
    st.divider()

    mode = st.radio(
        "Inspection source",
        ["📁 Upload image", "📷 Camera snapshot"],
    )

    model_choice = st.selectbox("Segmentation model", list(MODELS))
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    image_size = st.select_slider(
        "Inference size",
        options=[320, 416, 512, 640, 960],
        value=640 if model_choice.startswith("Balanced") else 960,
    )

    st.divider()
    st.markdown("**System status**")
    st.success("Model weights available" if MODELS[model_choice].is_file()
               else "Model weights missing")
    st.caption("Inference device: CPU")


st.sidebar.markdown("---")

st.sidebar.link_button(
    "Connect with Developer",
    "https://www.linkedin.com/in/priyanshu-sharma-646233394/",
    use_container_width=True,
)

st.markdown("""
<div class="hero">
    <div class="eyebrow">COMPUTER VISION · INDUSTRIAL AI</div>
    <h1>CrackVision <span style="color:#22d3ee">/</span> Surface Intelligence</h1>
    <p>Inspect surfaces, segment visible defects, and review AI predictions
    in one workspace.</p>
</div>
""", unsafe_allow_html=True)

if not MODELS[model_choice].is_file():
    st.error(f"Model not found: {MODELS[model_choice]}")
    st.stop()

model = load_model(str(MODELS[model_choice]))

if mode == "📁 Upload image":
    st.markdown("### Choose a surface to inspect")
    uploaded = st.file_uploader(
        "Upload a steel surface image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Choose a clear image of the steel surface.",
    )

    if uploaded:
        image_rgb = np.array(Image.open(uploaded).convert("RGB"))

        if st.button(" Analyse surface", type="primary",
                     use_container_width=True):
            run_inspection(model, image_rgb, image_size, confidence)


else:
    st.markdown("### Live camera inspection")
    st.caption(
        "Start the feed to detect and segment defects from your local camera."
    )

    camera_index = st.selectbox("Camera device", [0, 1], index=0)

    if "live_camera_running" not in st.session_state:
        st.session_state.live_camera_running = False

    if st.session_state.live_camera_running:
        if st.button("Stop live inspection", type="primary"):
            st.session_state.live_camera_running = False
            stream = st.session_state.pop("live_camera", None)
            if stream is not None:
                stream.release()
            st.rerun()
    else:
        if st.button("Start live inspection", type="primary"):
            st.session_state.live_camera_running = True
            st.rerun()

    if st.session_state.live_camera_running:
        stream = st.session_state.get("live_camera")
        if stream is None:
            stream = CameraStream(camera_index)
            if not stream.running:
                st.error(f"Could not open camera {camera_index}.")
                st.session_state.live_camera_running = False
                st.stop()
            st.session_state.live_camera = stream

        left, right = st.columns([3, 1])
        image_slot = left.empty()
        count_slot = right.empty()
        fps_slot = right.empty()
        size_slot = right.empty()

        fps_smooth = 0.0
        while st.session_state.live_camera_running:
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            result = predict_image(
                model=model,
                image=frame,
                image_size=320,
                confidence=confidence,
            )
            annotated = cv2.cvtColor(
                result.plot(labels=False, conf=False), cv2.COLOR_BGR2RGB
            )
            elapsed = time.perf_counter() - t0

            fps_smooth = (
                1 / elapsed if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 / elapsed
            )
            count = len(result.boxes) if result.boxes is not None else 0

            image_slot.image(annotated, output_format="JPEG", use_container_width=True)
            count_slot.metric("Detected instances", count)
            fps_slot.metric("Inference speed", f"{fps_smooth:.1f} FPS")
            size_slot.caption(f"Frame: {frame.shape[1]} × {frame.shape[0]}")
    else:
        st.info("Camera is stopped. Start inspection to begin.")
