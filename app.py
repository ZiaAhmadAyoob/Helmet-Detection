import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Site Safety AI",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Safety Theme) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=80) 
with col2:
    st.title("Industrial Safety Compliance System")
    st.markdown("**Real-time PPE & Helmet Detection** | Powered by YOLOv8")

st.markdown("---")

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Safety Config")
st.sidebar.subheader("Detection Parameters")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.40, 
    step=0.01,
    help="Higher values reduce false positives but might miss some helmets."
)

app_mode = st.sidebar.selectbox(
    "Select Input Source",
    ["Image Audit", "Video Footage Analysis", "Live Site Feed"]
)

st.sidebar.markdown("---")
st.sidebar.info("Ensure your trained 'best.pt' is in the root directory.")

# --- Model Loading (Absolute Path Fix) ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get absolute path to ensure file is found
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"⚠️ Model file not found at: {model_path}")
    st.info("Please rename your model file to 'best.pt' and place it in the same folder as this script.")
    model = None

# --- Main Logic ---

if model:
    # ---------------- IMAGE AUDIT MODE ----------------
    if app_mode == "Image Audit":
        st.subheader("📸 Site Image Audit")
        uploaded_file = st.file_uploader("Upload site image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                # Updated parameter here
                st.image(image, caption="Uploaded Site Photo", use_container_width=True)

            with col2:
                if st.button("Run Safety Check", type="primary"):
                    with st.spinner("Scanning for safety gear..."):
                        results = model(img_array, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        
                        # Updated parameter here
                        st.image(annotated_img, caption="Compliance Result", use_container_width=True)
                        
                        count = len(results[0].boxes)
                        st.metric(label="Objects/Persons Detected", value=count)
                        if count > 0:
                            st.success("Detection Completed.")
                        else:
                            st.warning("No helmets/people detected.")

    # ---------------- VIDEO ANALYSIS MODE ----------------
    elif app_mode == "Video Footage Analysis":
        st.subheader("🎥 CCTV Footage Analysis")
        video_file = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov"])

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            st.sidebar.markdown("---")
            stop_button = st.sidebar.button("Stop Processing")

            col1, col2 = st.columns([3, 1])
            with col1:
                stframe = st.empty()
            with col2:
                st.markdown("### Live Stats")
                kpi_text = st.empty()
                st.info("Processing...")

            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
                    break

                results = model(frame, conf=conf_threshold)
                annotated = results[0].plot()
                
                # Updated parameter here
                stframe.image(annotated, channels="BGR", use_container_width=True)
                
                obj_count = len(results[0].boxes)
                kpi_text.markdown(f"**Detected Objects:** {obj_count}")

            cap.release()
            st.success("Analysis Finished.")

    # ---------------- LIVE SITE FEED (WEBCAM) ----------------
    elif app_mode == "Live Site Feed":
        st.subheader("🔴 Real-Time Site Monitoring")
        st.write("Connect to local camera for real-time compliance checks.")

        run = st.checkbox('Activate Safety Feed', value=False)
        
        frame_window = st.image([])
        cap = cv2.VideoCapture(0)

        if run:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera error.")
                    break
                
                results = model(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Updated parameter implicit in st.image but safe to add if creating new elements
                frame_window.image(annotated_frame)
        else:
            cap.release()
            st.write("Feed is offline.")