import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import collections
from processor import DataProcessor
from mock_model import MockDualStreamModel

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="MSL Translation App")

# Custom CSS to match the "Blue & White" Dashboard look
st.markdown("""
<style>
    /* Main Background - Light Grey */
    .stApp { 
        background-color: #f0f2f6; 
    }
    
    /* Blue Header Bar */
    header[data-testid="stHeader"] {
        background-color: #0052cc;
    }

    /* Text Color Fix - Force Black/Dark Grey */
    h1, h2, h3, h4, h5, h6, p, li, span {
        color: #333333 !important;
    }
    
    /* White Card Containers */
    div.css-1r6slb0.e1tzin5v2, div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    
    /* Hide default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. SETUP & CONSTANTS ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# MediaPipe Setup
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'hand_landmarker.task'

try:
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
except ImportError:
    st.error("MediaPipe Tasks API not found.")

@st.cache_resource
def load_msl_model():
    return MockDualStreamModel()

# --- 3. HELPER FUNCTIONS ---
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    for hand_landmarks in hand_landmarks_list:
        for idx1, idx2 in HAND_CONNECTIONS:
            p1 = hand_landmarks[idx1]
            p2 = hand_landmarks[idx2]
            cv2.line(annotated_image, (int(p1.x * w), int(p1.y * h)), 
                     (int(p2.x * w), int(p2.y * h)), (200, 200, 200), 2)
        for lm in hand_landmarks:
            cv2.circle(annotated_image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                       
    return annotated_image, hand_landmarks_list

# --- 4. VIDEO PROCESSOR ---
class DashboardTransformer(VideoProcessorBase):
    def __init__(self):
        self.landmarker = HandLandmarker.create_from_options(options)
        self.model = load_msl_model()
        self.landmark_buffer = collections.deque(maxlen=90)
        
        # State
        self.mode = "Dashboard Mode" 
        self.current_prediction = "Waiting..."
        self.current_confidence = 0.0
        self.weak_joint = None

    def transform(self, frame):
        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            img_bgr = cv2.flip(img_bgr, 1)
            h, w, _ = img_bgr.shape
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            detection_result = self.landmarker.detect(mp_image)
            
            current_frame_landmarks = []
            img_out = np.copy(img_bgr)

            if detection_result.hand_landmarks:
                annotated_rgb, all_landmarks = draw_landmarks_on_image(img_rgb, detection_result)
                img_out = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                first_hand = all_landmarks[0]
                for lm in first_hand:
                    current_frame_landmarks.append([lm.x, lm.y, lm.z])
                
                # XAI: Red Circle on weak joint
                if self.weak_joint is not None and self.current_confidence < 0.8:
                    wk = first_hand[self.weak_joint]
                    cx, cy = int(wk.x * w), int(wk.y * h)
                    cv2.circle(img_out, (cx, cy), 15, (0, 0, 255), 3)

            if current_frame_landmarks:
                self.landmark_buffer.append(current_frame_landmarks)

            if len(self.landmark_buffer) >= 70:
                raw_sequence = list(self.landmark_buffer)
                processed_input = DataProcessor.gate_and_resample(raw_sequence)
                
                if processed_input is not None:
                    pred, conf, weak_joint = self.model.predict(processed_input)
                    self.current_prediction = pred
                    self.current_confidence = conf
                    self.weak_joint = weak_joint
                    self.landmark_buffer.clear()

            # --- OVERLAY DESIGN ---
            # White bottom bar
            cv2.rectangle(img_out, (0, h-60), (w, h), (255, 255, 255), -1) 
            
            # Confidence Bar (Visualized as a line)
            bar_width = int(w * self.current_confidence)
            cv2.rectangle(img_out, (0, h-5), (bar_width, h), (0, 120, 255), -1) # Orange progress line
            
            # Text
            text = f"{self.current_prediction.upper()} ({int(self.current_confidence*100)}%)"
            cv2.putText(img_out, text, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            return img_out

        except Exception as e:
            print(f"Error: {e}")
            return frame.to_ndarray(format="bgr24")

# --- 5. MAIN DASHBOARD UI ---

st.markdown("## ✋ Demo")
st.markdown("#### Malaysian Sign Language Translation Case Study")
st.markdown("---")

# Layout: Left (Controls), Center (Video), Right (Info)
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- LEFT COLUMN: CONTROLS ---
with col_left:
    st.markdown("### 🎛 Controls")
    
    # Primary Action Buttons
    st.button("📸 Capture", type="primary")
    st.button("📎 Upload", type="secondary")
    
    st.markdown("---")
    
    # Settings Area
    st.markdown("**Settings**")
    language = st.selectbox("Language", ["English (UK)", "Bahasa Melayu"])
    mode = st.radio("Mode", ["Translation", "Learning"])
    
    st.markdown("---")
    
    # Big Metric Display
    st.markdown("**Confidence**")
    st.metric(label="Model Confidence", value="85%", delta="Live")

# --- CENTER COLUMN: VIDEO FEED ---
with col_center:
    st.info("💡 **MSL Translation App** - Active")
    
    # The WebRTC Component
    ctx = webrtc_streamer(
        key="msl-dashboard", 
        video_processor_factory=DashboardTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Static UI to match screenshot "Prediction Results"
    st.markdown("### 📊 Prediction Results")
    
    st.write("**Success**")
    st.progress(85)
    
    st.write("**Neutral**")
    st.progress(10)
    
    st.write("**Fail**")
    st.progress(5)

# --- RIGHT COLUMN: FEATURES INFO ---
with col_right:
    st.markdown("### 🔑 Key Demo Features")
    
    st.markdown("""
    **⚙️ MediaPipe Preprocessing** Real-time skeleton tracking and normalization of hand landmarks.
    
    **🧠 ST-GCN Model Prediction** Dual stream architecture processing morphology and trajectory data.
    
    **🔍 Explainable AI** Heatmap visualization showing which parts of the sign contribute most.
    
    **🚀 Performance Metrics** Real-time feedback on prediction confidence.
    
    **💡 Future Improvements** Enhanced vocabulary and improved lighting resilience.
    """)

# Footer
st.markdown("---")
# UPDATED: Changed from Edu-Quest to MSL Translation App
st.caption("MSL Translation App Prototype v1.0 | Powered by Streamlit & MediaPipe")