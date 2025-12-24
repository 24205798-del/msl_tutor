import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import collections
import time

# Import our new modules
from processor import DataProcessor
from mock_model import MockDualStreamModel

# --- 1. CONFIG & CONSTANTS ---
st.set_page_config(layout="wide", page_title="MSL Edu-Quest")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# --- 2. AI ENGINE SETUP ---
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

# --- 3. DRAWING HELPER ---
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

# --- 4. THE VIDEO PROCESSOR ---
class EduQuestTransformer(VideoProcessorBase):
    def __init__(self):
        self.landmarker = HandLandmarker.create_from_options(options)
        self.model = load_msl_model()
        self.landmark_buffer = collections.deque(maxlen=90)
        
        # STATE VARIABLES
        self.mode = "Learn Mode"   # Default
        self.target_word = ""      # For Quiz Mode
        
        self.current_prediction = "..."
        self.current_confidence = 0.0
        self.weak_joint = None
        self.status_message = "Ready"

    # FUNCTION TO RECEIVE DATA FROM STREAMLIT SIDEBAR
    def update_game_state(self, mode, target_word):
        self.mode = mode
        self.target_word = target_word

    def transform(self, frame):
        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            img_bgr = cv2.flip(img_bgr, 1)
            h, w, _ = img_bgr.shape
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            detection_result = self.landmarker.detect(mp_image)
            
            current_frame_landmarks = []
            img_out = np.copy(img_bgr) # Fallback if no hands

            if detection_result.hand_landmarks:
                annotated_rgb, all_landmarks = draw_landmarks_on_image(img_rgb, detection_result)
                img_out = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                first_hand = all_landmarks[0]
                for lm in first_hand:
                    current_frame_landmarks.append([lm.x, lm.y, lm.z])

                # XAI: Show weak joint ONLY if confidence is low
                if self.weak_joint is not None and self.current_confidence < 0.8:
                    wk = first_hand[self.weak_joint]
                    cx, cy = int(wk.x * w), int(wk.y * h)
                    cv2.circle(img_out, (cx, cy), 15, (0, 0, 255), 3)
                    cv2.putText(img_out, "Check Hand Shape", (cx+20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if current_frame_landmarks:
                self.landmark_buffer.append(current_frame_landmarks)

            # --- LOGIC SPLIT BASED ON MODE ---
            if len(self.landmark_buffer) >= 70:
                raw_sequence = list(self.landmark_buffer)
                processed_input = DataProcessor.gate_and_resample(raw_sequence)
                
                if processed_input is not None:
                    pred, conf, weak_joint = self.model.predict(processed_input)
                    
                    self.current_prediction = pred
                    self.current_confidence = conf
                    self.weak_joint = weak_joint
                    
                    # Logic: Determine what to show user
                    if self.mode == "Quiz Mode":
                        # STRICT VALIDATION
                        if pred.lower() == self.target_word.lower() and conf > 0.8:
                            self.status_message = "✅ CORRECT!"
                            self.weak_joint = None # Clear XAI
                        else:
                            self.status_message = f"❌ Try Again (Saw: {pred})"
                    else:
                        # LEARN MODE (Free Play)
                        self.status_message = f"Detected: {pred}"
                    
                    self.landmark_buffer.clear()

            # --- UI OVERLAY ---
            # Top Bar Background
            cv2.rectangle(img_out, (0, 0), (w, 100), (30, 30, 30), -1)
            
            # 1. Show Mode
            cv2.putText(img_out, f"MODE: {self.mode.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 2. Show Main Status
            # Color logic: Green if correct/high conf, Yellow otherwise
            color = (0, 255, 0) if "✅" in self.status_message else (0, 165, 255)
            cv2.putText(img_out, self.status_message, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            return img_out

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            return frame.to_ndarray(format="bgr24")

# --- 5. STREAMLIT FRONTEND ---
st.sidebar.header("🛠️ Dev Controls")
mode = st.sidebar.radio("Mode", ["Learn Mode", "Quiz Mode"])

# If Quiz Mode, pick a target word
target_word = ""
if mode == "Quiz Mode":
    target_word = st.sidebar.selectbox("Select Target Word", ["nasi lemak", "thank you", "good morning"])

col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"MSL Edu-Quest")
    
    ctx = webrtc_streamer(
        key="msl-quest", 
        video_processor_factory=EduQuestTransformer,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False}
    )
    
    # --- MAGIC BRIDGE: SEND DATA TO PROCESSOR ---
    if ctx.video_processor:
        ctx.video_processor.update_game_state(mode, target_word)

with col2:
    st.info("Instructions")
    if mode == "Quiz Mode":
        st.write(f"### Target: **{target_word.upper()}**")
        st.write("Perform the sign. You must get >80% accuracy.")
    else:
        st.write("### Free Play")
        st.write("Perform any sign to see the AI analysis.")