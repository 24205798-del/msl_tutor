import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. AI SETUP ---
model_path = 'hand_landmarker.task'

# Create the landmarker once outside the loop
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# --- 2. THE VIDEO PROCESSOR ---
# This class handles the camera feed from the user's browser
class MSLTransformer(VideoTransformerBase):
    def __init__(self):
        self.landmarker = HandLandmarker.create_from_options(options)

    def transform(self, frame):
        # Convert the frame to a format OpenCV/MediaPipe understands
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror the image (better for user experience)
        img = cv2.flip(img, 1)
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run AI detection
        result = self.landmarker.detect(mp_image)

        # Draw dots if hand is found
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for lm in landmarks:
                    x = int(lm.x * img.shape[1])
                    y = int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                    
            # Placeholder for your MSL logic
            cv2.putText(img, "Hand Detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# --- 3. STREAMLIT UI ---
st.title("🤟 Malaysian Sign Language Tutor")
st.write("Work is still in progress. Stay tuned for updates!")

webrtc_streamer(
    key="msl-tutor", 
    video_transformer_factory=MSLTransformer,
    # This line ensures the connection is stable on the web:
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)