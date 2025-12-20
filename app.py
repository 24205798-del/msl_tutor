import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. SETUP: New MediaPipe Tasks API ---
# Path to the model file you just downloaded
model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# We create the "Landmarker" (the brain)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, # Streamlit works best with IMAGE mode in loops
    num_hands=1
)

# --- 2. STREAMLIT UI ---
st.title("🤟 Sign Language Tutor (New API)")
run = st.checkbox('Open Camera')
FRAME_WINDOW = st.image([])

# Helper to draw landmarks (The new API doesn't have a simple drawing tool yet, 
# so we use this common helper)
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    for landmarks in hand_landmarks_list:
        for landmark in landmarks:
            x = int(landmark.x * rgb_image.shape[1])
            y = int(landmark.y * rgb_image.shape[0])
            cv2.circle(rgb_image, (x, y), 5, (0, 255, 0), -1)
    return rgb_image

# --- 3. THE LOOP ---
with HandLandmarker.create_from_options(options) as landmarker:
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert to RGB and then to MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process the image
        landmarker_result = landmarker.detect(mp_image)

        # Draw the result
        if landmarker_result.hand_landmarks:
            annotated_image = draw_landmarks_on_image(rgb_frame, landmarker_result)
            FRAME_WINDOW.image(annotated_image)
        else:
            FRAME_WINDOW.image(rgb_frame)

    camera.release()