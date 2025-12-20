import urllib.request
import os

# The official URL from Google/MediaPipe
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
filename = "hand_landmarker.task"

print(f"Downloading {filename}...")

try:
    urllib.request.urlretrieve(url, filename)
    print("Successfully downloaded!")
    print(f"File location: {os.path.abspath(filename)}")
except Exception as e:
    print(f"Error: {e}")