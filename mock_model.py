# mock_model.py
import random
import time

class MockDualStreamModel:
    def __init__(self):
        print("MOCK MODEL: Initialized (Waiting for real weights...)")
        self.gloss_map = {0: "nasi lemak", 1: "thank you", 2: "good morning"}

    def predict(self, input_sequence):
        """
        Simulates the Dual-Stream Network (ST-GCN + TCN).
        Input: (70, 21, 3) array
        Output: (predicted_gloss, confidence_score, weak_joint_index)
        """
        # Simulate inference time (<500ms requirement)
        # time.sleep(0.1) 
        
        # --- MOCK LOGIC ---
        # Randomly decide if the user is "Correct" or "Incorrect" for testing
        confidence = random.uniform(0.6, 0.99)
        predicted_idx = random.choice([0, 1, 2])
        prediction = self.gloss_map[predicted_idx]
        
        # If confidence is low, simulate the "XAI" finding a weak joint
        # (e.g., Index Finger Tip is index 8)
        weak_joint = None
        if confidence < 0.8:
            weak_joint = 8 # Simulate error at Index Finger Tip
            
        return prediction, confidence, weak_joint