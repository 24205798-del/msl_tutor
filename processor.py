import numpy as np

class DataProcessor:
    @staticmethod
    def gate_and_resample(landmark_history, target_frames=70):
        try:
            # Safety Check 1: Ensure we have a list
            if not landmark_history:
                return None
            
            # Safety Check 2: Convert to NumPy and check shape
            # We expect shape (N, 21, 3). If data is jagged, this will fail or result in shape (N,)
            data = np.array(landmark_history)
            
            if data.ndim != 3:
                print(f"PROCESSOR ERROR: Invalid shape {data.shape}. Expected (frames, 21, 3).")
                return None

            if len(data) < 10: 
                return None

            curr_frames = data.shape[0]
            num_landmarks = data.shape[1] # Should be 21
            dims = data.shape[2] # Should be 3

            original_indices = np.linspace(0, curr_frames - 1, num=curr_frames)
            target_indices = np.linspace(0, curr_frames - 1, num=target_frames)
            
            resampled_data = np.zeros((target_frames, num_landmarks, dims))

            for lm_idx in range(num_landmarks):
                for dim in range(dims):
                    resampled_data[:, lm_idx, dim] = np.interp(
                        target_indices, 
                        original_indices, 
                        data[:, lm_idx, dim]
                    )
            
            return resampled_data

        except Exception as e:
            print(f"PROCESSOR CRASH: {e}")
            return None