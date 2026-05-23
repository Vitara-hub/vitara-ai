import os
import sys
import json
import numpy as np
# pyrefly: ignore [missing-import]
import onnxruntime as ort

class TypingStressPredictor:
    """
    Predictor class for Typing Stress LSTM model using ONNX Runtime.
    Loads the ONNX model and preprocesses incoming inputs matching the API contract
    to generate the final stress score.
    """
    def __init__(self, model_path=None, max_seq_len=50):
        # The sequential input shape of our trained model is (None, 50, 1), so max_seq_len is 50.
        self.max_seq_len = max_seq_len
        
        # Determine default model path if not specified
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "typing_model", "typing_stress_lstm.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Typing stress model not found at: {model_path}")
            
        print(f"Loading typing stress model from: {model_path}", file=sys.stderr)
        
        # Initialize ONNX inference session
        self.session = ort.InferenceSession(model_path)
        
        # Identify input names dynamically by shape
        inputs = self.session.get_inputs()
        self.seq_input_name = None
        self.static_input_name = None
        
        for inp in inputs:
            shape = list(inp.shape)
            # seq_input shape is [None, 50, 1] or similar (second dimension is max_seq_len)
            if len(shape) == 3 and shape[1] == self.max_seq_len:
                self.seq_input_name = inp.name
            # static_input shape is [None, 3] or similar (second dimension is 3)
            elif len(shape) == 2 and shape[1] == 3:
                self.static_input_name = inp.name
                
        # Robust fallback if shape matching fails
        if self.seq_input_name is None or self.static_input_name is None:
            if len(inputs) >= 2:
                if "seq" in inputs[0].name.lower() or "input_1" in inputs[0].name.lower():
                    self.seq_input_name = inputs[0].name
                    self.static_input_name = inputs[1].name
                else:
                    self.seq_input_name = inputs[1].name
                    self.static_input_name = inputs[0].name
            else:
                raise ValueError("Expected at least 2 input layers in the ONNX model.")
                
        self.output_name = self.session.get_outputs()[0].name
        
        # Robust fallback scaling parameters (Standardization: (x - mean) / std)
        # Calibrated using actual training set distributions
        self.seq_mean = 560.21773
        self.seq_std = 4842.44977
        
        self.static_means = np.array([42.33, 0.206, 0.041], dtype=np.float32)  # wpm, typing_variance, backspace_rate
        self.static_stds = np.array([10.72, 0.455, 0.014], dtype=np.float32)

    def preprocess(self, wpm, backspace_rate, inter_key_timings):
        """
        Preprocesses raw features to match model inputs:
        1. Calculates typing variance from inter_key_timings (in seconds squared).
        2. Pads inter_key_timings sequence to max_seq_len.
        3. Scales sequence and static features using standardization.
        """
        # 1. Parse inter_key_timings if it's a comma-separated string
        if isinstance(inter_key_timings, str):
            try:
                inter_key_timings = [float(x.strip()) for x in inter_key_timings.split(',') if x.strip()]
            except ValueError:
                inter_key_timings = []
                
        if not inter_key_timings:
            inter_key_timings = [0.0]

        # 2. Calculate typing variance dynamically (standard deviation of timing intervals in seconds)
        if len(inter_key_timings) > 1:
            typing_variance = float(np.std(np.array(inter_key_timings) / 1000.0))
        else:
            typing_variance = 0.0

        # 3. Process Sequence Input (inter_key_timings)
        seq_len = len(inter_key_timings)
        if seq_len > self.max_seq_len:
            padded_seq = np.array(inter_key_timings[:self.max_seq_len], dtype=np.float32)
        else:
            padded_seq = np.pad(
                inter_key_timings, 
                (0, self.max_seq_len - seq_len), 
                mode='constant', 
                constant_values=0.0
            ).astype(np.float32)

        # Scale sequence (Standardization), but keep padded values (0.0) as 0.0 to respect Masking layer
        mask = padded_seq != 0.0
        scaled_seq = np.zeros_like(padded_seq)
        scaled_seq[mask] = (padded_seq[mask] - self.seq_mean) / self.seq_std
        
        # Reshape to expected input shape: (batch_size, time_steps, features) -> (1, max_seq_len, 1)
        X_seq = np.expand_dims(np.expand_dims(scaled_seq, axis=-1), axis=0)

        # 4. Process Static Input (wpm, typing_variance, backspace_rate)
        static_features = np.array([wpm, typing_variance, backspace_rate], dtype=np.float32)
        scaled_static = (static_features - self.static_means) / self.static_stds
        
        # Reshape to expected input shape: (batch_size, num_features) -> (1, 3)
        X_static = np.expand_dims(scaled_static, axis=0)

        return X_seq.astype(np.float32), X_static.astype(np.float32)

    def predict(self, wpm, backspace_rate, inter_key_timings):
        """
        Generates typing stress probability score.
        """
        X_seq, X_static = self.preprocess(wpm, backspace_rate, inter_key_timings)
        
        # Run inference
        ort_inputs = {
            self.seq_input_name: X_seq,
            self.static_input_name: X_static
        }
        prediction = self.session.run([self.output_name], ort_inputs)
        stress_score = float(prediction[0][0][0])
        
        return stress_score

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_typing.py '<json_input>' [model_path]", file=sys.stderr)
        print("Example: python inference_typing.py '{\"wpm\": 58.3, \"backspace_rate\": 0.12, \"inter_key_timings\": [120, 98, 145, 87, 203, 110]}'", file=sys.stderr)
        sys.exit(1)

    json_input_str = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        data = json.loads(json_input_str)
        wpm = float(data.get("wpm", 0.0))
        backspace_rate = float(data.get("backspace_rate", 0.0))
        inter_key_timings = data.get("inter_key_timings", [])

        # Initialize predictor (using max_seq_len = 50 as defined in the compiled model structure)
        predictor = TypingStressPredictor(model_path=model_path, max_seq_len=50)
        
        # Run prediction
        stress_score = predictor.predict(
            wpm=wpm,
            backspace_rate=backspace_rate,
            inter_key_timings=inter_key_timings
        )

        output = {
            "stress_score": round(stress_score, 4)
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
