import os
import sys
import json
# pyrefly: ignore [missing-import]
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # Fallback for systems without ai_edge_litert or tflite_runtime installed
        # during generic linting or lightweight executions
        Interpreter = None

class SleepPredictor:
    """
    Predictor class for Sleep Quality Score using TFLite.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Default path relative to this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "sleep_model", "sleep_scoring_model.tflite")
        
        self.model_path = model_path
        self.interpreter = None
        self._init_model()

    def _init_model(self):
        if Interpreter is None:
            print("Warning: TFLite Interpreter is not available.", file=sys.stderr)
            return

        if not os.path.exists(self.model_path):
            print(f"Warning: Sleep model not found at {self.model_path}", file=sys.stderr)
            return

        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"Error initializing TFLite interpreter: {e}", file=sys.stderr)
            self.interpreter = None

    def predict(self, duration_hours, interruptions, sleep_debt_hours=None):
        # Default sleep_debt_hours to 0.0 if not provided
        if sleep_debt_hours is None:
            sleep_debt_hours = 0.0

        # If interpreter is not loaded, fallback to the simplified rule-based logic
        if self.interpreter is None:
            score = 100.0 - (interruptions * 10)
            if duration_hours < 7.0:
                score -= (7.0 - duration_hours) * 10.0
            score -= sleep_debt_hours * 5.0
            return max(0, min(100, int(round(score))))

        try:
            # Prepare input data as a 1D vector [duration, interruptions, sleep_debt]
            # with shape (1, 3) as expected by the model
            input_data = np.array([[
                float(duration_hours),
                float(interruptions),
                float(sleep_debt_hours)
            ]], dtype=np.float32)

            # Approximate normalization constants based on StandardScaler used during training
            # TODO: Replace with EXACT mean_ and scale_ from the saved scaler in Colab!
            means = np.array([7.2, 3.0, 1.2], dtype=np.float32)
            stds = np.array([1.4076, 2.0, 1.2461], dtype=np.float32)
            
            # Scale the inputs
            scaled_input = (input_data - means) / stds
            
            self.interpreter.set_tensor(self.input_details[0]['index'], scaled_input)
            self.interpreter.invoke()
            
            # Extract output prediction
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            # Output shape is typically [[score]], which is between 0.0 and 1.0
            raw_score = float(output_data[0][0])
            
            # Bound and round score to range 0.0 - 1.0
            return max(0.0, min(1.0, round(raw_score, 2)))
        except Exception as e:
            print(f"Error during TFLite inference: {e}", file=sys.stderr)
            # Fallback
            score = 100.0 - (interruptions * 10)
            if duration_hours < 7.0:
                score -= (7.0 - duration_hours) * 10.0
            score -= sleep_debt_hours * 5.0
            return max(0, min(100, int(round(score))))

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_sleep.py '<json_input_or_file_path>' [model_path]", file=sys.stderr)
        print("Example 1 (Inline JSON): python inference_sleep.py '{\"duration_hours\": 6.5, \"interruptions\": 2, \"sleep_debt_hours\": 1.0}'", file=sys.stderr)
        print("Example 2 (JSON file): python inference_sleep.py sample/sleep/good_sleep.json", file=sys.stderr)
        sys.exit(1)

    input_arg = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load JSON from file or parse inline string
    json_input_str = ""
    if os.path.exists(input_arg):
        try:
            with open(input_arg, "r") as f:
                json_input_str = f.read()
        except Exception as e:
            print(json.dumps({"error": f"Failed to read file: {e}"}))
            sys.exit(1)
    else:
        json_input_str = input_arg

    try:
        data = json.loads(json_input_str)
        duration_hours = float(data.get("duration_hours", 7.0))
        interruptions = int(data.get("interruptions", 0))
        sleep_debt_hours = data.get("sleep_debt_hours")
        if sleep_debt_hours is not None:
            sleep_debt_hours = float(sleep_debt_hours)

        predictor = SleepPredictor(model_path=model_path)
        quality_score = predictor.predict(
            duration_hours=duration_hours,
            interruptions=interruptions,
            sleep_debt_hours=sleep_debt_hours
        )

        output = {
            "quality_score": quality_score
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

