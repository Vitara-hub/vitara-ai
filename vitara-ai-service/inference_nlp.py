import os
import sys
import json
import numpy as np
# pyrefly: ignore [missing-import]
import onnxruntime as ort

# Supress PyTorch warning from transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

class NLPStressPredictor:
    """
    Predictor class for IndoBERT Emotion/Stress model using ONNX Runtime.
    """
    def __init__(self, model_path=None, tokenizer_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "nlp_model", "vitara_nlp_indobert.onnx")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(base_dir, "models", "nlp_model", "tokenizer")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NLP model not found at: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"NLP tokenizer not found at: {tokenizer_path}")
            
        print(f"Loading NLP model from: {model_path}", file=sys.stderr)
        self.session = ort.InferenceSession(model_path)
        print(f"Loading Tokenizer from: {tokenizer_path}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        self.emotion_labels = ["angry", "anxious", "happy", "neutral", "sad"]

    def predict(self, text: str):
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        
        feed_dict = {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32),
            "token_type_ids": encoded["token_type_ids"].astype(np.int32),
        }
        
        outputs = self.session.run(["emotion_output", "stress_output"], feed_dict)
        emotion_probs = outputs[0][0]
        stress_val = float(outputs[1][0][0])
        
        predicted_emotion_idx = np.argmax(emotion_probs)
        predicted_emotion = self.emotion_labels[predicted_emotion_idx]
        stress_val = max(0.0, min(1.0, stress_val))
        
        return {
            "emotion": predicted_emotion,
            "stress_level": round(stress_val, 4),
            "emotion_probabilities": {
                self.emotion_labels[i]: round(float(prob), 4) for i, prob in enumerate(emotion_probs)
            }
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_nlp.py '<text_to_analyze_or_json_file>' [model_path] [tokenizer_path]", file=sys.stderr)
        print("Example: python inference_nlp.py 'Hari ini saya merasa sangat cemas karena tugas menumpuk.'", file=sys.stderr)
        print("Example: python inference_nlp.py sample/nlp/happy_journal.json", file=sys.stderr)
        sys.exit(1)

    text_arg = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    tokenizer_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Check if the argument is a JSON file and load the text from it
    if os.path.exists(text_arg) and text_arg.endswith('.json'):
        try:
            with open(text_arg, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'text' in data:
                    text = data['text']
                else:
                    print(f"Error: JSON file {text_arg} must contain a 'text' key", file=sys.stderr)
                    sys.exit(1)
        except Exception as e:
            print(f"Error reading JSON file {text_arg}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = text_arg

    try:
        predictor = NLPStressPredictor(model_path=model_path, tokenizer_path=tokenizer_path)
        result = predictor.predict(text)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
