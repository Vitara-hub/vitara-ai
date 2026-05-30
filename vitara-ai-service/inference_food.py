import sys
import json
import numpy as np
from PIL import Image
import os

# Gunakan ai_edge_litert (ringan, ARM64 native)
# pyrefly: ignore [missing-import]
from ai_edge_litert.interpreter import Interpreter

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    # Preprocess specifically for MobileNetV2: values scaled to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_food.py <image_path> <tflite_model_path> [classes_txt_path]", file=sys.stderr)
        sys.exit(1)
        
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    classes_path = sys.argv[3] if len(sys.argv) > 3 else "classes.txt"
    
    try:
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    except Exception as e:
        # Fallback dummy classes if file doesn't exist
        classes = [f"food_class_{i}" for i in range(100)]
        
    try:
        # Load TFLite model and allocate tensors.
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Preprocess image
        input_data = load_image(image_path)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # We have two outputs: classification_head and calorie_head
        out_1 = interpreter.get_tensor(output_details[0]['index'])
        out_2 = interpreter.get_tensor(output_details[1]['index'])
        
        # Determine which output is classification and which is calorie based on shape
        if output_details[0]['shape'][-1] == 1:
            calorie_pred = out_1
            class_pred = out_2
        else:
            class_pred = out_1
            calorie_pred = out_2
            
        probabilities = class_pred[0]
        
        # Pengaman: Jika output model berupa raw logits, konversi ke softmax
        if not (0.0 <= np.min(probabilities) <= 1.0) or not (0.9 <= np.sum(probabilities) <= 1.1):
            exp_probs = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_probs / exp_probs.sum()
            
        predicted_class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_idx])
        
        predicted_class_name = classes[predicted_class_idx] if predicted_class_idx < len(classes) else f"class_{predicted_class_idx}"
        
        # Cetak debug ke stderr agar tidak mengganggu parsing JSON di stdout
        sys.stderr.write(f"🔍 Prediksi: {predicted_class_name} | Confidence: {confidence:.4f}\n")
        
        # Ambang batas deteksi makanan (98%)
        CONFIDENCE_THRESHOLD = 0.98
        
        if confidence < CONFIDENCE_THRESHOLD:
            output = {
                "foods": [],
                "estimated_calories": 0
            }
        else:
            # Denormalize calories (MAX_CALORIES = 1000)
            MAX_CALORIES = 1000.0
            estimated_calories = float(calorie_pred[0][0]) * MAX_CALORIES
            
            output = {
                "foods": [predicted_class_name],
                "estimated_calories": round(estimated_calories)
            }
        
        # Only print the JSON payload to stdout
        print(json.dumps(output))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
