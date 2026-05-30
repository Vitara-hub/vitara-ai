# pyrefly: ignore [missing-import]
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from schemas.food import FoodResponse
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
import io
import os
from services.llm_companion import memory_store

# Gunakan ai_edge_litert (lebih ringan untuk Mac/ARM)
# pyrefly: ignore [missing-import]  
from ai_edge_litert.interpreter import Interpreter

router = APIRouter(prefix="/predict", tags=["Food"])

MODEL_PATH = "models/vision_model/vision_model.tflite"
CLASSES_PATH = "models/vision_model/classes.txt"

interpreter = None
classes = []

def load_food_model():
    global interpreter, classes
    if os.path.exists(MODEL_PATH):
        try:
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            if os.path.exists(CLASSES_PATH):
                with open(CLASSES_PATH, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]
            
            print("✅ Food Vision Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading Food Model: {e}")
    else:
        print(f"⚠️ Warning: Food Model or Interpreter not found.")

# Load model saat module di-import
load_food_model()

def preprocess_image(image_bytes):
    """
    Preprocessing gambar: Resize ke 224x224 dan normalisasi ke range [-1, 1]
    sesuai dengan requirement MobileNetV2.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

@router.post("/food", response_model=FoodResponse)
async def predict_food(
    image: UploadFile = File(...),
    user_id: str = Form(None)
):
    """
    Menerima upload gambar makanan dan mengembalikan prediksi jenis makanan serta estimasi kalori.
    """
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Food model is not available.")
    
    # Validasi format file
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Hanya mendukung file JPEG atau PNG.")

    try:
        # 1. Baca dan preprocess gambar
        contents = await image.read()
        input_data = preprocess_image(contents)
        
        # 2. Jalankan inferensi TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 3. Ambil output (model memiliki 2 output head: classification & calorie)
        out_1 = interpreter.get_tensor(output_details[0]['index'])
        out_2 = interpreter.get_tensor(output_details[1]['index'])
        
        # Identifikasi mana yang merupakan classification head berdasarkan shape
        if output_details[0]['shape'][-1] == 1:
            calorie_pred, class_pred = out_1, out_2
        else:
            class_pred, calorie_pred = out_1, out_2
            
        # 4. Post-processing hasil
        probabilities = class_pred[0]
        
        # Pengaman: Jika output model berupa raw logits (tidak menjumlah ke 1.0),
        # kita konversi ke softmax secara manual agar nilainya berada di range [0, 1]
        if not (0.0 <= np.min(probabilities) <= 1.0) or not (0.9 <= np.sum(probabilities) <= 1.1):
            exp_probs = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_probs / exp_probs.sum()
            
        predicted_class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_idx])
        
        print(f"🔍 Prediksi: {classes[predicted_class_idx] if predicted_class_idx < len(classes) else 'Unknown'} | Confidence: {confidence:.4f}")
        
        # Ambang batas deteksi makanan (98%)
        CONFIDENCE_THRESHOLD = 0.98
        
        if confidence < CONFIDENCE_THRESHOLD:
            # Mengembalikan respon kosong dan melewati penyimpanan memori
            return FoodResponse(
                foods=[],
                estimated_calories=0
            )
            
        predicted_class_name = classes[predicted_class_idx] if predicted_class_idx < len(classes) else "Unknown"
        
        # Denormalisasi kalori (MAX_CALORIES = 1000 sesuai training)
        MAX_CALORIES = 1000.0
        estimated_calories = float(calorie_pred[0][0]) * MAX_CALORIES
        
        response_data = FoodResponse(
            foods=[predicted_class_name],
            estimated_calories=round(estimated_calories)
        )
        
        if user_id:
            memory_store.add_memory(
                user_id=user_id,
                text=f"Analisis makanan: Terdeteksi makanan '{', '.join(response_data.foods)}' dengan perkiraan energi sebesar {response_data.estimated_calories} kkal.",
                mem_type="food_prediction"
            )
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

