# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.typing import TypingPredictRequest, TypingPredictResponse
from services.llm_companion import memory_store
from inference_typing import TypingStressPredictor
import os

router = APIRouter(prefix="/predict", tags=["Typing"])

MODEL_PATH = "models/typing_model/typing_stress_lstm.onnx"
predictor = None

def load_typing_model():
    global predictor
    if os.path.exists(MODEL_PATH):
        try:
            predictor = TypingStressPredictor(model_path=MODEL_PATH)
            print("✅ Typing Stress Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading Typing Model: {e}")
    else:
        print(f"⚠️ Warning: Typing Model not found at {MODEL_PATH}. Using mock predictions for now.")

# Inisialisasi model saat module di-import
load_typing_model()

@router.post("/typing", response_model=TypingPredictResponse)
async def predict_typing(request: TypingPredictRequest):
    """
    Mendeteksi tingkat stres berdasarkan pola pengetikan (keystroke dynamics).
    """
    try:
        if predictor is not None:
            # Jalankan inferensi dari model asli
            stress_score = predictor.predict(
                wpm=request.wpm,
                backspace_rate=request.backspace_rate,
                inter_key_timings=request.inter_key_timings
            )
            # Batasi nilai ke range 0.0 - 1.0
            stress_score = max(0.0, min(1.0, float(stress_score)))
        else:
            # Fallback mock jika model tidak termuat
            mock_score = 0.3 + (request.backspace_rate * 0.5) - (min(100.0, request.wpm) / 100.0 * 0.2)
            stress_score = max(0.0, min(1.0, mock_score))
        
        response_data = TypingPredictResponse(stress_score=round(stress_score, 4))
        
        if request.user_id:
            memory_text = (
                f"Analisis pengetikan: Skor stres dinilai {stress_score:.2f} (skala 0.0 - 1.0). "
                f"Detail: Kecepatan {request.wpm} WPM, rasio backspace {request.backspace_rate:.2%}, "
                f"dan pola interval penekanan tombol ({len(request.inter_key_timings)} ketukan)."
            )
            # Daily upsert: simpan sesi pengetikan terbaru hari ini (overwrite sesi sebelumnya)
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            daily_id = f"typing_prediction_{request.user_id}_{current_date}"
            
            memory_store.add_memory(
                user_id=request.user_id,
                text=memory_text,
                mem_type="typing_prediction",
                doc_id=daily_id
            )
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
