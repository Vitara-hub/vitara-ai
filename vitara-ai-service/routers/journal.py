# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.journal import JournalRequest, JournalResponse
import tensorflow as tf
from models.custom_layers import AttentionLayer
from models.custom_losses import WeightedFocalLoss
import os
from services.llm_companion import memory_store

router = APIRouter(prefix="/predict", tags=["Journal"])

# Path ke model (sesuaikan jika nama file berbeda)
MODEL_PATH = "models/nlp_model/nlp_model.keras"
model = None

def load_nlp_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    "AttentionLayer": AttentionLayer,
                    "WeightedFocalLoss": WeightedFocalLoss
                }
            )
            print("✅ NLP Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading NLP Model: {e}")
    else:
        print(f"⚠️ Warning: NLP Model not found at {MODEL_PATH}. Using mock predictions for now.")

# Inisialisasi model saat module di-import
load_nlp_model()

@router.post("/journal", response_model=JournalResponse)
async def predict_journal(request: JournalRequest):
    """
    Menganalisis teks jurnal untuk mendeteksi emosi, tingkat stres, dan topik.
    """
    if model is None:
        # Fallback ke mock data jika model belum tersedia di local
        response_data = JournalResponse(
            emotion="neutral",
            stress_level=0.5,
            topics=["unknown"]
        )
        if request.user_id:
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Analisis jurnal: Emosi terdeteksi adalah '{response_data.emotion}' dengan tingkat stres sebesar {response_data.stress_level:.2f}. (Data Uji Coba)",
                mem_type="nlp_prediction"
            )
        return response_data
    
    try:
        # TODO: Implementasi preprocessing (Tokenization/Padding) jika tidak termasuk dalam model
        # prediction = model.predict([request.text])
        
        # Placeholder sementara menunggu koordinasi format output model dari Putri
        response_data = JournalResponse(
            emotion="anxious",
            stress_level=0.82,
            topics=["deadline", "kerja"]
        )
        if request.user_id:
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Analisis jurnal: Emosi terdeteksi adalah '{response_data.emotion}' dengan tingkat stres sebesar {response_data.stress_level:.2f}. Topik jurnal: {', '.join(response_data.topics)}.",
                mem_type="nlp_prediction"
            )
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

