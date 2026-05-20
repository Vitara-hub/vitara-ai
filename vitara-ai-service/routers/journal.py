# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.journal import JournalRequest, JournalResponse
import tensorflow as tf
from models.custom_layers import AttentionLayer
from models.custom_losses import WeightedFocalLoss
import os
from services.llm_companion import memory_store
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List
import json

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

# Inisialisasi Google GenAI Client
api_key = os.getenv("GEMINI_API_KEY")
is_genai_configured = api_key is not None and api_key != "your_gemini_api_key_here"
genai_client = genai.Client(api_key=api_key) if is_genai_configured else None

# Model untuk ekstraksi topik
TOPIC_MODEL_NAME = os.getenv("TOPIC_MODEL_NAME", "gemma-4-31b")

class TopicExtractionResponse(BaseModel):
    topics: List[str] = Field(description="List of 1 to 3 keywords/topics extracted from the text.")

# Kamus pemetaan kata kunci ke topik
TOPIC_MAP = {
    "kerja": ["kerja", "proyek", "deadline", "tugas", "kantor", "bos", "atasan", "lembur", "karir"],
    "keluarga": ["orang tua", "ibu", "ayah", "anak", "suami", "istri", "keluarga", "rumah"],
    "akademik": ["kuliah", "sekolah", "skripsi", "dosen", "ujian", "tugas", "pelajaran", "nilai"],
    "sosial": ["teman", "sahabat", "pacar", "hubungan", "sosialisasi", "rekan", "pesta", "kumpul"],
    "keuangan": ["uang", "keuangan", "gaji", "tabungan", "belanja", "utang", "biaya", "investasi"],
    "kesehatan": ["sakit", "dokter", "obat", "lelah", "pusing", "tidur", "olahraga", "fisik", "diet"]
}

def extract_topics_fallback(text: str) -> List[str]:
    text_lower = text.lower()
    detected_topics = []
    for topic, keywords in TOPIC_MAP.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_topics.append(topic)
                break
    return detected_topics if detected_topics else ["umum"]

async def extract_topics_via_gemma(text: str) -> List[str]:
    if not is_genai_configured or not genai_client:
        return extract_topics_fallback(text)
    
    prompt = (
        f"Ekstrak 1 sampai 3 topik utama (masing-masing maksimal 2 kata) "
        f"dalam Bahasa Indonesia dari teks jurnal berikut:\n\n"
        f"'{text}'\n\n"
        f"Berikan output hanya dalam format JSON yang sesuai dengan skema."
    )
    
    try:
        response = await genai_client.aio.models.generate_content(
            model=TOPIC_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TopicExtractionResponse,
                temperature=0.1
            )
        )
        parsed = json.loads(response.text)
        topics = parsed.get("topics", [])
        return topics if topics else extract_topics_fallback(text)
    except Exception as e:
        print(f"⚠️ Error extracting topics via model '{TOPIC_MODEL_NAME}': {e}. Falling back to rule-based.")
        return extract_topics_fallback(text)

@router.post("/journal", response_model=JournalResponse)
async def predict_journal(request: JournalRequest):
    """
    Menganalisis teks jurnal untuk mendeteksi emosi, tingkat stres, dan topik.
    """
    # Ekstrak topik menggunakan Gemma/Fallback secara dinamis
    extracted_topics = await extract_topics_via_gemma(request.text)
    
    if model is None:
        # Fallback ke mock data jika model belum tersedia di local
        response_data = JournalResponse(
            emotion="neutral",
            stress_level=0.5,
            topics=extracted_topics
        )
        if request.user_id:
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Analisis jurnal: Emosi terdeteksi adalah '{response_data.emotion}' dengan tingkat stres sebesar {response_data.stress_level:.2f}. Topik jurnal: {', '.join(response_data.topics)}. (Model ML Non-aktif)",
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
            topics=extracted_topics
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


