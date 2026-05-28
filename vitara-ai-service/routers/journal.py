# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.journal import JournalRequest, JournalResponse
# pyrefly: ignore [missing-import]
import onnxruntime as ort
import os
# Supress PyTorch warning from transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer
import numpy as np
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

# Path ke model ONNX dan tokenizer
MODEL_PATH = "models/nlp_model/vitara_nlp_indobert.onnx"
TOKENIZER_PATH = "models/nlp_model/tokenizer"
model = None
tokenizer = None

def load_nlp_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        try:
            model = ort.InferenceSession(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            print("✅ NLP ONNX Model and Tokenizer loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading NLP Model or Tokenizer: {e}")
    else:
        print(f"⚠️ Warning: NLP Model/Tokenizer not found. Expected model at {MODEL_PATH} and tokenizer at {TOKENIZER_PATH}. Using mock predictions for now.")

# Inisialisasi model saat module di-import
load_nlp_model()

# Inisialisasi Google GenAI Client
api_key = os.getenv("GEMINI_API_KEY")
is_genai_configured = api_key is not None and api_key != "your_gemini_api_key_here"
genai_client = genai.Client(api_key=api_key) if is_genai_configured else None

# Model untuk ekstraksi topik
TOPIC_MODEL_NAME = os.getenv("TOPIC_MODEL_NAME", "gemma-4-31b-it")

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
    
    if model is None or tokenizer is None:
        # Fallback ke mock data jika model belum tersedia di local
        response_data = JournalResponse(
            emotion="neutral",
            stress_level=0.5,
            topics=extracted_topics
        )
        if request.user_id:
            # Save raw journal text for semantic similarity matching
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Pengguna menulis jurnal: \"{request.text}\"",
                mem_type="user_journal"
            )
            # Save NLP analysis result as structured health context
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Analisis jurnal: Emosi terdeteksi adalah '{response_data.emotion}' dengan tingkat stres sebesar {response_data.stress_level:.2f}. Topik jurnal: {', '.join(response_data.topics)}. (Model ML Non-aktif)",
                mem_type="nlp_prediction"
            )
        return response_data
    
    try:
        # Tokenisasi input teks
        encoded = tokenizer(
            request.text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        
        # Buat feed dict dengan tipe int32
        feed_dict = {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32),
            "token_type_ids": encoded["token_type_ids"].astype(np.int32),
        }
        
        # Jalankan model ONNX
        outputs = model.run(["emotion_output", "stress_output"], feed_dict)
        
        emotion_probs = outputs[0][0]  # shape (5,)
        stress_val = float(outputs[1][0][0])  # shape (1,) -> float
        
        # Pemetaan label emosi (urutan alfabetis)
        EMOTION_LABELS = ["angry", "anxious", "happy", "neutral", "sad"]
        predicted_emotion_idx = int(np.argmax(emotion_probs))
        predicted_emotion = EMOTION_LABELS[predicted_emotion_idx]
        
        # Batasi nilai tingkat stres ke rentang 0.0 - 1.0
        stress_val = max(0.0, min(1.0, stress_val))
        
        response_data = JournalResponse(
            emotion=predicted_emotion,
            stress_level=round(stress_val, 4),
            topics=extracted_topics
        )
        
        if request.user_id:
            # Save raw journal text for semantic similarity matching
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Pengguna menulis jurnal: \"{request.text}\"",
                mem_type="user_journal"
            )
            # Save NLP analysis result as structured health context
            memory_store.add_memory(
                user_id=request.user_id,
                text=f"Analisis jurnal: Emosi terdeteksi adalah '{response_data.emotion}' dengan tingkat stres sebesar {response_data.stress_level:.2f}. Topik jurnal: {', '.join(response_data.topics)}.",
                mem_type="nlp_prediction"
            )
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


