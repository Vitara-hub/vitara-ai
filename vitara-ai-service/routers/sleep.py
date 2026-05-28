# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.sleep import SleepPredictRequest, SleepPredictResponse
from services.llm_companion import memory_store
from inference_sleep import SleepPredictor

router = APIRouter(prefix="/predict", tags=["Sleep"])

# Instantiate the predictor once
predictor = SleepPredictor()

@router.post("/sleep", response_model=SleepPredictResponse)
async def predict_sleep(request: SleepPredictRequest):
    """
    Menghitung skor kualitas tidur pengguna berdasarkan data tidur menggunakan model ML TFLite.
    """
    try:
        # Predict using TFLite ML model
        quality_score = predictor.predict(
            duration_hours=request.duration_hours,
            interruptions=request.interruptions,
            sleep_debt_hours=request.sleep_debt_hours
        )
        
        response_data = SleepPredictResponse(quality_score=quality_score)
        
        if request.user_id:
            memory_text = (
                f"Analisis tidur: Kualitas tidur dinilai {quality_score:.2f}. "
                f"Detail: Durasi {request.duration_hours} jam, waktu tidur {request.bedtime} - {request.wake_time}, "
                f"dengan {request.interruptions} kali terbangun di malam hari."
            )
            if request.sleep_debt_hours is not None:
                memory_text += f" Utang tidur: {request.sleep_debt_hours} jam."
            
            # Daily upsert: hanya 1 entri tidur per hari (overwrite jika dipanggil lagi)
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            daily_id = f"sleep_prediction_{request.user_id}_{current_date}"
            
            memory_store.add_memory(
                user_id=request.user_id,
                text=memory_text,
                mem_type="sleep_prediction",
                doc_id=daily_id
            )

            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating sleep score: {str(e)}")

