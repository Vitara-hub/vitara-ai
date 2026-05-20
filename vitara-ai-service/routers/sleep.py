# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.sleep import SleepPredictRequest, SleepPredictResponse
from services.llm_companion import memory_store

router = APIRouter(prefix="/predict", tags=["Sleep"])

@router.post("/sleep", response_model=SleepPredictResponse)
async def predict_sleep(request: SleepPredictRequest):
    """
    Menghitung skor kualitas tidur pengguna berdasarkan data tidur (menggunakan mock logic).
    """
    try:
        # Mock formula:
        # Base score 100
        # Deduct 10 points for each interruption
        # Deduct 10 points for each hour of sleep below 7 hours
        # Deduct 5 points for sleep debt (if provided)
        score = 100.0 - (request.interruptions * 10)
        
        if request.duration_hours < 7.0:
            score -= (7.0 - request.duration_hours) * 10.0
            
        if request.sleep_debt_hours is not None:
            score -= request.sleep_debt_hours * 5.0
            
        # Bound score to range 0 - 100
        quality_score = max(0, min(100, int(round(score))))
        
        response_data = SleepPredictResponse(quality_score=quality_score)
        
        if request.user_id:
            memory_text = (
                f"Analisis tidur: Kualitas tidur dinilai {quality_score}/100. "
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
