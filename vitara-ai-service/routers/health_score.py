# pyrefly: ignore [missing-import]
from datetime import datetime
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, BackgroundTasks
from schemas.health_score import HealthScoreRequest, HealthScoreResponse
from services.health_score_service import HealthScoreService
from services.llm_companion import memory_store

router = APIRouter(prefix="/health", tags=["Health Score"])

def sync_health_to_rag(user_id: str, response: HealthScoreResponse):
    """
    Helper function to sync calculated health score to ChromaDB as a background task.
    Uses a deterministic daily ID to guarantee idempotency.
    """
    try:
        mood_str = f"Mood={response.breakdown.mood}/100" if response.breakdown.mood is not None else "Mood=Belum tercatat"
        nutrition_str = f"Nutrisi={response.breakdown.nutrition}/100" if response.breakdown.nutrition is not None else "Nutrisi=Belum tercatat"
        stress_str = f"Bebas Stres={response.breakdown.stress}/100" if response.breakdown.stress is not None else "Bebas Stres=Belum tercatat"
        sleep_str = f"Tidur={response.breakdown.sleep}/100" if response.breakdown.sleep is not None else "Tidur=Belum tercatat"

        memory_text = (
            f"Perhitungan Skor Kesehatan Keseluruhan: {response.health_score}/100. "
            f"Rincian dimensi kesehatan: {mood_str}, {nutrition_str}, {stress_str}, {sleep_str}."
        )

        current_date = datetime.now().strftime("%Y-%m-%d")
        deterministic_id = f"health_score_{user_id}_{current_date}"

        memory_store.add_memory(
            user_id=user_id,
            text=memory_text,
            mem_type="health_score",
            doc_id=deterministic_id
        )
    except Exception as e:
        # Menghindari crash response utama jika ChromaDB/Sync bermasalah
        print(f"Error syncing health score to RAG memory: {str(e)}")

@router.post("/score", response_model=HealthScoreResponse)
async def get_health_score(request: HealthScoreRequest, background_tasks: BackgroundTasks):
    """
    Menghitung skor kesehatan keseluruhan secara rule-based (deterministic)
    berdasarkan sub-hasil analisis mood (NLP), nutrisi (makanan), tidur, dan stres pengetikan.
    Mendukung input data parsial dan memproses RAG Sync secara asinkron.
    """
    try:
        response = HealthScoreService.calculate_health_score(request)
        
        # Simpan riwayat perhitungan skor kesehatan ke memori RAG ChromaDB secara asinkron
        if request.user_id:
            background_tasks.add_task(
                sync_health_to_rag,
                user_id=request.user_id,
                response=response
            )
            
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menghitung health score: {str(e)}"
        )

